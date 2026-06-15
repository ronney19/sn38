#!/usr/bin/env python3
"""
Generate benchmark items for SN38 leak detection using an LLM.

Each item tests whether a ChronoGPT model for year Y "knows" about
an event that happened AFTER year Y. A well-trained model should NOT
know these facts → very negative log-prob scores (good).

Usage:
    # Generate items for a single year (prints to stdout)
    OPENAI_API_KEY=sk-xxx python scripts/generate_benchmarks.py --year 2019

    # Generate items for all years and write to a JSON file
    OPENAI_API_KEY=sk-xxx python scripts/generate_benchmarks.py --all --output my_benchmarks.json

    # Merge generated items into existing benchmark file
    OPENAI_API_KEY=sk-xxx python scripts/generate_benchmarks.py --all --merge scripts/sample_benchmarks.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sn38.template.constants import ALL_YEARS

SYSTEM_PROMPT = """You are helping build a temporal knowledge benchmark for language models.

Your task: given a cutoff year Y, generate fill-in-the-blank benchmark items that test
whether a language model knows about events that happened STRICTLY AFTER year Y.

Rules:
1. Every event must have occurred in year Y+1 or later. Never include pre-cutoff events.
2. The phrase must be a SPECIFIC, UNAMBIGUOUS answer (exact date, name, number).
   Bad: "in the 2020s"  Good: "in March 2020"
3. The answer must not be guessable without knowing the specific event.
   Bad: "The 2020 US election was held in → November" (any model knows this)
   Good: "Joe Biden defeated Donald Trump in the US presidential election in → November 2020"
4. Cover diverse topics: politics, science, technology, disasters, sports, culture, economics.
5. Use factually accurate, verifiable events only.
6. Phrase prompts naturally, as if continuing a sentence in an article.

Return a JSON array of objects, each with exactly two keys: "prompt" and "phrase".
The phrase should include a leading space (e.g. " March 2020", not "March 2020").
Return ONLY the JSON array, no explanation."""


def generate_items(year: int, count: int, model: str) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    user_msg = (
        f"Generate {count} benchmark items for cutoff year {year}.\n"
        f"All events must have occurred in {year + 1} or later.\n"
        f"Aim for variety: politics, tech, science, disasters, culture, economics.\n"
        f"Return a JSON array of {{\"prompt\": ..., \"phrase\": ...}} objects."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    # Model may return {"items": [...]} or just [...]
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        # Find the first list value
        items = next(v for v in parsed.values() if isinstance(v, list))
    else:
        raise ValueError(f"Unexpected response format: {type(parsed)}")

    # Validate structure
    validated = []
    for item in items:
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt", "").strip()
        phrase = item.get("phrase", "")
        if not prompt or not phrase:
            continue
        # Ensure phrase has a leading space
        if not phrase.startswith(" "):
            phrase = " " + phrase
        validated.append({"prompt": prompt, "phrase": phrase})

    return validated


def build_benchmark_entry(items: list[dict]) -> dict:
    return {
        "threshold": 0.10,
        "epsilon": -6.0,
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate SN38 leak benchmark items using an LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--year", type=int, choices=ALL_YEARS,
                        help="Generate items for a single cutoff year")
    target.add_argument("--all", action="store_true",
                        help=f"Generate items for all years ({ALL_YEARS[0]}-{ALL_YEARS[-1]})")

    parser.add_argument("--count", type=int, default=10,
                        help="Number of items to generate per year (default: 10)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write full benchmark JSON to this file")
    parser.add_argument("--merge", type=Path, default=None,
                        help="Merge generated items INTO an existing benchmark JSON file")

    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set", file=sys.stderr)
        return 1

    years = ALL_YEARS if args.all else [args.year]
    results: dict[str, dict] = {}

    # Load existing data if merging
    existing: dict = {}
    if args.merge and args.merge.exists():
        with args.merge.open() as f:
            existing = json.load(f)
        print(f"Loaded existing benchmark: {args.merge}")

    for year in years:
        print(f"Generating {args.count} items for year {year}...", end=" ", flush=True)
        try:
            items = generate_items(year, args.count, args.model)
            results[str(year)] = build_benchmark_entry(items)
            print(f"got {len(items)} items")
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue

    if not results:
        print("No items generated.")
        return 1

    # Print to stdout if no file output requested
    if not args.output and not args.merge:
        print()
        print(json.dumps(results, indent=2))
        return 0

    # Write to output file
    if args.output:
        with args.output.open("w") as f:
            note = {"_note": "Generated by generate_benchmarks.py. Only post-cutoff events per year."}
            json.dump({**note, **results}, f, indent=2)
        print(f"\nWritten to {args.output}")

    # Merge into existing file
    if args.merge:
        for year_str, entry in results.items():
            if year_str in existing and isinstance(existing[year_str], dict):
                # Append new items to existing ones, dedup by prompt
                old_prompts = {i["prompt"] for i in existing[year_str].get("items", [])}
                new_items = [i for i in entry["items"] if i["prompt"] not in old_prompts]
                existing[year_str]["items"].extend(new_items)
                print(f"  {year_str}: added {len(new_items)} new items "
                      f"(total {len(existing[year_str]['items'])})")
            else:
                existing[year_str] = entry
                print(f"  {year_str}: created with {len(entry['items'])} items")

        with args.merge.open("w") as f:
            json.dump(existing, f, indent=2)
        print(f"\nMerged into {args.merge}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
