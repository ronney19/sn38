"""
SN38 ChronoGPT — Dataset Generation Script

Builds year-filtered training datasets for ChronoGPT models (years 2013-2024).
Each model must only see data available up to its cutoff year.

Sources used:
  1. FineWeb       — HF-native web crawl, date-filtered (replaces C4/CC-News)
  2. Wikipedia     — wikimedia/wikipedia, title-filtered by cutoff year
  3. ChronoInstruct-SFT-v1 — timeless SFT data (pre-2000 safe, all years)

Why FineWeb instead of C4 / CC-News:
  allenai/c4 and stanford-oval/ccnews store their parquet files on external
  CDNs (GCS/AWS) which may not be reachable from all servers. FineWeb
  (HuggingFaceFW/fineweb) is stored directly on HuggingFace xet storage
  and streams reliably. It has a 'date' field for year filtering and covers
  2013-2024.

Output per year:
  data/{year}/train.bin   — tokenized uint16 array (GPT-2 tiktoken, vocab 50304)
  data/{year}/meta.json   — token count, source breakdown

Tokenizer: GPT-2 tiktoken (same as validator's leak.py and quality.py)

Usage:
    # Single year
    python build_dataset.py --year 2018

    # All years
    python build_dataset.py --all

    # Custom token target
    python build_dataset.py --year 2020 --target-tokens 500_000_000

    # Only SFT data (fast, no streaming)
    python build_dataset.py --year 2018 --sft-only

    # Skip SFT, only temporal sources
    python build_dataset.py --year 2018 --no-sft

    # Validate output after building
    python build_dataset.py --year 2018 --validate

Requirements:
    pip install datasets tiktoken numpy tqdm
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import tiktoken
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit("Run: pip install datasets tiktoken numpy tqdm")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

ALL_YEARS = list(range(2013, 2025))

# GPT-2 tokenizer — must match the validator's leak.py and quality.py
TOKENIZER = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = 50304  # ChronoGPT config

EOT_TOKEN = TOKENIZER.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

# Default token targets per year (adjust to your storage / compute budget)
# 1B tokens ~ 4GB on disk as uint16
DEFAULT_TARGET_TOKENS = 200_000_000   # 200M tokens per year

# SFT dataset from manelalab (pre-2000 safe, all years)
SFT_DATASET = "manelalab/ChronoInstruct-SFT-v1"
SFT_TARGET_TOKENS = 20_000_000

OUTPUT_DIR = Path("data")


# ─────────────────────────────────────────────
# Tokenization helpers
# ─────────────────────────────────────────────

def tokenize(text: str) -> list[int]:
    """Tokenize text with EOT separator. Returns list of token ids."""
    return TOKENIZER.encode(text, allowed_special={"<|endoftext|>"}) + [EOT_TOKEN]


def tokens_to_bin(tokens: list[int], path: Path):
    """Append tokens to a binary file as uint16."""
    arr = np.array(tokens, dtype=np.uint16)
    with open(path, "ab") as f:
        f.write(arr.tobytes())


# ─────────────────────────────────────────────
# Date parsing utilities
# ─────────────────────────────────────────────

def parse_year_from_date(date_str: str) -> int | None:
    """Extract year from various date string formats."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str[:len(fmt)], fmt).year
        except ValueError:
            pass
    m = re.search(r"\b(20\d{2}|19\d{2})\b", date_str)
    if m:
        return int(m.group(1))
    return None


def text_is_clean(text: str) -> bool:
    """Basic quality filter — reject very short or mostly non-ascii text."""
    if not text or len(text) < 200:
        return False
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
    return ascii_ratio > 0.85


# ─────────────────────────────────────────────
# Source 1: FineWeb (replaces C4 + CC-News)
# ─────────────────────────────────────────────
#
# HuggingFaceFW/fineweb is stored on HF's own xet storage — streams reliably
# without depending on external CDNs.
#
# Config options:
#   "sample-10BT"  — 10B tokens, all years mixed, fast to stream
#   "sample-100BT" — 100B tokens, more coverage
#   "sample-350BT" — 350B tokens, maximum coverage
#
# Fields: text, id, dump, url, date, file_path, language, token_count
# The 'date' field format: "2013-05-09T12:37:14Z"
# The 'dump' field encodes crawl period: "CC-MAIN-2013-20"
#
# For a 200M token budget: sample-10BT is sufficient.
# For 1B+ tokens per year: use sample-100BT or sample-350BT.

FINEWEB_CONFIG = "sample-10BT"   # change to sample-100BT for larger budgets


def yield_fineweb(cutoff_year: int, config: str = FINEWEB_CONFIG):
    """
    Stream FineWeb filtered to cutoff_year.
    Filters on the 'date' field (ISO format) — keeps only docs with date <= Dec 31 cutoff_year.
    """
    print(f"  [FineWeb] Streaming HuggingFaceFW/fineweb ({config}) cutoff <= {cutoff_year}...")
    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb",
            name=config,
            split="train",
            streaming=True,
        )
        for row in ds:
            date_str = row.get("date", "")
            year = parse_year_from_date(date_str)
            if year is None or year > cutoff_year:
                continue
            text = row.get("text", "")
            if text_is_clean(text):
                yield text, year
    except Exception as e:
        print(f"  [FineWeb] FAILED: {type(e).__name__}: {e}")
        raise   # re-raise so build_year can log and skip this source


# ─────────────────────────────────────────────
# Source 2: Wikipedia
# ─────────────────────────────────────────────
#
# wikimedia/wikipedia "20231101.en" — stored on HF xet storage.
# Articles themselves are timeless knowledge; we title-filter to exclude
# articles whose title contains a post-cutoff year (e.g. "2024 US Election").
#
# Fields: id, url, title, text

def yield_wikipedia(cutoff_year: int):
    """
    Stream English Wikipedia, skipping articles whose title contains a year
    after cutoff_year.
    """
    print(f"  [Wikipedia] Streaming wikimedia/wikipedia (cutoff title filter <= {cutoff_year})...")
    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
        )
        post_year_strs = [str(y) for y in range(cutoff_year + 1, 2026)]
        for row in ds:
            title = row.get("title", "")
            if any(py in title for py in post_year_strs):
                continue
            text = row.get("text", "")
            if text_is_clean(text):
                yield text, cutoff_year
    except Exception as e:
        print(f"  [Wikipedia] FAILED: {type(e).__name__}: {e}")
        raise


# ─────────────────────────────────────────────
# Source 3: ChronoInstruct-SFT-v1 (SFT layer)
# ─────────────────────────────────────────────
#
# All 648K examples verified pre-2000 by GPT-4.1 — safe for every year.
# Trains instruction-following behavior, directly helps Stage 2 quality score.
#
# Fields: conversation (dict with instruction/input/output), label, source

SFT_TEMPLATE = """\
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

SFT_TEMPLATE_NO_INPUT = """\
### Instruction:
{instruction}

### Response:
{output}"""


def format_sft_example(row: dict) -> str:
    """Convert a ChronoInstruct row to plain text for causal LM training."""
    conv = row.get("conversation", {})
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except Exception:
            return conv

    instruction = conv.get("instruction", "").strip()
    inp = conv.get("input", "").strip()
    output = conv.get("output", "").strip()

    if not instruction or not output:
        return ""
    if inp:
        return SFT_TEMPLATE.format(instruction=instruction, input=inp, output=output)
    return SFT_TEMPLATE_NO_INPUT.format(instruction=instruction, output=output)


def yield_sft(target_tokens: int):
    """
    Stream ChronoInstruct-SFT-v1. Pre-2000 verified — safe for all years.
    Only keeps rows with label=0, confidence=10 (maximum filter strictness).
    """
    print(f"  [SFT] Streaming {SFT_DATASET} (target {target_tokens:,} tokens)...")
    collected = 0
    try:
        ds = load_dataset(
            SFT_DATASET,
            split="train",
            streaming=True,
        )
        for row in ds:
            label_info = row.get("label", {})
            if isinstance(label_info, str):
                try:
                    label_info = json.loads(label_info)
                except Exception:
                    label_info = {}
            if label_info.get("label", 1) != 0:
                continue
            if label_info.get("confidence", 0) < 10:
                continue

            text = format_sft_example(row)
            if not text:
                continue

            tokens = tokenize(text)
            collected += len(tokens)
            yield text, "sft"

            if collected >= target_tokens:
                break
    except Exception as e:
        print(f"  [SFT] FAILED: {type(e).__name__}: {e}")
        raise


# ─────────────────────────────────────────────
# Main build function
# ─────────────────────────────────────────────

def build_year(
    year: int,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    include_sft: bool = True,
    sft_only: bool = False,
    fineweb_config: str = FINEWEB_CONFIG,
    output_dir: Path = OUTPUT_DIR,
):
    year_dir = output_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    bin_path = year_dir / "train.bin"
    meta_path = year_dir / "meta.json"

    if bin_path.exists():
        bin_path.unlink()

    total_tokens = 0
    source_counts: dict[str, int] = {}

    def write_from_source(source_name: str, generator, token_budget: int):
        nonlocal total_tokens
        written = 0
        bar = tqdm(
            desc=f"  [{source_name}]",
            unit=" tok",
            unit_scale=True,
            total=token_budget,
        )
        try:
            for text, _ in generator:
                tokens = tokenize(text)
                if written + len(tokens) > token_budget:
                    tokens = tokens[: token_budget - written]
                tokens_to_bin(tokens, bin_path)
                written += len(tokens)
                total_tokens += len(tokens)
                bar.update(len(tokens))
                if written >= token_budget:
                    break
        except Exception as e:
            print(f"\n  [{source_name}] Source failed mid-stream: {type(e).__name__}: {e}")
            print(f"  [{source_name}] Collected {written:,} tokens before failure")
        finally:
            bar.close()

        source_counts[source_name] = written
        print(f"  [{source_name}] wrote {written:,} / {token_budget:,} tokens")

    print(f"\n{'='*60}")
    print(f"Building dataset for year {year}")
    print(f"  Target tokens  : {target_tokens:,}")
    print(f"  FineWeb config : {fineweb_config}")
    print(f"  Output         : {bin_path}")
    print(f"{'='*60}")

    if sft_only:
        write_from_source("SFT", yield_sft(target_tokens), target_tokens)
    else:
        temporal_budget = target_tokens - (SFT_TARGET_TOKENS if include_sft else 0)
        temporal_budget = max(0, temporal_budget)

        # FineWeb gets 80%, Wikipedia 20%
        source_budget = {
            "FineWeb":   int(temporal_budget * 0.80),
            "Wikipedia": int(temporal_budget * 0.20),
        }

        write_from_source("FineWeb",   yield_fineweb(year, fineweb_config), source_budget["FineWeb"])
        write_from_source("Wikipedia", yield_wikipedia(year),               source_budget["Wikipedia"])

        if include_sft:
            write_from_source("SFT", yield_sft(SFT_TARGET_TOKENS), SFT_TARGET_TOKENS)

    meta = {
        "year": year,
        "total_tokens": total_tokens,
        "vocab_size": VOCAB_SIZE,
        "tokenizer": "gpt2-tiktoken",
        "dtype": "uint16",
        "eot_token": EOT_TOKEN,
        "fineweb_config": fineweb_config,
        "sources": source_counts,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nDone: {total_tokens:,} tokens → {bin_path}")
    print(f"Metadata: {meta_path}")
    return total_tokens


# ─────────────────────────────────────────────
# Validation helper
# ─────────────────────────────────────────────

def validate_bin(bin_path: Path, n_samples: int = 5):
    """Read a few decoded samples from a .bin file to verify correctness."""
    print(f"\nValidating {bin_path}...")
    data = np.fromfile(bin_path, dtype=np.uint16)
    print(f"  Total tokens : {len(data):,}")
    print(f"  File size    : {bin_path.stat().st_size / 1e6:.1f} MB")

    eot_positions = np.where(data == EOT_TOKEN)[0]
    print(f"  Documents    : {len(eot_positions):,}")

    prev = 0
    for i, pos in enumerate(eot_positions[:n_samples]):
        chunk = data[prev:pos].tolist()
        text = TOKENIZER.decode(chunk[:200])
        print(f"\n  --- Sample {i+1} (tokens {prev}..{pos}) ---")
        print(f"  {repr(text[:300])}")
        prev = pos + 1


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build year-filtered ChronoGPT training datasets for SN38"
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--year", type=int, choices=ALL_YEARS,
                        help="Build dataset for a single year (2013-2024)")
    target.add_argument("--all", action="store_true",
                        help="Build datasets for all years (2013-2024)")

    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS,
                        help=f"Token target per year (default: {DEFAULT_TARGET_TOKENS:,})")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory (default: data/)")
    parser.add_argument("--sft-only", action="store_true",
                        help="Only ChronoInstruct-SFT-v1, skip temporal sources")
    parser.add_argument("--no-sft", action="store_true",
                        help="Skip SFT, only use FineWeb + Wikipedia")
    parser.add_argument("--fineweb-config", type=str, default=FINEWEB_CONFIG,
                        choices=["sample-10BT", "sample-100BT", "sample-350BT"],
                        help=f"FineWeb config size (default: {FINEWEB_CONFIG})")
    parser.add_argument("--validate", action="store_true",
                        help="After building, validate the .bin file by sampling docs")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    years = ALL_YEARS if args.all else [args.year]

    for year in years:
        build_year(
            year=year,
            target_tokens=args.target_tokens,
            include_sft=not args.no_sft,
            sft_only=args.sft_only,
            fineweb_config=args.fineweb_config,
            output_dir=output_dir,
        )
        if args.validate:
            validate_bin(output_dir / str(year) / "train.bin")

    print("\n\nAll done.")
    if len(years) > 1:
        print("Remember: each year's model must be trained ONLY on its own data/{year}/train.bin")
        print("Do NOT mix datasets across years.")


if __name__ == "__main__":
    main()
