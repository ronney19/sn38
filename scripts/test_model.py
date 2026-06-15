#!/usr/bin/env python3
"""
Local SN38 model tester — validator-identical Stage 1 logic.

Differences from the real validator (unavoidable):
  1. Benchmarks: validator fetches private data from a TEE-authenticated backend.
     This script uses a local JSON file. A local PASS does NOT guarantee an on-chain PASS.
  2. No SQLite result cache (validator skips re-evaluation of already-seen repo+year combos).
  3. No Bittensor chain interaction.

Everything else is identical to validator.py:
  - year_scores initialise to 0.0 (WORST_SCORE) for all 12 years
  - Pre-download file-size check; skip repo if too large
  - eval_start timer starts AFTER download, BEFORE load_model (load time counts against timeout)
  - Param count checked after load; del model + skip all years if over limit
  - Per-repo timeout uses break (remaining years in that repo are abandoned)
  - Repo deduplication (one download for multiple years sharing the same repo)
  - leak_score = sum(all_12_year_scores) / 12
  - Qualification: leak_score < min_leak_score (-20.0)

Examples:
  # All years from models.json  (main use case)
  python scripts/test_model.py --models models.json

  # Single HuggingFace repo for one year
  python scripts/test_model.py --repo your-user/chronogpt-2019 --year 2019

  # Local directory
  python scripts/test_model.py --path ./chronogpt-2019 --year 2019

  # Add Stage-2-style generation preview
  python scripts/test_model.py --models models.json --generate

  # Custom benchmark file
  python scripts/test_model.py --models models.json --benchmark my_bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sn38.template.chronogpt_model import load_model
from sn38.template.constants import ALL_YEARS
from sn38.template.leak import evaluate
from sn38.template.model_store import (
    count_model_params,
    download_model,
    get_device,
    get_repo_file_size,
    parse_repo,
    validate_models_json,
)

SAMPLE_BENCHMARKS_PATH = Path(__file__).with_name("sample_benchmarks.json")
NUM_YEARS = len(ALL_YEARS)   # 12  (2013-2024)
WORST_SCORE = 0.0            # same constant as validator.py


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def fetch_config(url: str | None) -> dict:
    config = {
        "max_parameters":  2_000_000_000,
        "max_model_bytes": 8_000_000_000,
        "max_eval_seconds": 120,
        "leak_epsilon":    -6.0,
        "min_leak_score":  -20.0,
        "top_n_for_quality": 10,
        "leak_weight":     0.7,
        "quality_weight":  0.3,
    }
    if not url:
        return config
    try:
        resp = requests.get(f"{url.rstrip('/')}/config", timeout=10)
        resp.raise_for_status()
        config.update(resp.json())
        print(f"[config] Loaded from {url}")
    except Exception as exc:
        print(f"[config] Could not fetch ({exc}); using defaults")
    return config


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------

def load_benchmarks(path: Path) -> dict[str, dict]:
    """Return {year_str: benchmark_dict}. Empty dict if file missing."""
    if not path.exists():
        print(f"[warn] Benchmark file not found: {path} — leak checks will be skipped")
        return {}
    with path.open() as f:
        data = json.load(f)
    # Strip non-year keys (comments, defaults)
    return {k: v for k, v in data.items() if k.isdigit()}


# ---------------------------------------------------------------------------
# Stage 1: Leak detection — mirrors validator.py exactly
# ---------------------------------------------------------------------------

def run_stage1(models: dict[str, str], config: dict, benchmarks: dict[str, dict]) -> dict:
    """
    Args:
        models:     {year_str: repo_str}  — same format as the on-chain submission.
        config:     validator config dict.
        benchmarks: {year_str: benchmark_dict} from local file.

    Returns dict with keys: year_scores, leak_score, year_details.
    """

    # --- validator.py line 78: all years start at WORST_SCORE ---
    year_scores: dict[int, float] = {year: WORST_SCORE for year in ALL_YEARS}
    year_details: dict[int, dict] = {year: {"status": "missing"} for year in ALL_YEARS}

    # --- validator.py line 77-89: group years by repo (deduplication) ---
    repo_to_years: dict[str, list[int]] = {}
    for year in ALL_YEARS:
        repo_str = models.get(str(year))
        if not repo_str:
            continue
        # (validator also checks cache here; we skip that)
        repo_to_years.setdefault(repo_str, []).append(year)

    # --- validator.py line 91-130: evaluate each unique repo ---
    for repo_str, years in repo_to_years.items():
        repo_id, revision = parse_repo(repo_str)
        print(f"\n[repo] {repo_str}  →  years {years}")

        # --- validator.py line 93-96: pre-download size check ---
        file_size = get_repo_file_size(repo_id, revision)
        if file_size > config["max_model_bytes"]:
            print(f"  SKIP: {file_size:,} bytes > limit {config['max_model_bytes']:,}")
            for year in years:
                year_details[year] = {"status": "skipped", "reason": "too_large"}
            continue

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                print("  Downloading...")
                path = download_model(repo_id, tmpdir, revision=revision)

                # --- validator.py line 102: eval_start is AFTER download ---
                eval_start = time.time()
                device = get_device()
                model = load_model(path, device)
                param_count = count_model_params(model)
                load_s = time.time() - eval_start
                print(f"  Loaded in {load_s:.1f}s  |  {param_count:,} params ({param_count / 1e9:.2f}B)")
                if file_size:
                    print(f"  File size: {file_size:,} bytes")

                # --- validator.py line 107-110: param check after load ---
                if param_count > config["max_parameters"]:
                    print(f"  SKIP: {param_count / 1e9:.2f}B params > limit {config['max_parameters'] / 1e9:.1f}B")
                    for year in years:
                        year_details[year] = {"status": "skipped", "reason": "too_many_params"}
                    del model
                    continue

                # --- validator.py line 112-125: per-year evaluation ---
                for year in years:

                    # validator.py line 113-115: timeout uses break (not continue)
                    if time.time() - eval_start > config["max_eval_seconds"]:
                        print(f"  TIMEOUT  ({config['max_eval_seconds']}s elapsed) — remaining years abandoned")
                        year_details[year] = {"status": "timeout"}
                        # Note: validator breaks here, remaining years in `years` also stay at WORST_SCORE
                        break

                    benchmark = benchmarks.get(str(year))
                    if not benchmark:
                        # validator.py line 118-120: no benchmark → score stays WORST_SCORE
                        print(f"  [year {year}] no benchmark — score stays {WORST_SCORE} (worst)")
                        year_details[year] = {"status": "no_benchmark"}
                        continue

                    failed, score = evaluate(model, device, benchmark)
                    year_scores[year] = score
                    status = "FAIL" if failed else "PASS"
                    print(f"  [year {year}] {status}  score={score:.4f}  passed={not failed}")
                    year_details[year] = {"status": status, "score": score, "failed": failed}

                del model  # validator.py line 127

        except Exception as exc:
            # validator.py line 129-130
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            for year in years:
                if year_details[year].get("status") in ("missing", None):
                    year_details[year] = {"status": "error", "error": str(exc)}

    # --- validator.py line 132: always divide by NUM_YEARS (12), not len(tested) ---
    leak_score = sum(year_scores.values()) / NUM_YEARS

    return {
        "year_scores": year_scores,
        "leak_score": leak_score,
        "year_details": year_details,
    }


# ---------------------------------------------------------------------------
# Stage 2 preview: generation (identical to quality.py generate_answer)
# ---------------------------------------------------------------------------

SAMPLE_QUESTIONS = [
    "What were the main causes of the 2008 financial crisis?",
    "How did the Fukushima disaster impact global energy policy?",
    "What were the economic consequences of Brexit on UK finance?",
]


def run_generation_preview(repo_str: str, config: dict):
    """
    Downloads the repo and generates sample answers using the exact same sampling
    logic as quality.py's generate_answer(). Runs on the first qualifying repo only.
    """
    import tiktoken
    import torch

    tokenizer = tiktoken.get_encoding("gpt2")

    repo_id, revision = parse_repo(repo_str)
    print(f"\n[generation preview] {repo_str}")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = download_model(repo_id, tmpdir, revision=revision)
        device = get_device()
        model = load_model(path, device)

        for q in SAMPLE_QUESTIONS:
            tokens = torch.tensor(
                tokenizer.encode(q), dtype=torch.long
            ).unsqueeze(0).to(device)
            xgen = tokens.clone()
            with torch.no_grad():
                for _ in range(128):
                    logits = model(xgen)[:, -1, :]
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    next_token = torch.gather(
                        topk_indices, -1, torch.multinomial(topk_probs, 1)
                    )
                    xgen = torch.cat([xgen, next_token], dim=1)
            answer = tokenizer.decode(xgen[0][tokens.shape[1]:].tolist())
            preview = answer.replace("\n", " ")[:240]
            print(f"  Q: {q}")
            print(f"  A: {preview}{'...' if len(answer) > 240 else ''}")

        del model


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(result: dict, config: dict):
    year_scores = result["year_scores"]
    year_details = result["year_details"]

    print()
    print("=" * 60)
    print("STAGE 1 REPORT")
    print("=" * 60)
    for year in ALL_YEARS:
        score = year_scores[year]
        status = year_details.get(year, {}).get("status", "missing")
        print(f"  {year}:  score={score:9.4f}   [{status}]")

    leak_score = result["leak_score"]
    min_ls = config["min_leak_score"]
    qualified = leak_score < min_ls

    print()
    print(f"leak_score = sum({NUM_YEARS} year scores) / {NUM_YEARS} = {leak_score:.4f}")
    print(f"threshold  = {min_ls}  (must be LESS than this to qualify for Stage 2)")
    print()
    if qualified:
        print(f"RESULT: PASS — would qualify for Stage 2  (leak_score {leak_score:.4f} < {min_ls})")
    else:
        print(f"RESULT: FAIL — would NOT qualify  (leak_score {leak_score:.4f} >= {min_ls})")
    print()
    print("NOTE: real validator benchmarks are private (TEE-only).")
    print("Local scores are illustrative — local PASS does not guarantee on-chain PASS.")


# ---------------------------------------------------------------------------
# Local-path shim: lets --path bypass HuggingFace entirely
# ---------------------------------------------------------------------------

_LOCAL_PREFIX = "__local__:"


def _install_local_path_shim():
    """Patch model_store so local paths work without HuggingFace."""
    from sn38.template import model_store

    def _local_download(repo_id, local_dir, revision=None):
        return repo_id.removeprefix(_LOCAL_PREFIX)

    def _local_size(repo_id, revision=None):
        p = Path(repo_id.removeprefix(_LOCAL_PREFIX))
        total = 0
        for name in ("model.safetensors", "pytorch_model.bin"):
            f = p / name
            if f.exists():
                total += f.stat().st_size
        return total

    model_store.download_model = _local_download
    model_store.get_repo_file_size = _local_size


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local SN38 tester — validator-identical Stage 1 logic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--models", type=Path,
                     help="models.json mapping year → HF repo (tests all years)")
    src.add_argument("--repo",   type=str,
                     help="Single HF repo: owner/repo or owner/repo@revision")
    src.add_argument("--path",   type=str,
                     help="Local model directory (config.json + model.safetensors)")

    parser.add_argument("--year", type=int,
                        help="Cutoff year — required with --repo or --path")
    parser.add_argument("--benchmark", type=Path, default=None,
                        help="Benchmark JSON file (default: sample_benchmarks.json)")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip leak evaluation entirely")
    parser.add_argument("--generate", action="store_true",
                        help="Run Stage-2-style generation preview on the first repo")
    parser.add_argument("--config-url", type=str, default="https://api.chronollm.com",
                        help="Backend URL for live config limits")
    args = parser.parse_args()

    if (args.repo or args.path) and args.year is None:
        parser.error("--year is required with --repo or --path")

    # Build models dict identical to what the validator receives on-chain
    if args.models:
        with args.models.open() as f:
            models = json.load(f)
        missing = validate_models_json(models)
        if missing:
            print(f"[warn] models.json missing years {missing} — score stays {WORST_SCORE} for those years")
    elif args.repo:
        models = {str(args.year): args.repo}
    else:
        _install_local_path_shim()
        models = {str(args.year): f"{_LOCAL_PREFIX}{args.path}"}

    config = fetch_config(args.config_url)

    benchmarks: dict = {}
    if not args.no_benchmark:
        bench_path = args.benchmark or SAMPLE_BENCHMARKS_PATH
        benchmarks = load_benchmarks(bench_path)

    result = run_stage1(models, config, benchmarks)
    print_report(result, config)

    if args.generate:
        # Use first listed repo for generation (matches validator's run_quality_duels which
        # takes list(submissions[uid].values())[0])
        first_repo = next(iter(models.values()))
        if first_repo.startswith(_LOCAL_PREFIX):
            print("\n[generate] --generate not supported with --path (local model skips HF download)")
        else:
            run_generation_preview(first_repo, config)

    return 0 if result["leak_score"] < config["min_leak_score"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
