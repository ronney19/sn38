#!/usr/bin/env python3
"""
Automate the full SN38 submission pipeline:

1. Transfer source HF model repos to your target repos (one per year)
2. Write models.json from the transfer results
3. Optionally upload models.json and commit on-chain via sn38.neurons.miner

Example (matches the manual workflow):
  python automate_submission.py \\
      --hf-token hf_xxx \\
      --wallet.name pkp --wallet.hotkey hotkey1 \\
      --submit

  # Transfer only (no on-chain submit):
  python automate_submission.py --hf-token hf_xxx --transfer-only

  # Submit an existing models.json (skip transfers):
  python automate_submission.py --hf-token hf_xxx \\
      --wallet.name pkp --wallet.hotkey hotkey1 \\
      --submit-only --models models.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

from hf_repo_transfer import transfer_repo

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_FILE = SCRIPT_DIR / "transfer_state.json"


def parse_year_range(value: str) -> list[int]:
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(value)]


def year_short(year: int) -> int:
    return year - 2000


def source_repo(source_user: str, source_prefix: str, year: int) -> str:
    return f"{source_user}/{source_prefix}-{year}"


def target_repo(target_user: str, target_prefix: str, target_suffix: str, year: int) -> str:
    return f"{target_user}/{target_prefix}-{year_short(year)}-{target_suffix}"


def load_state(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"completed": {}}


def save_state(path: Path, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.write("\n")


def target_repo_exists(api: HfApi, repo_id: str) -> bool:
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        return bool(info.siblings)
    except RepositoryNotFoundError:
        return False
    except Exception:
        return False


async def run_transfers(
    years: list[int],
    hf_token: str,
    source_user: str,
    source_prefix: str,
    target_user: str,
    target_prefix: str,
    target_suffix: str,
    modify_hashes: bool,
    max_workers: int,
    skip_existing: bool,
    state_file: Path,
) -> dict[str, str]:
    api = HfApi(token=hf_token)
    state = load_state(state_file)
    models: dict[str, str] = {}

    print("=" * 60)
    print(f"Phase 1: Transfer {len(years)} model repos")
    print("=" * 60)

    for i, year in enumerate(years, start=1):
        src = source_repo(source_user, source_prefix, year)
        dst = target_repo(target_user, target_prefix, target_suffix, year)
        year_key = str(year)
        models[year_key] = dst

        print(f"\n[{i}/{len(years)}] Year {year}")
        print(f"  Source: {src}")
        print(f"  Target: {dst}")

        if year_key in state["completed"]:
            print(f"  ⏭️  Already completed (revision {state['completed'][year_key]['revision']})")
            continue

        if skip_existing and target_repo_exists(api, dst):
            print("  ⏭️  Target repo already exists, skipping transfer")
            state["completed"][year_key] = {"source": src, "target": dst, "revision": "skipped"}
            save_state(state_file, state)
            continue

        revision = await transfer_repo(
            source_repo_id=src,
            target_repo_id=dst,
            token=hf_token,
            repo_type="model",
            modify_hashes=modify_hashes,
            max_workers=max_workers,
        )

        state["completed"][year_key] = {"source": src, "target": dst, "revision": revision}
        save_state(state_file, state)

    return models


def write_models_json(models: dict[str, str], output_path: Path) -> None:
    ordered = {str(year): models[str(year)] for year in sorted(int(y) for y in models)}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2)
        f.write("\n")
    print(f"\n✅ Wrote {output_path} ({len(ordered)} years)")


def submit_on_chain(
    models_path: Path,
    hf_token: str,
    wallet_name: str,
    wallet_hotkey: str,
    network: str,
) -> None:
    print("\n" + "=" * 60)
    print("Phase 3: Upload models.json and commit on-chain")
    print("=" * 60)

    cmd = [
        sys.executable,
        "-m",
        "sn38.neurons.miner",
        "--wallet.name",
        wallet_name,
        "--wallet.hotkey",
        wallet_hotkey,
        "--subtensor.network",
        network,
        "--models",
        str(models_path),
        "--hf-token",
        hf_token,
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automate SN38 model transfers, models.json creation, and on-chain submission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--years",
        type=parse_year_range,
        default=parse_year_range("2013-2024"),
        help="Years to transfer, e.g. 2013-2024 (default: 2013-2024)",
    )
    parser.add_argument("--hf-token", required=True, help="Hugging Face write token")
    parser.add_argument("--source-user", default="nanonoa002", help="Source HF username")
    parser.add_argument("--source-prefix", default="chronogpt", help="Source repo name prefix")
    parser.add_argument("--target-user", default="Coffiee-new", help="Target HF username")
    parser.add_argument("--target-prefix", default="model", help="Target repo name prefix")
    parser.add_argument("--target-suffix", default="v1", help="Target repo suffix (default: v1)")
    parser.add_argument("--models", default="models.json", help="Output path for models.json")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE), help="Transfer progress file")
    parser.add_argument("--max-workers", type=int, default=32, help="Parallel HF workers")
    parser.add_argument("--no-modify-hashes", dest="modify_hashes", action="store_false", help="Skip safetensor hash tweak")
    parser.add_argument("--skip-existing", action="store_true", help="Skip transfer if target repo already exists")
    parser.add_argument("--transfer-only", action="store_true", help="Only run transfers and write models.json")
    parser.add_argument("--submit-only", action="store_true", help="Skip transfers; only submit models.json on-chain")
    parser.add_argument("--submit", action="store_true", help="Upload models.json and commit on-chain after transfers")
    parser.add_argument("--wallet.name", dest="wallet_name", default=None, help="Bittensor wallet name (required with --submit)")
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", default=None, help="Bittensor wallet hotkey (required with --submit)")
    parser.add_argument("--subtensor.network", dest="network", default="finney", help="Bittensor network (default: finney)")

    args = parser.parse_args()
    models_path = Path(args.models)
    state_file = Path(args.state_file)

    if args.submit_only:
        if not models_path.exists():
            parser.error(f"--submit-only requires existing file: {models_path}")
        if not args.wallet_name or not args.wallet_hotkey:
            parser.error("--wallet.name and --wallet.hotkey are required with --submit-only")
        submit_on_chain(models_path, args.hf_token, args.wallet_name, args.wallet_hotkey, args.network)
        return

    models = await run_transfers(
        years=args.years,
        hf_token=args.hf_token,
        source_user=args.source_user,
        source_prefix=args.source_prefix,
        target_user=args.target_user,
        target_prefix=args.target_prefix,
        target_suffix=args.target_suffix,
        modify_hashes=args.modify_hashes,
        max_workers=args.max_workers,
        skip_existing=args.skip_existing,
        state_file=state_file,
    )

    print("\n" + "=" * 60)
    print("Phase 2: Write models.json")
    print("=" * 60)
    write_models_json(models, models_path)

    if args.transfer_only:
        print("\nDone (transfer-only). Run again with --submit to commit on-chain.")
        return

    if args.submit:
        if not args.wallet_name or not args.wallet_hotkey:
            parser.error("--wallet.name and --wallet.hotkey are required with --submit")
        submit_on_chain(models_path, args.hf_token, args.wallet_name, args.wallet_hotkey, args.network)
        return

    print(
        "\nTransfers and models.json are ready. Submit on-chain with:\n"
        f"  python automate_submission.py --hf-token <token> "
        f"--wallet.name <name> --wallet.hotkey <hotkey> --submit-only --models {models_path}"
    )


if __name__ == "__main__":
    asyncio.run(main())
