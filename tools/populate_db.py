"""
Utility wrapper around chordcodex.scripts.db_fill_v2 to standardise local runs.

Examples
--------
python -m tools.populate_db --mode quick
python -m tools.populate_db --mode full --batch-size 15000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import config as cfg
from chordcodex.model import QueryExecutor

DEFAULT_MODES = {
    "quick": {
        "mode": "benchmark-insert",
        "limit": 100_000,
        "batch_size": 5_000,
        "description": "Fast sanity run (~100k rows) to validate connectivity.",
    },
    "full": {
        "mode": "full-run",
        "limit": None,
        "batch_size": 10_000,
        "description": "Complete load (~2.6M rows). Expect several minutes.",
    },
}


def build_command(args: argparse.Namespace) -> Sequence[str]:
    base = [sys.executable, "-m", "chordcodex.scripts.db_fill_v2", "--mode", args.mode_name]
    if args.limit is not None:
        base.extend(["--limit", str(args.limit)])
    if args.batch_size is not None:
        base.extend(["--batch-size", str(args.batch_size)])
    if args.extra:
        base.extend(args.extra)
    return base


def resolve_mode(user_mode: str, limit: int | None, batch_size: int | None) -> tuple[str, int | None, int | None]:
    mode_key = user_mode.lower()
    if mode_key not in DEFAULT_MODES:
        raise SystemExit(f"Unknown mode '{user_mode}'. Valid options: {', '.join(DEFAULT_MODES)}.")

    default = DEFAULT_MODES[mode_key]
    mode_name = default["mode"]
    resolved_limit = limit if limit is not None else default["limit"]
    resolved_batch_size = batch_size if batch_size is not None else default["batch_size"]
    return mode_name, resolved_limit, resolved_batch_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate the ChordCodex database with canonical datasets.")
    parser.add_argument(
        "--mode",
        default="quick",
        help="Run preset (quick/full). Defaults to 'quick'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override row limit (only honoured for benchmark modes).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Rows per batch when streaming inserts.",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed verbatim to chordcodex.scripts.db_fill_v2.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing.",
    )
    return parser.parse_args()


def check_rowcount() -> int:
    executor = QueryExecutor(**cfg.config_db)
    result = executor.as_pandas("SELECT COUNT(*) AS total FROM chords;")
    return int(result.iloc[0]["total"])


def main() -> None:
    args = parse_args()
    mode_name, limit, batch_size = resolve_mode(args.mode, args.limit, args.batch_size)
    args.mode_name = mode_name
    args.limit = limit
    args.batch_size = batch_size

    command = build_command(args)
    printable = " ".join(command)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Prepared command:")
    print(f"  {printable}")

    if args.dry_run:
        print("Dry run requested; exiting without executing.")
        return

    subprocess.run(command, check=True, cwd=Path.cwd())

    total = check_rowcount()
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Completed. Current chords count: {total:,}.")


if __name__ == "__main__":
    main()
