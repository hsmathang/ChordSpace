"""
Debug tool: reproduce GUI population mixing (base + pops A/B/C),
report counts per source, and show dedupe impact.

Usage examples
  # Base + A + B
  python -m tools.debug_population_mixer --base QUERY_TRIADS_CORE \
      --pop A:QUERY_TRIADS_CORE --pop B:QUERY_TRIADS_CORE \
      --out outputs/debug_mix

  # Only pops (no base)
  python -m tools.debug_population_mixer --pop A:QUERY_TRIADS_CORE --pop B:QUERY_TRIADS_CORE

What it does
  - Loads each source using the same helpers the GUI uses.
  - Tags rows with __source__ = BASE:<Q> or <TYPE>:<Q>.
  - Prints counts per source before and after dedupe.
  - Dedupe policy: prefer 'abs_mask_int' if present; fallback to (code, interval) normalized.
  - Optionally dumps CSVs to inspect.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from tools.experiment_inversions import (
    _parse_pop_spec,
    _build_population,
)
from tools.query_registry import resolve_query_sql
from config import config_db
from tools.population_utils import dedupe_population

try:  # prefer packaged executor
    from chordcodex.model import QueryExecutor  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from synth_tools import QueryExecutor  # type: ignore


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Debug population mixer (base + pops A/B/C)")
    ap.add_argument("--base", default=None, help="Optional base QUERY_* constant")
    ap.add_argument("--pop", action="append", default=None, help="Population spec like A:QUERY_* (repeatable)")
    ap.add_argument("--out", type=Path, default=None, help="Optional output folder for CSV dumps")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    frames: List[pd.DataFrame] = []
    qe = QueryExecutor(**config_db)

    if args.base:
        sql = resolve_query_sql(args.base)
        df = qe.as_pandas(sql)
        df = df.copy()
        df["__source__"] = f"BASE:{args.base}"
        frames.append(df)

    pops = args.pop or []
    for spec in pops:
        ptype, qname = _parse_pop_spec(spec)
        df = _build_population(ptype, qname)
        df = df.copy()
        df["__source__"] = f"{ptype}:{qname}"
        frames.append(df)

    if not frames:
        raise SystemExit("No inputs. Use --base or --pop.")

    mix = pd.concat(frames, ignore_index=True)
    print("=== RAW COUNTS BY SOURCE ===")
    print(mix["__source__"].value_counts())

    dedup, key = dedupe_population(mix)
    print(f"\n=== DEDUP KEY: {key} ===")
    print("=== COUNTS BY SOURCE AFTER DEDUPE ===")
    print(dedup["__source__"].value_counts())
    print(f"\nTOTAL RAW: {len(mix)} | TOTAL DEDUP: {len(dedup)} | REMOVED: {len(mix) - len(dedup)}")

    if args.out:
        outdir = args.out.resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        mix.to_csv(outdir / "mix_raw.csv", index=False)
        dedup.to_csv(outdir / "mix_dedup.csv", index=False)
        print(f"CSV dumps written to: {outdir}")


if __name__ == "__main__":
    main()
