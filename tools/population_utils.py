"""Utilities for building/deduplicating chord populations."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def normalize_interval_value(value) -> str:
    """Return a canonical string representation for an interval sequence."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return ",".join(str(int(x)) for x in value)
    if value is None:
        return ""
    s = str(value)
    s = s.strip("{}[]() ")
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return ",".join(parts)


def _fallback_key(df: pd.DataFrame) -> pd.Series:
    pieces = []
    for _, row in df.iterrows():
        code_val = ""
        if "code" in row and pd.notnull(row["code"]):
            code_val = str(row["code"])
        elif "id" in row and pd.notnull(row["id"]):
            code_val = f"id:{row['id']}"

        interval_val = ""
        if "interval" in row:
            interval_val = normalize_interval_value(row["interval"])

        pieces.append(f"{code_val}|{interval_val}")
    return pd.Series(pieces, index=df.index)


def dedupe_population(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Drop duplicates while keeping rows with missing abs_mask_int distinct."""

    def _priority(src: str) -> int:
        if isinstance(src, str):
            if src.startswith("BASE:"):
                return 0
            if src.startswith("A:"):
                return 1
            if src.startswith("B:"):
                return 2
            if src.startswith("C:"):
                return 3
        return 4

    ordered = df.copy()
    if "__source__" in ordered.columns:
        ordered = ordered.sort_values(
            by="__source__",
            key=lambda s: s.map(_priority),
            kind="stable",
        )

    if "abs_mask_int" in ordered.columns:
        work = ordered.copy()
        key_series = work["abs_mask_int"].apply(
            lambda v: f"mask:{int(v)}" if pd.notnull(v) else None
        )
        missing = key_series.isna()
        if missing.any():
            key_series[missing] = _fallback_key(work.loc[missing]).apply(
                lambda s: f"fallback:{s}"
            )
        work["__dedupe_key__"] = key_series
        dedup = work.drop_duplicates(subset="__dedupe_key__", keep="first").copy()
        dedup.drop(columns=["__dedupe_key__"], inplace=True)
        return dedup, "abs_mask_int+fallback"

    work = ordered.copy()
    if "interval" in work.columns or "code" in work.columns or "id" in work.columns:
        work["__dedupe_key__"] = _fallback_key(work)
        dedup = work.drop_duplicates(subset="__dedupe_key__", keep="first").copy()
        dedup.drop(columns=["__dedupe_key__"], inplace=True)
        return dedup, "code+interval"

    return ordered.copy(), "none"
