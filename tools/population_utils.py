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
    """Elimina duplicados conservando inversiones marcadas."""

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

        if "__inv_flag" in work.columns:
            inv_flag_series = work["__inv_flag"]
            inv_mask = inv_flag_series.apply(lambda v: bool(v) if pd.notnull(v) else False)

            def _inv_suffix(row: pd.Series) -> str:
                parts: list[str] = []
                src = row.get("__inv_source_id")
                if pd.notnull(src):
                    try:
                        parts.append(f"src:{int(src)}")
                    except Exception:  # pragma: no cover - defensive
                        parts.append(f"src:{src}")
                else:
                    base_id = row.get("id")
                    if pd.notnull(base_id):
                        parts.append(f"srcid:{base_id}")
                rot = row.get("__inv_rotation")
                if pd.notnull(rot):
                    try:
                        parts.append(f"rot:{int(rot)}")
                    except Exception:  # pragma: no cover - defensive
                        parts.append(f"rot:{rot}")
                mask_val = row.get("abs_mask_int")
                if pd.notnull(mask_val):
                    try:
                        parts.append(f"mask:{int(mask_val)}")
                    except Exception:  # pragma: no cover - defensive
                        parts.append(f"mask:{mask_val}")
                if not parts:
                    parts.append(f"idx:{row.name}")
                return "inv|" + "|".join(parts)

            if inv_mask.any():
                base_masks = set()
                if (~inv_mask).any():
                    try:
                        base_masks = set(
                            work.loc[~inv_mask, "abs_mask_int"]
                            .dropna()
                            .astype(int)
                            .tolist()
                        )
                    except Exception:  # pragma: no cover - defensive
                        base_masks = set()

                inv_rows = work.loc[inv_mask]

                def _needs_suffix(row: pd.Series) -> bool:
                    mask_val = row.get("abs_mask_int")
                    if mask_val is None or pd.isna(mask_val):
                        return True
                    try:
                        mask_int = int(mask_val)
                    except Exception:
                        return True
                    return mask_int not in base_masks

                suffix_mask = inv_rows.apply(_needs_suffix, axis=1)
                if suffix_mask.any():
                    inv_suffix = inv_rows.loc[suffix_mask].apply(_inv_suffix, axis=1)
                    key_series.loc[inv_suffix.index] = (
                        key_series.loc[inv_suffix.index].astype(str) + "|" + inv_suffix
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
