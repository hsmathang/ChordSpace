"""Utilidades de formateo y agregación usadas por el renderer de reportes."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "format_rate",
    "format_optional",
    "format_rate_with_std",
    "format_value_with_std",
    "format_seed_list",
    "compute_rank",
]


def format_rate(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def format_optional(value: Optional[float]) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value:.4f}"


def format_rate_with_std(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None or np.isnan(mean):
        return "n/a"
    text = f"{100.0 * mean:.1f}%"
    if std is not None and not np.isnan(std) and std > 0:
        text += f" +/- {100.0 * std:.1f}%"
    return text


def format_value_with_std(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None or np.isnan(mean):
        return "n/a"
    text = f"{mean:.4f}"
    if std is not None and not np.isnan(std) and std > 0:
        text += f" +/- {std:.4f}"
    return text


def format_seed_list(seeds: Optional[Sequence[int]]) -> str:
    if not seeds:
        return "-"
    return ", ".join(str(s) for s in seeds)


def compute_rank(df: pd.DataFrame) -> List[int]:
    tmp = df.copy()
    tmp["_stress"] = tmp["stress_mean"].fillna(np.inf)
    tmp["_trust"] = tmp["trustworthiness_mean"].fillna(-np.inf)
    ordered = tmp.sort_values(by=["_stress", "_trust"], ascending=[True, False])
    rank_map = {idx: rank for rank, idx in enumerate(ordered.index, start=1)}
    return [rank_map[idx] for idx in df.index]
