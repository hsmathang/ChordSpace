"""Utilities to prepare serialisable payloads for visualisations."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .data import ChordEntry

COLOR_PER_PAIR_SUBTRACT: float = 0.0
COLOR_PER_NOTE_SUBTRACT: float = 0.0
COLOR_PER_EXISTING_SUBTRACT: float = 0.0
COLOR_EXISTING_THRESHOLD: float = 1e-6
COLOR_DEN_EXPONENT: float = 1.0
COLOR_OUTPUT_EXPONENT: float = 1.0
COLOR_EXPONENTS: Tuple[float, ...] = tuple(i / 20.0 for i in range(0, 21))


def _safe_denominator(raw: np.ndarray, subtract: float = 0.0) -> np.ndarray:
    den = np.asarray(raw, dtype=float) - float(subtract)
    den[den < 1.0] = 1.0
    return den


def apply_color_mode(
    mode: str,
    exponent: Optional[float],
    totals_raw: np.ndarray,
    totals_adjusted: np.ndarray,
    pairs_arr: np.ndarray,
    types_arr: np.ndarray,
) -> Tuple[np.ndarray, str]:
    mode_lower = mode.lower()

    if mode_lower == "pair_exp":
        exp = exponent if exponent is not None else 1.0
        denom = _safe_denominator(pairs_arr, subtract=COLOR_PER_PAIR_SUBTRACT)
        denom = np.power(denom, exp)
        if not np.isclose(COLOR_DEN_EXPONENT, 1.0):
            denom = np.power(denom, COLOR_DEN_EXPONENT)
        vals = totals_raw / denom
        title = f"Total/Pares^{_format_exp(exp)}"
    elif mode_lower == "types_exp":
        exp = exponent if exponent is not None else 1.0
        denom = _safe_denominator(types_arr, subtract=COLOR_PER_EXISTING_SUBTRACT)
        denom = np.power(denom, exp)
        if not np.isclose(COLOR_DEN_EXPONENT, 1.0):
            denom = np.power(denom, COLOR_DEN_EXPONENT)
        vals = totals_adjusted / denom
        title = f"Total ajustado/Tipos^{_format_exp(exp)}"
    elif mode_lower == "raw_total":
        vals = totals_raw.copy()
        title = "Total bruto"
    else:
        raise ValueError(f"Modo de color no soportado: {mode}")

    if not np.isclose(COLOR_OUTPUT_EXPONENT, 1.0):
        vals = np.power(np.clip(vals, 0.0, None), COLOR_OUTPUT_EXPONENT)
    return vals, title


def _format_exp(val: float) -> str:
    return f"{val:.2f}".rstrip("0").rstrip(".")


def serialise_entries(entries: Sequence[ChordEntry]) -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    for entry in entries:
        data = asdict(entry)
        data["hist"] = entry.hist.tolist()
        data["counts"] = entry.counts.tolist()
        payload.append(data)
    return payload


def build_visualisation_payload(
    embedding: np.ndarray,
    entries: Sequence[ChordEntry],
    color_values: np.ndarray,
    color_title: str,
    *,
    extras: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    base: Dict[str, object] = {
        "embedding": np.asarray(embedding, dtype=float).tolist(),
        "entries": serialise_entries(entries),
        "color": {
            "values": np.asarray(color_values, dtype=float).tolist(),
            "min": float(np.min(color_values)) if len(color_values) else 0.0,
            "max": float(np.max(color_values)) if len(color_values) else 0.0,
            "title": color_title,
        },
    }
    if extras:
        base.update(dict(extras))
    return base


__all__ = [
    "COLOR_EXPONENTS",
    "COLOR_EXISTING_THRESHOLD",
    "apply_color_mode",
    "build_visualisation_payload",
    "serialise_entries",
]
