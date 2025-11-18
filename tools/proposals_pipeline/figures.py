"""Generación de figuras Plotly para el reporte."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

from tools.proposals_pipeline.metrics import BASE_VECTOR_METRICS
from tools.proposals_pipeline.population import ChordEntry
from visualisations.proposals import build_scatter_payload


@dataclass(frozen=True)
class ColorSettings:
    per_pair_subtract: float
    per_note_subtract: float
    per_existing_subtract: float
    existing_threshold: float
    denominator_exponent: float
    output_exponent: float
    exponents: Sequence[float]


@dataclass(frozen=True)
class HighlightSettings:
    threshold: int
    size_scale: float
    size_delta: float
    selected_opacity: float
    fade_factor: float


def _format_exp(val: float) -> str:
    return f"{val:.2f}".rstrip("0").rstrip(".")


def _safe_denominator(raw: np.ndarray, subtract: float = 0.0) -> np.ndarray:
    den = np.asarray(raw, dtype=float)
    den = den - float(subtract)
    den[den < 1.0] = 1.0
    return den


def _apply_color_mode(
    mode: str,
    exponent: Optional[float],
    totals_raw: np.ndarray,
    totals_adjusted: np.ndarray,
    pairs_arr: np.ndarray,
    types_arr: np.ndarray,
    *,
    settings: ColorSettings,
) -> Tuple[np.ndarray, str]:
    mode_lower = mode.lower()
    if mode_lower == "pair_exp":
        exp_val = exponent if exponent is not None else 1.0
        denom = _safe_denominator(pairs_arr, subtract=settings.per_pair_subtract)
        denom = np.power(denom, exp_val)
        if not np.isclose(settings.denominator_exponent, 1.0):
            denom = np.power(denom, settings.denominator_exponent)
        vals = totals_raw / denom
        title = f"Total/Pares^{_format_exp(exp_val)}"
    elif mode_lower == "types_exp":
        exp_val = exponent if exponent is not None else 1.0
        denom = _safe_denominator(types_arr, subtract=settings.per_existing_subtract)
        denom = np.power(denom, exp_val)
        if not np.isclose(settings.denominator_exponent, 1.0):
            denom = np.power(denom, settings.denominator_exponent)
        vals = totals_adjusted / denom
        title = f"Total ajustado/Tipos^{_format_exp(exp_val)}"
    elif mode_lower == "raw_total":
        vals = totals_raw.copy()
        title = "Total bruto"
    else:
        raise ValueError(f"Modo de color no soportado: {mode}")

    if not np.isclose(settings.output_exponent, 1.0):
        vals = np.power(np.clip(vals, 0.0, None), settings.output_exponent)
    return vals, title


def generate_figures(
    payloads: Sequence[Dict[str, Any]],
    entries: List[ChordEntry],
    totals: np.ndarray,
    pairs: np.ndarray,
    preproc_cache: Dict[str, np.ndarray],
    dist_simplex_cache: Dict[str, np.ndarray],
    distance_cache: Dict[Tuple[str, str], np.ndarray],
    *,
    color_settings: ColorSettings,
    highlight_settings: HighlightSettings,
) -> List[Tuple[str, go.Figure]]:
    """Construye las figuras a partir de los payloads en paralelo."""

    if not payloads:
        return []

    totals = np.asarray(totals, dtype=float)
    pairs = np.asarray(pairs, dtype=float)

    def _build_single(payload: Dict[str, Any]) -> List[Tuple[str, go.Figure]]:
        scenario_name = payload["scenario"]
        preproc_id = payload["preproc_id"]
        metric = payload["metric"]
        reduction = payload["reduction"]
        figure_seed = payload["figure_seed"]
        embedding = np.asarray(payload["embedding"], dtype=float)
        base_matrix = (
            np.asarray(preproc_cache[preproc_id], dtype=float)
            if metric in BASE_VECTOR_METRICS
            else np.asarray(dist_simplex_cache[preproc_id], dtype=float)
        )
        vectors_adjusted = np.asarray(preproc_cache[preproc_id], dtype=float)
        totals_adj = np.sum(vectors_adjusted, axis=1)
        existing_counts = np.sum(
            vectors_adjusted > color_settings.existing_threshold,
            axis=1,
        ).astype(float)
        substitution_options = payload.get("substitution_options")

        color_modes: List[Tuple[str, Optional[float]]] = []
        if preproc_id == "identity":
            color_modes.append(("raw_total", None))
        else:
            for exp in color_settings.exponents:
                color_modes.append(("pair_exp", exp))
                color_modes.append(("types_exp", exp))

        fig_title = f"{scenario_name} (seed {figure_seed})"
        figs: List[Tuple[str, go.Figure]] = []
        for mode, exponent in color_modes:
            vals, ctitle = _apply_color_mode(
                mode,
                exponent,
                totals,
                totals_adj,
                pairs,
                existing_counts,
                settings=color_settings,
            )
            suffix = mode if exponent is None else f"{mode}_{int(round(exponent * 100)):03d}"
            fig = build_scatter_figure(
                embedding=embedding,
                entries=entries,
                color_values=vals,
                pair_counts=pairs,
                type_counts=existing_counts,
                vectors=base_matrix,
                adjusted_vectors=vectors_adjusted,
                title=fig_title,
                is_proposal=(preproc_id != "identity"),
                color_title=ctitle,
                meta={
                    "scenario": scenario_name,
                    "mode": mode,
                    "exponent": exponent,
                    "metric": metric,
                    "reduction": reduction,
                },
                substitution_options=substitution_options,
                highlight_settings=highlight_settings,
            )
            figs.append((f"{scenario_name}||{suffix}", fig))
        return figs

    max_workers = min(len(payloads), max(1, os.cpu_count() or 1))
    ordered: List[Tuple[str, go.Figure]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_build_single, payload): idx for idx, payload in enumerate(payloads)}
        partial_results: Dict[int, List[Tuple[str, go.Figure]]] = {}
        for future in as_completed(future_map):
            idx = future_map[future]
            partial_results[idx] = future.result()
        for idx in range(len(payloads)):
            ordered.extend(partial_results.get(idx, []))
    return ordered


def build_scatter_figure(
    embedding: np.ndarray,
    entries: List[ChordEntry],
    color_values: np.ndarray,
    pair_counts: np.ndarray,
    type_counts: np.ndarray,
    vectors: np.ndarray,
    adjusted_vectors: np.ndarray,
    title: str,
    *,
    is_proposal: bool = False,
    color_title: str = "Color",
    meta: Optional[Dict[str, Any]] = None,
    substitution_options: Optional[Dict[str, Any]] = None,
    highlight_settings: Optional[HighlightSettings] = None,
) -> go.Figure:
    highlight_cfg = {
        "threshold": highlight_settings.threshold if highlight_settings else 2000,
        "size_scale": highlight_settings.size_scale if highlight_settings else 1.35,
        "size_delta": highlight_settings.size_delta if highlight_settings else 3.0,
        "selected_opacity": highlight_settings.selected_opacity if highlight_settings else 0.95,
        "fade_factor": highlight_settings.fade_factor if highlight_settings else 0.25,
    }
    payload = build_scatter_payload(
        embedding=embedding,
        entries=entries,
        color_values=color_values,
        pair_counts=pair_counts,
        type_counts=type_counts,
        vectors=vectors,
        adjusted_vectors=adjusted_vectors,
        title=title,
        color_title=color_title,
        is_proposal=is_proposal,
        highlight=highlight_cfg,
        meta=meta,
        substitution_options=substitution_options,
    )
    return go.Figure(data=payload["data"], layout=payload["layout"])


__all__ = [
    "ColorSettings",
    "HighlightSettings",
    "generate_figures",
    "build_scatter_figure",
]
