"""Metric utilities for proposal comparisons."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform, jensenshannon

from lab import kruskal_stress_1
from metrics import (
    compute_continuity,
    compute_knn_recall,
    compute_rank_correlation,
    compute_trustworthiness,
)

from .data import ChordEntry

BASE_VECTOR_METRICS = {"cosine", "euclidean", "l1", "l2", "cityblock", "manhattan"}


def metric_distance(metric: str, X: np.ndarray, dist_simplex: np.ndarray) -> np.ndarray:
    metric = metric.lower()
    if metric == "cosine":
        return pdist(X, metric="cosine")
    if metric == "js":
        def _js(u: np.ndarray, v: np.ndarray) -> float:
            su = float(np.sum(u))
            sv = float(np.sum(v))
            uu = (u / su) if su > 0 else u
            vv = (v / sv) if sv > 0 else v
            return jensenshannon(uu, vv, base=2.0)
        return pdist(dist_simplex, _js)
    if metric == "hellinger":
        def _norm(u: np.ndarray) -> np.ndarray:
            s = float(np.sum(u))
            return (u / s) if s > 0 else u
        root = np.sqrt(np.apply_along_axis(_norm, 1, dist_simplex))
        return pdist(root, metric="euclidean") / np.sqrt(2.0)
    if metric in {"euclidean", "l2"}:
        return pdist(X, metric="euclidean")
    if metric in {"l1", "cityblock", "manhattan"}:
        return pdist(X, metric="cityblock")
    raise ValueError(f"Métrica no soportada: {metric}")


def top_bins(dist_vector: np.ndarray, top_k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    if not np.any(dist_vector > 0):
        return np.array([], dtype=int), np.array([], dtype=float)
    idx_sorted = np.argsort(dist_vector)[::-1][:top_k]
    weights = dist_vector[idx_sorted]
    positive_mask = weights > 0
    return idx_sorted[positive_mask], weights[positive_mask]


def evaluate_nn_hits(
    dist_matrix: np.ndarray,
    entries: Sequence[ChordEntry],
    simplex: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    if not any(e.n_notes == 3 for e in entries):
        return None, None
    hits_top1: List[int] = []
    hits_top2: List[int] = []
    for idx, entry in enumerate(entries):
        if entry.n_notes != 3:
            continue
        row = dist_matrix[idx].copy()
        row[idx] = np.inf
        neighbor = int(np.argmin(row))
        if entries[neighbor].n_notes != 2:
            hits_top1.append(0)
            hits_top2.append(0)
            continue
        bins, _ = top_bins(simplex[idx], top_k=2)
        if bins.size == 0:
            hits_top1.append(0)
            hits_top2.append(0)
            continue
        target_bins = set(int(b) for b in bins)
        neighbor_bin = entries[neighbor].dyad_bin
        hit1 = 1 if neighbor_bin is not None and neighbor_bin == int(bins[0]) else 0
        hit_any = 1 if neighbor_bin is not None and neighbor_bin in target_bins else 0
        hits_top1.append(hit1)
        hits_top2.append(hit_any)
    if hits_top1:
        top1_rate = float(np.mean(hits_top1))
        top2_rate = float(np.mean(hits_top2))
    else:
        top1_rate = None
        top2_rate = None
    return top1_rate, top2_rate


def evaluate_mixture_error(
    simplex: np.ndarray,
    entries: Sequence[ChordEntry],
) -> Tuple[Optional[float], Optional[float]]:
    errors: List[float] = []
    for idx, entry in enumerate(entries):
        if entry.n_notes != 3:
            continue
        bins, weights = top_bins(simplex[idx], top_k=2)
        if bins.size == 0:
            continue
        weights = weights / weights.sum()
        mixture = np.zeros(12, dtype=float)
        for bin_idx, weight in zip(bins, weights):
            mixture[int(bin_idx)] = weight
        error = float(np.linalg.norm(simplex[idx] - mixture, ord=1))
        errors.append(error)
    if not errors:
        return None, None
    return float(np.mean(errors)), float(np.max(errors))


def summarise_embedding_metrics(
    X_original: np.ndarray,
    embedding: np.ndarray,
    dist_matrix: np.ndarray,
) -> Dict[str, Optional[float]]:
    def _safe(callable_):
        try:
            return float(callable_())
        except Exception:
            return None

    return {
        "trustworthiness": _safe(lambda: compute_trustworthiness(X_original, embedding)),
        "continuity": _safe(lambda: compute_continuity(X_original, embedding)),
        "knn_recall": _safe(lambda: compute_knn_recall(X_original, embedding)),
        "rank_corr": _safe(lambda: compute_rank_correlation(X_original, embedding)),
        "stress": _safe(
            lambda: kruskal_stress_1(dist_matrix, squareform(pdist(embedding, metric="euclidean")))
        ),
    }


def aggregate_seed_results(
    seed_rows: Iterable[Mapping[str, Optional[float]]],
    seed_list: Sequence[int],
) -> Dict[str, Optional[float]]:
    numeric_fields = {
        key
        for row in seed_rows
        for key, value in row.items()
        if isinstance(value, (int, float)) and key not in {"seed"}
    }
    summary: Dict[str, Optional[float]] = {}
    rows = list(seed_rows)
    for field in numeric_fields:
        values = [
            float(row[field])
            for row in rows
            if row.get(field) is not None and isinstance(row.get(field), (int, float))
        ]
        summary[field] = float(np.mean(values)) if values else None
    summary["seeds"] = list(seed_list)
    return summary


__all__ = [
    "BASE_VECTOR_METRICS",
    "metric_distance",
    "top_bins",
    "evaluate_nn_hits",
    "evaluate_mixture_error",
    "summarise_embedding_metrics",
    "aggregate_seed_results",
]
