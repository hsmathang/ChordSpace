"""Funciones relacionadas con métricas, distancias y escenarios."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform, jensenshannon
from sklearn.manifold import Isomap, MDS, TSNE

try:  # pragma: no cover - UMAP opcional
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None

from tools.proposals_pipeline.population import ChordEntry
from lab import kruskal_stress_1
from metrics import (
    compute_continuity,
    compute_knn_recall,
    compute_rank_correlation,
    compute_trustworthiness,
)

BASE_VECTOR_METRICS = {"cosine", "euclidean", "l1", "l2", "cityblock", "manhattan"}
_PARALLEL_CONTEXT: Dict[str, Any] | None = None


def metric_distance(metric: str, X: np.ndarray, dist_simplex: np.ndarray) -> np.ndarray:
    """Calcula matrices de distancia condendadas según la métrica solicitada."""

    metric = metric.lower()
    if metric == "cosine":
        return pdist(X, metric="cosine")
    if metric == "euclidean":
        return pdist(X, metric="euclidean")
    if metric in {"l1", "cityblock", "manhattan"}:
        return pdist(X, metric="cityblock")
    if metric == "js":
        def _js(u, v):
            return float(jensenshannon(u, v, base=2))
        return pdist(dist_simplex, metric=_js)
    if metric == "hellinger":
        def _norm(u):
            return np.sqrt(np.clip(u, 0.0, None))
        dist = pdist(np.apply_along_axis(_norm, 1, dist_simplex), metric="euclidean")
        return dist / np.sqrt(2.0)
    raise ValueError(f"Métrica desconocida: {metric}")


def parallel_worker_setup(context: Dict[str, Any]) -> None:
    """Inicializa el contexto compartido para los trabajadores."""

    global _PARALLEL_CONTEXT
    _PARALLEL_CONTEXT = context


def run_scenario_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Ejecuta un escenario sobre todas las reducciones solicitadas."""

    if _PARALLEL_CONTEXT is None:  # pragma: no cover - validación
        raise RuntimeError("Parallel context not initialised.")

    entries = _PARALLEL_CONTEXT["entries"]
    preproc_cache: Dict[str, np.ndarray] = _PARALLEL_CONTEXT["preproc_cache"]
    dist_simplex_cache: Dict[str, np.ndarray] = _PARALLEL_CONTEXT["dist_simplex_cache"]
    distance_cache: Dict[Tuple[str, str], np.ndarray] = _PARALLEL_CONTEXT["distance_cache"]

    scenario = task["scenario"]
    reductions: Sequence[str] = task["reductions"]
    seed_list: Sequence[int] = task["seed_list"]
    deterministic: bool = task["deterministic"]
    jobs = task["jobs"]
    mds_n_init = task["mds_n_init"]

    scenario_name_base: str = scenario["name"]
    metric: str = scenario["metric"]
    preproc_id: str = scenario["preproc_id"]
    description: str = scenario["description"]

    X = np.asarray(preproc_cache[preproc_id], dtype=float)
    simplex = np.asarray(dist_simplex_cache[preproc_id], dtype=float)
    dist_condensed_base = distance_cache[(preproc_id, metric)]
    dist_matrix_base = squareform(dist_condensed_base)
    substitution_options = {
        "susti_basic": {
            "label": "susti_basic(vecino del espacio original)",
            "description": f"Vecinos según la métrica '{metric}' del escenario.",
            "distance_matrix": dist_matrix_base,
            "metric": metric,
        }
    }

    warnings: List[str] = []
    results: List[Dict[str, Any]] = []
    per_seed_records: List[Dict[str, Any]] = []
    figure_payloads: List[Dict[str, Any]] = []
    reduction_timings: List[Tuple[str, float]] = []

    for reduction in reductions:
        t_red_start = time.perf_counter()
        dist_matrix = dist_matrix_base
        base_matrix = X if metric in BASE_VECTOR_METRICS else simplex

        nn_top1, nn_top2 = evaluate_nn_hits(dist_matrix, entries, simplex)
        mix_mean, mix_max = evaluate_mixture_error(simplex, entries)

        seed_rows: List[Dict[str, Optional[float]]] = []
        figure_embedding: Optional[np.ndarray] = None
        figure_seed: Optional[int] = None

        for seed in seed_list:
            embedding = compute_embeddings(
                dist_condensed_base,
                reduction,
                seed,
                base_matrix=base_matrix,
                n_jobs=jobs,
                deterministic=deterministic,
                mds_n_init=mds_n_init,
            )
            metrics_summary = summarise_embedding_metrics(base_matrix, embedding, dist_matrix)
            row: Dict[str, Optional[float]] = {
                "scenario": f"{reduction}:{scenario_name_base}",
                "description": description,
                "metric": metric,
                "preproc_id": preproc_id,
                "seed": seed,
                "reduction": reduction,
                "nn_hit_top1": nn_top1,
                "nn_hit_top2": nn_top2,
                "mixture_l1_mean": mix_mean,
                "mixture_l1_max": mix_max,
                **metrics_summary,
            }
            seed_rows.append(row)
            if figure_embedding is None:
                figure_embedding = embedding
                figure_seed = seed

        if not seed_rows:
            continue

        summary = aggregate_seed_results(seed_rows, seed_list)
        summary.update(
            {
                "scenario": f"{reduction}:{scenario_name_base}",
                "description": description,
                "metric": metric,
                "preproc_id": preproc_id,
                "figure_seed": figure_seed,
                "reduction": reduction,
            }
        )
        results.append(summary)
        per_seed_records.extend(seed_rows)
        if figure_embedding is not None:
            figure_payloads.append(
                {
                    "scenario": f"{reduction}:{scenario_name_base}",
                    "preproc_id": preproc_id,
                    "metric": metric,
                    "description": description,
                    "reduction": reduction,
                    "figure_seed": figure_seed,
                    "embedding": figure_embedding,
                    "substitution_options": substitution_options,
                }
            )
        t_red_end = time.perf_counter()
        scenario_key = f"{reduction}:{scenario_name_base}"
        reduction_timings.append((scenario_key, t_red_end - t_red_start))

    return {
        "warnings": warnings,
        "results": results,
        "per_seed_records": per_seed_records,
        "figure_payloads": figure_payloads,
        "timings": reduction_timings,
    }


def compute_embeddings(
    dist_condensed: np.ndarray,
    reduction: str,
    seed: int,
    *,
    base_matrix: np.ndarray,
    n_jobs: Optional[int],
    deterministic: bool,
    mds_n_init: int,
) -> np.ndarray:
    """Genera un embedding 2D con la técnica de reducción solicitada."""

    reduction = reduction.upper()
    dist_matrix = squareform(dist_condensed)
    if reduction == "MDS":
        mds = MDS(
            n_components=2,
            metric=True,
            dissimilarity="precomputed",
            random_state=seed if deterministic else None,
            n_init=mds_n_init,
            n_jobs=n_jobs,
        )
        return mds.fit_transform(dist_matrix)
    if reduction == "TSNE":
        tsne = TSNE(
            n_components=2,
            metric="precomputed",
            perplexity=30,
            init="pca",
            learning_rate="auto",
            random_state=seed if deterministic else None,
        )
        return tsne.fit_transform(dist_matrix)
    if reduction == "ISOMAP":
        iso = Isomap(n_components=2, metric="precomputed")
        return iso.fit_transform(dist_matrix)
    if reduction == "UMAP":
        if umap is None:
            raise RuntimeError("UMAP no está disponible en este entorno.")
        reducer = umap.UMAP(
            n_components=2,
            metric="precomputed",
            init="spectral",
            random_state=seed if deterministic else None,
        )
        return reducer.fit_transform(dist_matrix)
    raise ValueError(f"Reducción desconocida: {reduction}")


def top_bins(dist_vector: np.ndarray, top_k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Obtiene los bins principales de un vector normalizado."""

    if dist_vector.ndim != 1:
        raise ValueError("dist_vector debe ser 1-D")
    idx = np.argsort(dist_vector)[::-1][:top_k]
    return idx, dist_vector[idx]


def evaluate_nn_hits(
    dist_matrix: np.ndarray,
    entries: List[ChordEntry],
    simplex: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """Evalúa recuperación de vecinos cercanos por cardinalidad."""

    n = len(entries)
    if n < 2:
        return None, None
    hits1 = []
    hits2 = []
    for idx in range(n):
        row = dist_matrix[idx]
        order = np.argsort(row)
        best = order[1] if order[0] == idx else order[0]
        second = order[2] if order[1] == idx else order[1]
        hits1.append(1.0 if entries[best].n_notes == entries[idx].n_notes else 0.0)
        hits2.append(1.0 if entries[second].n_notes == entries[idx].n_notes else 0.0)
    return float(np.mean(hits1)), float(np.mean(hits2))


def evaluate_mixture_error(
    simplex: np.ndarray,
    entries: List[ChordEntry],
) -> Tuple[Optional[float], Optional[float]]:
    """Calcula error promedio/máximo entre histogramas normalizados."""

    if simplex.shape[0] != len(entries):
        return None, None
    errors_mean: List[float] = []
    errors_max: List[float] = []
    for idx in range(simplex.shape[0]):
        bins, weights = top_bins(simplex[idx], top_k=2)
        target = np.zeros_like(simplex[idx])
        target[bins] = weights
        diff = np.abs(simplex[idx] - target)
        errors_mean.append(float(np.mean(diff)))
        errors_max.append(float(np.max(diff)))
    return float(np.mean(errors_mean)), float(np.mean(errors_max))


def summarise_embedding_metrics(
    X_original: np.ndarray,
    embedding: np.ndarray,
    dist_matrix: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Calcula métricas de calidad para un embedding."""

    trust = None
    try:
        trust = float(compute_trustworthiness(X_original, embedding))
    except Exception:
        trust = None
    try:
        cont = float(compute_continuity(X_original, embedding))
    except Exception:
        cont = None
    try:
        knn = float(compute_knn_recall(X_original, embedding))
    except Exception:
        knn = None
    try:
        rank_corr = float(compute_rank_correlation(X_original, embedding))
    except Exception:
        rank_corr = None
    try:
        stress = float(kruskal_stress_1(dist_matrix, squareform(pdist(embedding, metric="euclidean"))))
    except Exception:
        stress = None
    return {
        "trustworthiness": trust,
        "continuity": cont,
        "knn_recall": knn,
        "rank_corr": rank_corr,
        "stress": stress,
    }


def mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    clean = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return None, None
    arr = np.asarray(clean, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate_seed_results(seed_rows: List[Dict[str, Optional[float]]], seeds: Sequence[int]) -> Dict[str, Any]:
    metrics_keys = [
        "nn_hit_top1",
        "nn_hit_top2",
        "mixture_l1_mean",
        "mixture_l1_max",
        "trustworthiness",
        "continuity",
        "knn_recall",
        "rank_corr",
        "stress",
    ]
    summary: Dict[str, Any] = {"seeds": list(seeds)}
    for key in metrics_keys:
        values = [row.get(key) for row in seed_rows]
        mean_val, std_val = mean_std(values)
        summary[f"{key}_mean"] = mean_val
        summary[f"{key}_std"] = std_val
    return summary


__all__ = [
    "metric_distance",
    "parallel_worker_setup",
    "run_scenario_task",
    "compute_embeddings",
    "evaluate_nn_hits",
    "evaluate_mixture_error",
    "summarise_embedding_metrics",
    "aggregate_seed_results",
    "mean_std",
]
