"""Funciones relacionadas con métricas, distancias y escenarios."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform, jensenshannon
from scipy.stats import linregress
from sklearn.manifold import Isomap, MDS, TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

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
MAX_SHEPARD_PAIRS = 20000  # Máximo de pares para cálculo de Shepard


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

    # Preparamos etiquetas para métricas de cluster (cardinalidad)
    labels_card = np.array([e.n_notes for e in entries])

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
        print(f"[pipeline] {scenario_name_base} -> {reduction} (n={len(entries)})")

        nn_top1, nn_top2 = evaluate_nn_hits(dist_matrix, entries, simplex)
        mix_mean, mix_max = evaluate_mixture_error(simplex, entries)

        seed_rows: List[Dict[str, Any]] = []
        figure_embedding: Optional[np.ndarray] = None
        figure_seed: Optional[int] = None

        # Captura hiperparámetros (comunes para todas las seeds)
        hyperparams = {}

        for seed in seed_list:
            embedding, params = compute_embeddings(
                dist_condensed_base,
                reduction,
                seed,
                base_matrix=base_matrix,
                n_jobs=jobs,
                deterministic=deterministic,
                mds_n_init=mds_n_init,
            )
            if not hyperparams:
                hyperparams = params

            print(f"[pipeline] terminado {reduction} seed={seed} (n={len(entries)})")

            metrics_summary = summarise_embedding_metrics(
                base_matrix,
                embedding,
                dist_matrix,
                dist_condensed_base,
                labels_card,
                seed
            )

            row: Dict[str, Any] = {
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
                "hyperparams": hyperparams,
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
                "hyperparams": hyperparams, # Agregar a summary también
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
                    "hyperparams": hyperparams,
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
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Genera un embedding 2D con la técnica de reducción solicitada.
    Retorna (embedding, hyperparams).
    """

    reduction = reduction.upper()
    dist_matrix = squareform(dist_condensed)
    n_samples = dist_matrix.shape[0]

    if reduction == "MDS":
        mds = MDS(
            n_components=2,
            metric=True,
            dissimilarity="precomputed",
            random_state=seed if deterministic else None,
            n_init=mds_n_init,
            n_jobs=n_jobs,
        )
        emb = mds.fit_transform(dist_matrix)
        params = {"n_init": mds_n_init}
        return emb, params

    if reduction == "TSNE":
        # Evita que perplexity supere el número de muestras; en datasets pequeños TSNE se queja.
        safe_perplexity = max(2, min(10, max(1, n_samples - 1)))
        tsne = TSNE(
            n_components=2,
            metric="precomputed",
            perplexity=safe_perplexity,
            init="random",
            learning_rate="auto",
            verbose=1,
            random_state=seed if deterministic else None,
        )
        emb = tsne.fit_transform(dist_matrix)
        kl_divergence = getattr(tsne, "kl_divergence_", None)
        params = {
            "perplexity": safe_perplexity,
            "n_iter": tsne.n_iter,
            "kl_divergence": kl_divergence
        }
        return emb, params

    if reduction == "ISOMAP":
        n_neighbors_iso = 5 # Default sklearn
        iso = Isomap(n_components=2, metric="precomputed", n_neighbors=n_neighbors_iso)
        emb = iso.fit_transform(dist_matrix)
        params = {"n_neighbors": n_neighbors_iso}
        return emb, params

    if reduction == "UMAP":
        if umap is None:
            raise RuntimeError("UMAP no está disponible en este entorno.")
        n_neighbors = max(2, min(15, n_samples - 1))
        n_epochs = 200
        reducer = umap.UMAP(
            n_components=2,
            metric="precomputed",
            init="spectral",
            random_state=seed if deterministic else None,
            n_neighbors=n_neighbors,
            n_epochs=n_epochs,
        )
        emb = reducer.fit_transform(dist_matrix)
        params = {"n_neighbors": n_neighbors, "n_epochs": n_epochs}
        return emb, params

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


def compute_shepard_metrics(
    dist_original_condensed: np.ndarray,
    embedding: np.ndarray,
    seed: int = 42
) -> Dict[str, Optional[float]]:
    """Calcula métricas de Shepard (slope, intercept, R2)."""

    # Calcular distancias en el embedding
    dist_emb_condensed = pdist(embedding, metric='euclidean')

    n_pairs = len(dist_original_condensed)

    # Subsampling si es necesario
    if n_pairs > MAX_SHEPARD_PAIRS:
        rng = np.random.RandomState(seed)
        indices = rng.choice(n_pairs, MAX_SHEPARD_PAIRS, replace=False)
        x = dist_original_condensed[indices]
        y = dist_emb_condensed[indices]
    else:
        x = dist_original_condensed
        y = dist_emb_condensed

    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return {
            "shepard_slope": float(slope),
            "shepard_intercept": float(intercept),
            "shepard_r2": float(r_value**2)
        }
    except Exception:
        return {
            "shepard_slope": None,
            "shepard_intercept": None,
            "shepard_r2": None
        }

def compute_cluster_metrics(
    embedding: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Optional[float]]:
    """Calcula Silhouette y Davies-Bouldin scores."""

    # Necesitamos al menos 2 clases y al menos 2 muestras para estas métricas
    if len(np.unique(labels)) < 2 or len(labels) < 2:
         return {
            "silhouette": None,
            "davies_bouldin": None
        }

    try:
        sil = float(silhouette_score(embedding, labels))
    except Exception:
        sil = None

    try:
        db = float(davies_bouldin_score(embedding, labels))
    except Exception:
        db = None

    return {
        "silhouette": sil,
        "davies_bouldin": db
    }

def summarise_embedding_metrics(
    X_original: np.ndarray,
    embedding: np.ndarray,
    dist_matrix: np.ndarray,
    dist_condensed: np.ndarray,
    labels: np.ndarray,
    seed: int
) -> Dict[str, Optional[float]]:
    """Calcula métricas de calidad para un embedding."""

    metrics: Dict[str, Optional[float]] = {}

    try:
        metrics["trustworthiness"] = float(compute_trustworthiness(X_original, embedding))
    except Exception:
        metrics["trustworthiness"] = None
    try:
        metrics["continuity"] = float(compute_continuity(X_original, embedding))
    except Exception:
        metrics["continuity"] = None
    try:
        metrics["knn_recall"] = float(compute_knn_recall(X_original, embedding))
    except Exception:
        metrics["knn_recall"] = None
    try:
        metrics["rank_corr"] = float(compute_rank_correlation(X_original, embedding))
    except Exception:
        metrics["rank_corr"] = None
    try:
        # Nota: aquí calculamos stress comparando dist_matrix con euclidean en embedding.
        # kruskal_stress_1 espera (observed, predicted).
        metrics["stress"] = float(kruskal_stress_1(dist_matrix, squareform(pdist(embedding, metric="euclidean"))))
    except Exception:
        metrics["stress"] = None

    # Shepard stats
    metrics.update(compute_shepard_metrics(dist_condensed, embedding, seed))

    # Cluster stats
    metrics.update(compute_cluster_metrics(embedding, labels))

    return metrics


def mean_std(values: Sequence[Any]) -> Tuple[Optional[float], Optional[float]]:
    clean = [float(v) for v in values if v is not None and not np.isnan(v) and isinstance(v, (int, float))]
    if not clean:
        return None, None
    arr = np.asarray(clean, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate_seed_results(seed_rows: List[Dict[str, Any]], seeds: Sequence[int]) -> Dict[str, Any]:
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
        "shepard_slope",
        "shepard_intercept",
        "shepard_r2",
        "silhouette",
        "davies_bouldin",
        "kl_divergence"
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
