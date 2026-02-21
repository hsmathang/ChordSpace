"""Funciones relacionadas con métricas, distancias y escenarios."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import pdist, squareform, jensenshannon
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress
from sklearn.manifold import Isomap, MDS, TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

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

BASE_VECTOR_METRICS = {
    "cosine",
    "euclidean",
    "euclid_cos_blend",
    "hybrid_ec30",
    "l1",
    "l2",
    "cityblock",
    "manhattan",
}
STRUCTURAL_ROUGHNESS_METRIC = "structural_roughness"
STRUCTURAL_ROUGHNESS_ALIASES = {
    STRUCTURAL_ROUGHNESS_METRIC,
    "structure_roughness",
    "srm",
}
VOICELEADING_QUINTAS_METRIC = "voiceleading_quintas"
VOICELEADING_QUINTAS_ALIASES = {
    VOICELEADING_QUINTAS_METRIC,
    "vl_quintas",
    "voiceleading5",
}
# Pesos calibrados sobre población estructural diatónica (2-3 notas, max intervalo interno <= 12)
# para mejorar trustworthiness/continuity sin sacrificar stress.
STRUCTURAL_ROUGHNESS_WEIGHTS = (0.325, 0.299, 0.214, 0.162)
EPSILON = 1e-12
EUCLID_COS_BLEND_LAMBDA = 0.30
MAX_SHEPARD_PAIRS = 20000  # Máximo de pares para cálculo de Shepard
VOICELEADING_QUINTAS_WEIGHTS = (0.55, 0.25, 0.20)
VOICELEADING_GAP_PENALTY = 6.5
CIRCLE_OF_FIFTHS = np.array([0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5], dtype=int)
FIFTH_INDEX = {int(pc): idx for idx, pc in enumerate(CIRCLE_OF_FIFTHS)}


def _normalize_simplex_rows(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, None)
    sums = np.sum(clipped, axis=1, keepdims=True)
    return np.divide(clipped, sums, out=np.zeros_like(clipped), where=sums > EPSILON)


def _structural_roughness_distance(X: np.ndarray, dist_simplex: np.ndarray) -> np.ndarray:
    """Distancia híbrida estructura+rugosidad robusta a cardinalidad."""

    probs = _normalize_simplex_rows(dist_simplex)
    presence = np.clip(np.asarray(dist_simplex, dtype=float), 0.0, None) > EPSILON

    totals = np.sum(np.clip(np.asarray(X, dtype=float), 0.0, None), axis=1)
    active_bins = np.maximum(np.sum(presence, axis=1).astype(float), 1.0)
    roughness_density = totals / active_bins
    log_roughness_density = np.log1p(roughness_density)

    d_structure = pdist(presence.astype(np.uint8), metric="jaccard")
    d_profile = pdist(np.sqrt(probs), metric="euclidean") / np.sqrt(2.0)
    d_per_dimension = 0.5 * pdist(probs, metric="cityblock")

    def _relative_total_delta(u: np.ndarray, v: np.ndarray) -> float:
        a = float(u[0])
        b = float(v[0])
        return abs(a - b) / (abs(a) + abs(b) + EPSILON)

    d_total = pdist(log_roughness_density[:, None], metric=_relative_total_delta)

    w_structure, w_profile, w_per_dimension, w_total = STRUCTURAL_ROUGHNESS_WEIGHTS
    return (
        w_structure * d_structure
        + w_profile * d_profile
        + w_per_dimension * d_per_dimension
        + w_total * d_total
    )


def _entry_notes(entry: ChordEntry) -> np.ndarray:
    acorde = getattr(entry, "acorde", None)
    notes_abs = getattr(acorde, "notes_abs", None) if acorde is not None else None
    notes: List[float] = []
    if isinstance(notes_abs, (list, tuple, np.ndarray)):
        for item in list(notes_abs):
            try:
                notes.append(float(int(round(float(item)))))
            except Exception:
                continue
    if notes:
        return np.asarray(sorted(notes), dtype=float)

    intervals = getattr(acorde, "intervals", []) if acorde is not None else []
    running = 0
    fallback: List[float] = [0.0]
    for step in intervals:
        try:
            running += int(step)
        except Exception:
            continue
        fallback.append(float(running))
    return np.asarray(sorted(fallback), dtype=float)


def _quintas_profile(notes: np.ndarray) -> np.ndarray:
    vec = np.zeros(12, dtype=float)
    for note in notes:
        pc = int(round(float(note))) % 12
        idx = FIFTH_INDEX.get(pc)
        if idx is not None:
            vec[idx] += 1.0
    smooth = 0.5 * vec + 0.25 * np.roll(vec, 1) + 0.25 * np.roll(vec, -1)
    total = float(np.sum(smooth))
    if total <= EPSILON:
        return np.full(12, 1.0 / 12.0, dtype=float)
    return smooth / total


def _param_float(
    metric_params: Mapping[str, float],
    keys: Sequence[str],
    *,
    default: float,
) -> float:
    for key in keys:
        if key in metric_params:
            try:
                return float(metric_params[key])
            except Exception as exc:  # pragma: no cover - defensivo
                raise ValueError(f"Parámetro inválido para {key}: {metric_params[key]}") from exc
    return float(default)


def _resolve_voiceleading_quintas_params(
    metric_params: Optional[Mapping[str, float]],
) -> Tuple[float, float, float, float]:
    params = metric_params or {}
    w_default_vl, w_default_q5, w_default_js = VOICELEADING_QUINTAS_WEIGHTS
    w_vl = _param_float(params, ("w_vl", "vl", "voiceleading"), default=w_default_vl)
    w_q5 = _param_float(params, ("w_q5", "q5", "quintas"), default=w_default_q5)
    w_js = _param_float(params, ("w_js", "js", "roughness"), default=w_default_js)
    gap_penalty = _param_float(params, ("gap_penalty", "gap"), default=VOICELEADING_GAP_PENALTY)

    if w_vl < 0 or w_q5 < 0 or w_js < 0:
        raise ValueError("Los pesos de voiceleading_quintas deben ser no negativos.")
    weight_sum = w_vl + w_q5 + w_js
    if weight_sum <= EPSILON:
        raise ValueError("La suma de pesos de voiceleading_quintas debe ser positiva.")
    if gap_penalty <= EPSILON:
        raise ValueError("gap_penalty debe ser mayor que 0 para voiceleading_quintas.")

    return (
        float(w_vl / weight_sum),
        float(w_q5 / weight_sum),
        float(w_js / weight_sum),
        float(gap_penalty),
    )


def _voice_step_cost(a: float, b: float) -> float:
    semitone_fold = abs(((a - b + 6.0) % 12.0) - 6.0)  # 0..6
    register_penalty = min(abs(a - b), 24.0) / 24.0
    return float(semitone_fold + 0.35 * register_penalty)


def _voice_leading_distance(notes_a: np.ndarray, notes_b: np.ndarray, gap_penalty: float) -> float:
    len_a = int(notes_a.size)
    len_b = int(notes_b.size)
    if len_a == 0 and len_b == 0:
        return 0.0
    size = max(len_a, len_b)
    costs = np.full((size, size), gap_penalty, dtype=float)
    for i in range(len_a):
        for j in range(len_b):
            costs[i, j] = _voice_step_cost(float(notes_a[i]), float(notes_b[j]))
    row_ind, col_ind = linear_sum_assignment(costs)
    total_cost = float(np.sum(costs[row_ind, col_ind]))
    normalized = total_cost / (size * gap_penalty)
    return float(np.clip(normalized, 0.0, 1.0))


def _voiceleading_quintas_distance(
    dist_simplex: np.ndarray,
    entries: Optional[Sequence[ChordEntry]],
    metric_params: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    if entries is None:
        raise ValueError(
            "La métrica 'voiceleading_quintas' requiere entradas con notas absolutas (entries)."
        )
    n = dist_simplex.shape[0]
    if len(entries) != n:
        raise ValueError(
            "La métrica 'voiceleading_quintas' recibió un número de entries distinto al de la población."
        )

    notes_by_entry = [_entry_notes(entry) for entry in entries]
    quintas_matrix = np.vstack([_quintas_profile(notes) for notes in notes_by_entry])
    simplex_probs = _normalize_simplex_rows(dist_simplex)

    d_js = pdist(simplex_probs, metric=lambda u, v: float(jensenshannon(u, v, base=2.0)))
    d_q5 = pdist(np.sqrt(quintas_matrix), metric="euclidean") / np.sqrt(2.0)

    w_vl, w_q5, w_js, gap_penalty = _resolve_voiceleading_quintas_params(metric_params)

    pair_count = n * (n - 1) // 2
    d_vl = np.zeros(pair_count, dtype=float)
    idx = 0
    for i in range(n - 1):
        notes_i = notes_by_entry[i]
        for j in range(i + 1, n):
            d_vl[idx] = _voice_leading_distance(notes_i, notes_by_entry[j], gap_penalty)
            idx += 1

    return w_vl * d_vl + w_q5 * d_q5 + w_js * d_js


def metric_reference_matrix(
    metric: str,
    X: np.ndarray,
    dist_simplex: np.ndarray,
    *,
    entries: Optional[Sequence[ChordEntry]] = None,
) -> np.ndarray:
    """Matriz de referencia para métricas de evaluación del embedding."""

    metric_key = metric.lower()
    if metric_key in BASE_VECTOR_METRICS:
        return X
    if metric_key in STRUCTURAL_ROUGHNESS_ALIASES:
        probs = _normalize_simplex_rows(dist_simplex)
        presence = (np.clip(np.asarray(dist_simplex, dtype=float), 0.0, None) > EPSILON).astype(float)
        totals = np.sum(np.clip(np.asarray(X, dtype=float), 0.0, None), axis=1, keepdims=True)
        active_bins = np.maximum(np.sum(presence, axis=1, keepdims=True), 1.0)
        log_total_density = np.log1p(totals / active_bins)
        return np.hstack([probs, presence, log_total_density])
    if metric_key in VOICELEADING_QUINTAS_ALIASES:
        probs = _normalize_simplex_rows(dist_simplex)
        if entries is None or len(entries) != probs.shape[0]:
            return probs
        quintas = np.vstack([_quintas_profile(_entry_notes(entry)) for entry in entries])
        return np.hstack([probs, quintas])
    return dist_simplex

@dataclass
class ScenarioResult:
    """Encapsulates the result of evaluating a scenario."""
    warnings: List[str] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    per_seed_records: List[Dict[str, Any]] = field(default_factory=list)
    figure_payloads: List[Dict[str, Any]] = field(default_factory=list)
    timings: List[Tuple[str, float]] = field(default_factory=list)

class ScenarioEvaluator:
    """Evaluates scenarios using a given context (population, caches)."""

    def __init__(self, context: Dict[str, Any]):
        self.context = context

    def evaluate(self, task: Dict[str, Any]) -> ScenarioResult:
        """Evaluates a single scenario task."""

        entries: List[ChordEntry] = self.context["entries"]
        preproc_cache: Dict[str, np.ndarray] = self.context["preproc_cache"]
        dist_simplex_cache: Dict[str, np.ndarray] = self.context["dist_simplex_cache"]
        distance_cache: Dict[Tuple[str, str], np.ndarray] = self.context["distance_cache"]

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
            base_matrix = metric_reference_matrix(metric, X, simplex, entries=entries)
            # print(f"[pipeline] {scenario_name_base} -> {reduction} (n={len(entries)})") # Moved to logger or suppressed

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

                # print(f"[pipeline] terminado {reduction} seed={seed} (n={len(entries)})")

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

            t_red_end = time.perf_counter()
            runtime_seconds = t_red_end - t_red_start

            summary = aggregate_seed_results(seed_rows, seed_list)
            summary.update(
                {
                    "scenario": f"{reduction}:{scenario_name_base}",
                    "description": description,
                    "metric": metric,
                    "preproc_id": preproc_id,
                    "figure_seed": figure_seed,
                    "reduction": reduction,
                    "hyperparams": hyperparams,
                    "runtime_seconds": runtime_seconds,
                    "runtime_seconds_per_seed": runtime_seconds / len(seed_rows),
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
            scenario_key = f"{reduction}:{scenario_name_base}"
            reduction_timings.append((scenario_key, runtime_seconds))

        return ScenarioResult(
            warnings=warnings,
            results=results,
            per_seed_records=per_seed_records,
            figure_payloads=figure_payloads,
            timings=reduction_timings
        )


def metric_distance(
    metric: str,
    X: np.ndarray,
    dist_simplex: np.ndarray,
    *,
    entries: Optional[Sequence[ChordEntry]] = None,
    metric_params: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Calcula matrices de distancia condendadas según la métrica solicitada."""

    metric = metric.lower()
    simplex_probs = _normalize_simplex_rows(dist_simplex)
    if metric == "cosine":
        return pdist(X, metric="cosine")
    if metric == "euclidean":
        return pdist(X, metric="euclidean")
    if metric in {"euclid_cos_blend", "hybrid_ec30"}:
        d_e = pdist(X, metric="euclidean")
        d_c = pdist(X, metric="cosine")
        d_e = d_e / (float(np.mean(d_e)) + EPSILON)
        d_c = d_c / (float(np.mean(d_c)) + EPSILON)
        lam = EUCLID_COS_BLEND_LAMBDA
        return (1.0 - lam) * d_e + lam * d_c
    if metric in VOICELEADING_QUINTAS_ALIASES:
        return _voiceleading_quintas_distance(
            dist_simplex,
            entries,
            metric_params=metric_params,
        )
    if metric in {"l1", "cityblock", "manhattan"}:
        return pdist(X, metric="cityblock")
    if metric == "js":
        def _js(u, v):
            return float(jensenshannon(u, v, base=2))
        return pdist(simplex_probs, metric=_js)
    if metric == "hellinger":
        dist = pdist(np.sqrt(simplex_probs), metric="euclidean")
        return dist / np.sqrt(2.0)
    if metric in STRUCTURAL_ROUGHNESS_ALIASES:
        return _structural_roughness_distance(X, dist_simplex)
    raise ValueError(f"Métrica desconocida: {metric}")


# Global context removed. Parallel setup now done via class/closure or careful passing if needed.
# For multiprocessing, we might need a top-level function that instantiates the evaluator.

_GLOBAL_EVALUATOR: Optional[ScenarioEvaluator] = None

def parallel_worker_setup(context: Dict[str, Any]) -> None:
    """Inicializa el contexto compartido para los trabajadores (compatibilidad con ProcessPoolExecutor)."""
    global _GLOBAL_EVALUATOR
    _GLOBAL_EVALUATOR = ScenarioEvaluator(context)


def run_scenario_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper function for parallel execution that uses the global evaluator instance."""
    if _GLOBAL_EVALUATOR is None:
        raise RuntimeError("Global evaluator not initialized in worker process.")

    # Convert ScenarioResult to dict for serialization if needed, or just return as dict
    # compatible with legacy callers.
    result = _GLOBAL_EVALUATOR.evaluate(task)
    return {
        "warnings": result.warnings,
        "results": result.results,
        "per_seed_records": result.per_seed_records,
        "figure_payloads": result.figure_payloads,
        "timings": result.timings
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
            "n_iter": getattr(tsne, "n_iter_", None),
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


def compute_cardinality_neighbor_hits(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    k_neighbors: int = 6,
) -> Dict[str, float]:
    """Calcula la proporción de vecinos con la misma cardinalidad por clase."""

    n = len(labels)
    if n <= 1:
        return {}
    k = min(max(k_neighbors, 1), n - 1)
    try:
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(embedding)
        _, indices = nbrs.kneighbors(embedding)
    except Exception:
        return {}

    indices = indices[:, 1:]
    hits: Dict[int, List[float]] = {}
    for idx in range(n):
        target_label = int(labels[idx])
        neigh_labels = labels[indices[idx]] if indices.shape[1] > 0 else np.array([], dtype=labels.dtype)
        ratio = float(np.mean(neigh_labels == target_label)) if neigh_labels.size else 0.0
        hits.setdefault(target_label, []).append(ratio)

    return {f"knn_hit_card_{card}": float(np.mean(values)) for card, values in hits.items()}


def compute_relative_rank_error(
    dist_condensed: np.ndarray,
    embedding: np.ndarray,
) -> Optional[float]:
    """Promedio de error relativo entre los rankings de distancias originales y embebidas."""

    if dist_condensed.size == 0:
        return None
    dist_emb = pdist(embedding, metric="euclidean")
    if dist_emb.size != dist_condensed.size:
        return None

    order_orig = np.argsort(dist_condensed)
    order_emb = np.argsort(dist_emb)
    ranks_orig = np.empty_like(order_orig)
    ranks_emb = np.empty_like(order_emb)
    ranks_orig[order_orig] = np.arange(len(order_orig))
    ranks_emb[order_emb] = np.arange(len(order_emb))
    denom = np.maximum(ranks_orig, 1)
    errors = np.abs(ranks_orig - ranks_emb) / denom
    return float(np.mean(errors))


def compute_embedding_variance_ratio(embedding: np.ndarray) -> Dict[str, Optional[float]]:
    """Relación de varianza explicada por cada componente del embedding."""

    if embedding.size == 0:
        return {"var_ratio_dim1": None, "var_ratio_dim2": None}

    try:
        cov = np.cov(embedding, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.clip(eigvals, 0.0, None))[::-1]
        total = float(np.sum(eigvals))
        if total <= 0:
            return {"var_ratio_dim1": None, "var_ratio_dim2": None}
        ratios = eigvals / total
        dim1 = float(ratios[0]) if ratios.size > 0 else None
        dim2 = float(ratios[1]) if ratios.size > 1 else None
        return {"var_ratio_dim1": dim1, "var_ratio_dim2": dim2}
    except Exception:
        return {"var_ratio_dim1": None, "var_ratio_dim2": None}


def compute_cardinality_logreg_accuracy(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    max_splits: int = 5,
) -> Optional[float]:
    """Evalúa qué tan separables son las cardinalidades mediante regresión logística."""

    unique_labels, counts = np.unique(labels, return_counts=True)
    if unique_labels.size < 2 or embedding.shape[0] <= unique_labels.size:
        return None

    max_possible = int(np.min(counts))
    n_splits = min(max_splits, max_possible)
    if n_splits < 2:
        return None

    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores: List[float] = []
        for train_idx, test_idx in skf.split(embedding, labels):
            model = LogisticRegression(max_iter=1000)
            model.fit(embedding[train_idx], labels[train_idx])
            scores.append(float(model.score(embedding[test_idx], labels[test_idx])))
        return float(np.mean(scores)) if scores else None
    except Exception:
        return None

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

    # Rank error y varianzas
    metrics["relative_rank_error"] = compute_relative_rank_error(dist_condensed, embedding)
    metrics.update(compute_embedding_variance_ratio(embedding))

    # Separabilidad supervisada por cardinalidad
    metrics["cardinality_logreg_acc"] = compute_cardinality_logreg_accuracy(embedding, labels)

    # Hits por cardinalidad (vecindario en el embedding)
    metrics.update(compute_cardinality_neighbor_hits(embedding, labels))

    # Uso de memoria del embedding
    metrics["embedding_mem_kb"] = float(embedding.nbytes / 1024.0)

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
        "kl_divergence",
        "relative_rank_error",
        "var_ratio_dim1",
        "var_ratio_dim2",
        "cardinality_logreg_acc",
        "embedding_mem_kb",
    ]
    card_metric_keys = sorted(
        {
            key
            for row in seed_rows
            for key in row.keys()
            if key.startswith("knn_hit_card_")
        }
    )
    metrics_keys.extend(card_metric_keys)
    summary: Dict[str, Any] = {"seeds": list(seeds)}
    for key in metrics_keys:
        values = [row.get(key) for row in seed_rows]
        mean_val, std_val = mean_std(values)
        summary[f"{key}_mean"] = mean_val
        summary[f"{key}_std"] = std_val
    return summary


__all__ = [
    "ScenarioEvaluator",
    "ScenarioResult",
    "metric_distance",
    "metric_reference_matrix",
    "parallel_worker_setup",
    "run_scenario_task",
    "compute_embeddings",
    "evaluate_nn_hits",
    "evaluate_mixture_error",
    "summarise_embedding_metrics",
    "aggregate_seed_results",
    "mean_std",
    "STRUCTURAL_ROUGHNESS_METRIC",
]
