"""Metric utilities for proposal comparisons."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform, jensenshannon
from scipy.optimize import linear_sum_assignment

from lab import kruskal_stress_1
from metrics import (
    compute_continuity,
    compute_knn_recall,
    compute_rank_correlation,
    compute_trustworthiness,
)

from .data import ChordEntry

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
EB_JSD_COMBO_METRIC = "eb_jsd_combo"
EB_JSD_COMBO_ALIASES = {
    EB_JSD_COMBO_METRIC,
    "eb_jsd",
    "d_combo",
    "combo",
}
EB_JSD_COMBO_DEFAULT_ALPHA = 0.20
EB_EUC_COMBO_METRIC = "eb_euc_combo"
EB_EUC_COMBO_ALIASES = {
    EB_EUC_COMBO_METRIC,
    "eb_euc",
    "d_combo_v2",
    "combo_v2",
}
EB_EUC_COMBO_DEFAULT_ALPHA = 0.20
# Pesos calibrados sobre población estructural diatónica (2-3 notas, max intervalo interno <= 12)
# para mejorar trustworthiness/continuity sin sacrificar stress.
STRUCTURAL_ROUGHNESS_WEIGHTS = (0.325, 0.299, 0.214, 0.162)
EPSILON = 1e-12
EUCLID_COS_BLEND_LAMBDA = 0.30
VOICELEADING_QUINTAS_WEIGHTS = (0.55, 0.25, 0.20)
VOICELEADING_GAP_PENALTY = 6.5
CIRCLE_OF_FIFTHS = np.array([0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5], dtype=int)
FIFTH_INDEX = {int(pc): idx for idx, pc in enumerate(CIRCLE_OF_FIFTHS)}


def _normalize_simplex_rows(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, None)
    sums = np.sum(clipped, axis=1, keepdims=True)
    return np.divide(clipped, sums, out=np.zeros_like(clipped), where=sums > EPSILON)


def _structural_roughness_distance(X: np.ndarray, dist_simplex: np.ndarray) -> np.ndarray:
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
            except Exception as exc:
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
    semitone_fold = abs(((a - b + 6.0) % 12.0) - 6.0)
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


# =====================================================================
# Expansion Bijection (EB) helpers for d_combo
# Source: experiments/chord_substitution/common.py:d_eb
# =====================================================================
from itertools import combinations_with_replacement as _cwr


def _circular_step(a: float, b: float) -> float:
    """Pure circular distance on R/12Z. True metric."""
    diff = abs(float(a) % 12.0 - float(b) % 12.0)
    return min(diff, 12.0 - diff)


def _distinct_notes_eb(notes: np.ndarray, tol: float = 1e-6) -> List[float]:
    """Get distinct notes (support) from MIDI values."""
    sorted_notes = sorted(float(n) for n in notes)
    if not sorted_notes:
        return []
    distinct = [sorted_notes[0]]
    for n in sorted_notes[1:]:
        if abs(n - distinct[-1]) > tol:
            distinct.append(n)
    return distinct


def _expansions_eb(notes: np.ndarray, K: int, tol: float = 1e-6) -> List[Tuple[float, ...]]:
    """Generate all expansions of chord to size K by duplicating support notes."""
    distinct = _distinct_notes_eb(notes, tol)
    m = len(distinct)
    if m == 0:
        return []
    if m >= K:
        return [tuple(distinct[:K])]
    extras = K - m
    expansions: set = set()
    for combo in _cwr(range(m), extras):
        exp = tuple(sorted(distinct + [distinct[i] for i in combo]))
        expansions.add(exp)
    return list(expansions)


def _pairwise_eb(notes_a: np.ndarray, notes_b: np.ndarray, tol: float = 1e-6) -> float:
    """Expansion Bijection dissimilarity between two chords.
    d_EB(A,B) = min over expansions to K=max(kappa(A),kappa(B))
    of the optimal bijective matching cost / K."""
    da = _distinct_notes_eb(notes_a, tol)
    db = _distinct_notes_eb(notes_b, tol)
    K = max(len(da), len(db))
    if K == 0:
        return 0.0
    exp_a = _expansions_eb(notes_a, K, tol)
    exp_b = _expansions_eb(notes_b, K, tol)
    best = float('inf')
    for ea in exp_a:
        for eb in exp_b:
            C = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    C[i, j] = _circular_step(ea[i], eb[j])
            ri, ci = linear_sum_assignment(C)
            cost = float(np.sum(C[ri, ci])) / K
            if cost < best:
                best = cost
    return best


def _eb_jsd_combo_distance(
    dist_simplex: np.ndarray,
    entries: Optional[Sequence[ChordEntry]],
    metric_params: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Composite dissimilarity: alpha * d_EB_hat + (1 - alpha) * d_JSD_hat.

    Both components are range-normalized to [0, 1].
    Default alpha = 0.20 (calibrated via 5-fold CV in Experiment 7).
    """
    if entries is None:
        raise ValueError(
            "La metrica 'eb_jsd_combo' requiere entries para acceder a notas absolutas."
        )
    n = dist_simplex.shape[0]
    if len(entries) != n:
        raise ValueError(
            "La metrica 'eb_jsd_combo' recibio un numero de entries distinto al de la poblacion."
        )

    params = metric_params or {}
    alpha = _param_float(params, ("alpha", "a"), default=EB_JSD_COMBO_DEFAULT_ALPHA)

    # --- d_EB component (pairwise on MIDI notes) ---
    notes_by_entry = [_entry_notes(entry) for entry in entries]
    pair_count = n * (n - 1) // 2
    d_eb = np.zeros(pair_count, dtype=float)
    idx = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d_eb[idx] = _pairwise_eb(notes_by_entry[i], notes_by_entry[j])
            idx += 1

    # --- d_JSD component (sqrt Jensen-Shannon on simplex) ---
    simplex_probs = _normalize_simplex_rows(dist_simplex)
    d_jsd = pdist(simplex_probs, metric=lambda u, v: float(jensenshannon(u, v, base=2.0)))

    # --- Range normalization ---
    eb_max = float(np.max(d_eb)) if np.any(d_eb > 0) else 1.0
    jsd_max = float(np.max(d_jsd)) if np.any(d_jsd > 0) else 1.0
    d_eb_hat = d_eb / (eb_max + EPSILON)
    d_jsd_hat = d_jsd / (jsd_max + EPSILON)

    return alpha * d_eb_hat + (1.0 - alpha) * d_jsd_hat


def _eb_euc_combo_distance(
    X: np.ndarray,
    entries: Optional[Sequence[ChordEntry]],
    metric_params: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Composite v2: alpha * d_EB_hat + (1 - alpha) * d_Euc_hat.

    Uses Euclidean distance on raw roughness vectors (Phi_raw)
    instead of sqrt(JSD) on Phi_simplex.
    Default alpha = 0.20.
    """
    if entries is None:
        raise ValueError(
            "La metrica 'eb_euc_combo' requiere entries para acceder a notas absolutas."
        )
    n = X.shape[0]
    if len(entries) != n:
        raise ValueError(
            "La metrica 'eb_euc_combo' recibio un numero de entries distinto al de la poblacion."
        )

    params = metric_params or {}
    alpha = _param_float(params, ("alpha", "a"), default=EB_EUC_COMBO_DEFAULT_ALPHA)

    # --- d_EB component ---
    notes_by_entry = [_entry_notes(entry) for entry in entries]
    pair_count = n * (n - 1) // 2
    d_eb = np.zeros(pair_count, dtype=float)
    idx = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d_eb[idx] = _pairwise_eb(notes_by_entry[i], notes_by_entry[j])
            idx += 1

    # --- d_Euc component (Euclidean on raw roughness vectors) ---
    d_euc = pdist(X, metric="euclidean")

    # --- Range normalization ---
    eb_max = float(np.max(d_eb)) if np.any(d_eb > 0) else 1.0
    euc_max = float(np.max(d_euc)) if np.any(d_euc > 0) else 1.0
    d_eb_hat = d_eb / (eb_max + EPSILON)
    d_euc_hat = d_euc / (euc_max + EPSILON)

    return alpha * d_eb_hat + (1.0 - alpha) * d_euc_hat


def _voiceleading_quintas_distance(
    dist_simplex: np.ndarray,
    entries: Optional[Sequence[ChordEntry]],
    metric_params: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    if entries is None:
        raise ValueError(
            "La métrica 'voiceleading_quintas' requiere entries para acceder a notas absolutas."
        )
    n = dist_simplex.shape[0]
    if len(entries) != n:
        raise ValueError(
            "La métrica 'voiceleading_quintas' recibió un número de entries distinto al tamaño de la población."
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


def metric_distance(
    metric: str,
    X: np.ndarray,
    dist_simplex: np.ndarray,
    *,
    entries: Optional[Sequence[ChordEntry]] = None,
    metric_params: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    metric = metric.lower()
    simplex_probs = _normalize_simplex_rows(dist_simplex)
    if metric == "cosine":
        return pdist(X, metric="cosine")
    if metric in {"euclid_cos_blend", "hybrid_ec30"}:
        d_e = pdist(X, metric="euclidean")
        d_c = pdist(X, metric="cosine")
        d_e = d_e / (float(np.mean(d_e)) + EPSILON)
        d_c = d_c / (float(np.mean(d_c)) + EPSILON)
        lam = EUCLID_COS_BLEND_LAMBDA
        return (1.0 - lam) * d_e + lam * d_c
    if metric == "js":
        def _js(u: np.ndarray, v: np.ndarray) -> float:
            return float(jensenshannon(u, v, base=2.0))
        return pdist(simplex_probs, _js)
    if metric == "hellinger":
        root = np.sqrt(simplex_probs)
        return pdist(root, metric="euclidean") / np.sqrt(2.0)
    if metric in STRUCTURAL_ROUGHNESS_ALIASES:
        return _structural_roughness_distance(X, dist_simplex)
    if metric in VOICELEADING_QUINTAS_ALIASES:
        return _voiceleading_quintas_distance(
            dist_simplex,
            entries,
            metric_params=metric_params,
        )
    if metric in EB_JSD_COMBO_ALIASES:
        return _eb_jsd_combo_distance(
            dist_simplex,
            entries,
            metric_params=metric_params,
        )
    if metric in EB_EUC_COMBO_ALIASES:
        return _eb_euc_combo_distance(
            X,
            entries,
            metric_params=metric_params,
        )
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
    "STRUCTURAL_ROUGHNESS_METRIC",
]
