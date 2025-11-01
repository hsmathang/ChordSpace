"""High level orchestration for proposal comparison workflows."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS

from .data import ChordEntry, PopulationLoader, stack_hist
from .metrics import (
    BASE_VECTOR_METRICS,
    evaluate_mixture_error,
    evaluate_nn_hits,
    metric_distance,
)
from .registry import PREPROCESSORS, PreprocessorSpec
from .report import (
    COLOR_EXISTING_THRESHOLD,
    apply_color_mode,
    build_visualisation_payload,
    serialise_entries,
)


class ProposalsComparisonService:
    """Provide serialisable outputs for GUI integrations."""

    def __init__(self) -> None:
        self._loader = PopulationLoader()
        self._entries: list[ChordEntry] = []
        self._hist: Optional[np.ndarray] = None
        self._totals: Optional[np.ndarray] = None
        self._counts: Optional[np.ndarray] = None
        self._pairs: Optional[np.ndarray] = None
        self._notes: Optional[np.ndarray] = None
        self._last_adjusted: Optional[np.ndarray] = None
        self._last_simplex: Optional[np.ndarray] = None
        self._last_preprocessor: Optional[str] = None

    # ------------------------------------------------------------------ helpers
    def _require_population(self) -> None:
        if self._hist is None or not self._entries:
            raise RuntimeError("Population not prepared. Call prepare_population() first.")

    def _require_preprocessor(self) -> None:
        if self._last_adjusted is None or self._last_simplex is None:
            raise RuntimeError("No hay resultados de normalización. Ejecuta compute_metrics primero.")

    # ------------------------------------------------------------------ API
    def prepare_population(
        self,
        dyads_query: str,
        triads_query: str,
        *,
        sevenths_query: Optional[str] = None,
        dataframe=None,
    ) -> Dict[str, object]:
        if dataframe is not None:
            entries = self._loader.from_dataframe(dataframe)
        else:
            entries = self._loader.from_queries(dyads_query, triads_query, sevenths_query)

        hist, totals, counts, pairs, notes = stack_hist(entries)

        self._entries = entries
        self._hist = hist
        self._totals = totals
        self._counts = counts
        self._pairs = pairs
        self._notes = notes
        self._last_adjusted = None
        self._last_simplex = None
        self._last_preprocessor = None

        return {
            "entries": serialise_entries(entries),
            "totals": totals.tolist(),
            "pairs": pairs.tolist(),
            "notes": notes.tolist(),
        }

    def compute_metrics(self, preprocessor_id: str, metric: str) -> Dict[str, object]:
        self._require_population()
        spec = self._get_preprocessor(preprocessor_id)

        adjusted, simplex = spec.function(
            self._hist,
            counts=self._counts,
            pairs=self._pairs,
            notes=self._notes,
            **spec.params,
        )
        self._last_adjusted = np.asarray(adjusted, dtype=float)
        self._last_simplex = np.asarray(simplex, dtype=float)
        self._last_preprocessor = preprocessor_id

        base_matrix = (
            self._last_adjusted if metric.lower() in BASE_VECTOR_METRICS else self._last_simplex
        )
        dist_condensed = metric_distance(metric, base_matrix, self._last_simplex)
        dist_matrix = squareform(dist_condensed)

        nn_top1, nn_top2 = evaluate_nn_hits(dist_matrix, self._entries, self._last_simplex)
        mix_mean, mix_max = evaluate_mixture_error(self._last_simplex, self._entries)

        return {
            "preprocessor": preprocessor_id,
            "preprocessor_label": spec.label,
            "preprocessor_params": dict(spec.params),
            "metric": metric,
            "distances": dist_condensed.tolist(),
            "nn_hit_top1": nn_top1,
            "nn_hit_top2": nn_top2,
            "mixture_l1_mean": mix_mean,
            "mixture_l1_max": mix_max,
        }

    def render_visualisations(
        self,
        *,
        color_mode: str = "pair_exp",
        exponent: Optional[float] = 1.0,
        embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        self._require_population()
        self._require_preprocessor()

        if embedding is None:
            dist_condensed = metric_distance(
                "euclidean", self._last_adjusted, self._last_simplex
            )
            dist_matrix = squareform(dist_condensed)
            reducer = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=42,
                normalized_stress=False,
            )
            embedding = reducer.fit_transform(dist_matrix)
        else:
            embedding = np.asarray(embedding, dtype=float)

        existing_counts = np.sum(
            self._last_adjusted > COLOR_EXISTING_THRESHOLD, axis=1
        ).astype(float)
        totals_adjusted = np.sum(self._last_adjusted, axis=1)
        color_values, color_title = apply_color_mode(
            color_mode,
            exponent,
            self._totals,
            totals_adjusted,
            self._pairs,
            existing_counts,
        )

        extras: Dict[str, object] = {
            "color_mode": color_mode,
            "exponent": exponent,
            "pair_counts": self._pairs.tolist() if self._pairs is not None else [],
            "type_counts": existing_counts.tolist(),
            "preprocessor": self._last_preprocessor,
        }
        return build_visualisation_payload(
            embedding,
            self._entries,
            color_values,
            color_title,
            extras=extras,
        )

    # ------------------------------------------------------------------ internals
    def _get_preprocessor(self, preprocessor_id: str) -> PreprocessorSpec:
        normalised = preprocessor_id.lower()
        if normalised not in PREPROCESSORS:
            raise KeyError(f"Preprocesador desconocido: {preprocessor_id}")
        return PREPROCESSORS[normalised]


__all__ = ["ProposalsComparisonService"]
