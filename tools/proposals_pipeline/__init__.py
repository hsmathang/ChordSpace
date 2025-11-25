"""Paquete con los bloques del pipeline de comparación de propuestas."""

from .population import ChordEntry, load_chords, stack_hist
from .metrics import (
    BASE_VECTOR_METRICS,
    metric_distance,
    parallel_worker_setup,
    run_scenario_task,
    compute_embeddings,
    evaluate_nn_hits,
    evaluate_mixture_error,
    summarise_embedding_metrics,
    aggregate_seed_results,
    mean_std,
)
from .figures import ColorSettings, HighlightSettings, generate_figures
from .experiment import build_scenarios
from .runner import run_experiment

__all__ = [
    "ChordEntry",
    "load_chords",
    "stack_hist",
    "metric_distance",
    "compute_embeddings",
    "parallel_worker_setup",
    "run_scenario_task",
    "generate_figures",
    "ColorSettings",
    "HighlightSettings",
    "evaluate_nn_hits",
    "evaluate_mixture_error",
    "summarise_embedding_metrics",
    "aggregate_seed_results",
    "mean_std",
    "BASE_VECTOR_METRICS",
    "build_scenarios",
    "run_experiment",
]
