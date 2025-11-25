"""Orquestación del ciclo de escenarios/propuestas."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from tools.proposals_pipeline.metrics import (
    metric_distance,
    parallel_worker_setup,
    run_scenario_task,
)


def run_experiment(
    *,
    entries: Sequence[Any],
    hist: np.ndarray,
    counts: np.ndarray,
    pairs: np.ndarray,
    seed_list: Sequence[int],
    reductions: Sequence[str],
    scenarios: Sequence[Mapping[str, Any]],
    deterministic: bool,
    jobs: int,
    mds_n_init: int,
    cpu_count: int | None = None,
) -> Dict[str, Any]:
    """Ejecuta los escenarios y retorna resultados agregados y payloads de figuras."""

    dist_simplex_cache: Dict[str, np.ndarray] = {}
    preproc_cache: Dict[str, np.ndarray] = {}
    distance_cache: Dict[Tuple[str, str], np.ndarray] = {}

    scenario_tasks: List[Dict[str, Any]] = []
    expected_order: List[str] = []

    for scenario in scenarios:
        preproc_id = scenario["preproc_id"]
        if preproc_id not in dist_simplex_cache:
            preproc_func = scenario["preproc_func"]
            kwargs = scenario["preproc_kwargs"]
            X, simplex = preproc_func(hist, counts=counts, pairs=pairs, **kwargs)
            preproc_cache[preproc_id] = X
            dist_simplex_cache[preproc_id] = simplex
        key = (preproc_id, scenario["metric"])
        if key not in distance_cache:
            X = preproc_cache[preproc_id]
            simplex = dist_simplex_cache[preproc_id]
            try:
                dist_condensed = metric_distance(scenario["metric"], X, simplex)
            except ValueError as exc:
                warnings.append(f"[skip] {scenario['name']}: {exc}")
                continue
            distance_cache[key] = dist_condensed
        for reduction in reductions:
            expected_order.append(f"{reduction}:{scenario['name']}")
        scenario_tasks.append(
            {
                "scenario": scenario,
                "reductions": list(reductions),
                "seed_list": list(seed_list),
                "deterministic": deterministic,
                "jobs": jobs,
                "mds_n_init": mds_n_init,
            }
        )

    results: List[Dict[str, Any]] = []
    per_seed_records: List[Dict[str, Any]] = []
    figure_payloads: List[Dict[str, Any]] = []
    warnings: List[str] = []
    scenario_time_details: List[Tuple[str, float]] = []

    if scenario_tasks:
        context = {
            "entries": entries,
            "preproc_cache": preproc_cache,
            "dist_simplex_cache": dist_simplex_cache,
            "distance_cache": distance_cache,
        }
        use_parallel = len(scenario_tasks) > 1 and (cpu_count or 1) > 1
        if use_parallel:
            max_workers = min(len(scenario_tasks), cpu_count or 1)
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=parallel_worker_setup,
                initargs=(context,),
            ) as executor:
                futures = [executor.submit(run_scenario_task, task) for task in scenario_tasks]
                for fut in as_completed(futures):
                    res = fut.result()
                    warnings.extend(res["warnings"])
                    results.extend(res["results"])
                    per_seed_records.extend(res["per_seed_records"])
                    figure_payloads.extend(res["figure_payloads"])
                    scenario_time_details.extend(res.get("timings", []))
        else:
            parallel_worker_setup(context)
            for task in scenario_tasks:
                res = run_scenario_task(task)
                warnings.extend(res["warnings"])
                results.extend(res["results"])
                per_seed_records.extend(res["per_seed_records"])
                figure_payloads.extend(res["figure_payloads"])
                scenario_time_details.extend(res.get("timings", []))

    return {
        "results": results,
        "per_seed_records": per_seed_records,
        "figure_payloads": figure_payloads,
        "warnings": warnings,
        "scenario_time_details": scenario_time_details,
        "expected_order": expected_order,
        "preproc_cache": preproc_cache,
        "dist_simplex_cache": dist_simplex_cache,
        "distance_cache": distance_cache,
    }


__all__ = ["run_experiment"]
