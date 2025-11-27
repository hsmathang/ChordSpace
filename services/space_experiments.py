"""
Service for running space experiments.
Orchestrates loading population, calculating roughness/metrics, dimensionality reduction,
and gathering results.
"""
from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import pandas as pd
import numpy as np

# Domain & Configs
from services.domain import (
    ExperimentConfig,
    ExperimentResult,
    PopulationConfig,
    RoughnessConfig,
    ReductionConfig
)
from services.population_store import get_population_store

# Existing Pipeline Logic
from tools.proposals_pipeline.population import load_chords, stack_hist, ChordEntry
from tools.proposals_pipeline.experiment import build_scenarios
from tools.proposals_pipeline.runner import run_experiment as pipeline_run
from tools.proposals_pipeline.metrics import (
    summarise_embedding_metrics
)
from tools.reporting.utils import compute_rank
from pre_process import ModeloSetharesVec # For raw loading if needed, but handled by load_chords logic usually

class ExperimentService:

    def run_experiment(self, config: ExperimentConfig, logger: Optional[Callable[[str], None]] = None) -> ExperimentResult:
        """
        Runs the full experiment pipeline based on the provided configuration.
        """
        start_time = time.perf_counter()
        timing_log = []
        def _log(msg: str) -> None:
            if logger:
                try:
                    logger(msg)
                except Exception:
                    pass

        # 1. Load Population
        # We need to adapt the `load_chords` logic to use our `PopulationStore` or vice-versa.
        # Currently `load_chords` in `tools` does a lot of heavy lifting (Sethares calc, wrapping in ChordEntry).
        # We should use `PopulationStore` to get the raw DF, then convert to ChordEntries.

        print(f"[Experiment] Loading population source: {config.population.source_type}")
        store = get_population_store(config.population)
        pop_data = store.load(config.population)
        _log(f"[población] cargada: {len(pop_data.dataframe)} acordes")

        # We need to convert this DataFrame into the list of ChordEntry objects expected by the pipeline
        # `load_chords` does this but it also fetches from DB. We want to inject the DF.
        # Let's use `load_chords` with `df_override`.

        # NOTE: `load_chords` is in `tools/proposals_pipeline/population.py`.
        # It takes `df_override`.
        entries = load_chords(
            dyads_query="", triads_query="", # Not used if override is present
            df_override=pop_data.dataframe
        )
        timing_log.append(("load_population", time.perf_counter() - start_time))
        _log(f"[población] preparada en {timing_log[-1][1]:.2f}s")

        # 2. Prepare histograms (Stacking)
        t0 = time.perf_counter()
        hist, totals, counts, pairs, notes = stack_hist(entries)
        timing_log.append(("stack_histograms", time.perf_counter() - t0))
        _log(f"[hist] stack en {timing_log[-1][1]:.2f}s")

        # 3. Build Scenarios (Roughness Proposals + Metrics)
        # Map RoughnessConfig to scenarios
        # We need to import PREPROCESSORS from somewhere or define them.
        # They are currently in `tools/compare_proposals.py`. Ideally they should be in domain or proposals_pipeline.
        # For now, to avoid circular deps or massive refactor, I will import them from `compare_proposals` IF possible,
        # but `compare_proposals` is a script. I should probably MOVE them to `tools/proposals_pipeline/transforms.py` or similar.
        # Or, I can re-import them here if I move them to `services/domain` or `services/transforms`.

        # DECISION: To avoid breaking `compare_proposals` yet, I will rely on the fact that I should probably
        # extract them. But for this step, let's assume I can import them from `tools.compare_proposals`
        # OR better, I will duplicate/move the registry to `tools/proposals_pipeline/preprocessors.py` in a bit.
        # For now, I will try to import from `tools.compare_proposals`.
        # WARNING: Importing from a script is bad practice. I will create `tools/proposals_pipeline/preprocessors.py` quickly.

        from tools.compare_proposals import PREPROCESSORS # This might run main if not careful? No, it has `if __name__`.

        scenarios = build_scenarios(
            proposals=config.roughness.proposals,
            metrics=config.roughness.metrics,
            preprocessors=PREPROCESSORS,
            include_identity=not config.roughness.disable_baseline
        )
        _log(f"[escenarios] proposals={len(config.roughness.proposals)} · métricas={len(config.roughness.metrics)} · escenarios={len(scenarios)}")

        # 4. Run Pipeline (Reductions & Metrics)
        t0 = time.perf_counter()
        _log(f"[pipeline] reducciones={config.reduction.methods} · seeds={config.execution.seeds}")
        run_output = pipeline_run(
            entries=entries,
            hist=hist,
            counts=counts,
            pairs=pairs,
            seed_list=config.execution.seeds,
            reductions=config.reduction.methods,
            scenarios=scenarios,
            deterministic=config.execution.deterministic,
            jobs=config.reduction.n_jobs,
            mds_n_init=config.reduction.n_init,
            cpu_count=None # Auto-detect
        )
        timing_log.append(("run_pipeline", time.perf_counter() - t0))
        _log(f"[pipeline] completado en {timing_log[-1][1]:.2f}s")
        scenario_times = run_output.get("scenario_time_details", [])
        for red_name, t_secs in scenario_times:
            _log(f"[pipeline] {red_name} -> {t_secs:.2f}s")
        if scenario_times:
            total_red_time = sum(t for _, t in scenario_times)
            timing_log.append(("reductions_total", total_red_time))
            _log(f"[pipeline] reducciones (suma) -> {total_red_time:.2f}s")

        # 5. Process Results
        t0 = time.perf_counter()
        results_list = run_output["results"]
        if not results_list:
             raise RuntimeError("No results generated from pipeline.")

        metrics_df = pd.DataFrame(results_list)
        metrics_df["rank"] = compute_rank(metrics_df)
        metrics_df.sort_values(by=["rank"], inplace=True)
        _log(f"[métricas] filas={len(metrics_df)} en {time.perf_counter() - t0:.2f}s")

        # Extract embeddings if needed?
        # The pipeline returns `figure_payloads` which contain coordinates.
        # `run_output` also has caches but maybe not raw embeddings in a dict structure.
        # We might want to construct `embeddings` dict from `figure_payloads`.

        embeddings = {}
        for payload in run_output["figure_payloads"]:
             # key: "{reduction}:{scenario}"
             # payload has 'embedding' which is the 2D array [N, 2]
             key = f"{payload['reduction']}:{payload['scenario']}"
             embeddings[key] = payload['embedding']

        timing_log.append(("process_results", time.perf_counter() - t0))

        # 6. Save Artifacts (if output_dir configured)
        artifacts_path = None
        if config.execution.output_dir:
            out_dir = Path(config.execution.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            artifacts_path = out_dir

            # Save Metrics
            metrics_df.to_csv(out_dir / "metrics.csv", index=False)
            metrics_df.to_json(out_dir / "metrics.json", orient="records", indent=2)

            # Save Config
            # (Simple JSON dump of config)
            # We skip this for now or use a helper

            # Save Embeddings? (Optional, maybe large)

        timing_log.append(("total", time.perf_counter() - start_time))
        _log(f"[total] {timing_log[-1][1]:.2f}s")

        return ExperimentResult(
            config=config,
            population_df=pop_data.dataframe,
            metrics_df=metrics_df,
            embeddings=embeddings,
            dist_matrices=run_output["distance_cache"],
            preproc_cache=run_output["preproc_cache"],
            dist_simplex_cache=run_output["dist_simplex_cache"],
            entries=entries,
            totals=totals,
            pairs=pairs,
            visualization_payloads=run_output["figure_payloads"],
            warnings=run_output["warnings"],
            timing=timing_log,
            output_path=artifacts_path
        )
