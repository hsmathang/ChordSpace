"""
Controller for the Experiment Launcher.
Encapsulates business logic, state management for experiments, and service orchestration.
"""
from __future__ import annotations

import queue
import threading
import tempfile
import datetime as dt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
import re

# Domain & Configs
from services.domain import (
    ExperimentConfig,
    PopulationConfig,
    RoughnessConfig,
    ReductionConfig,
    ExecutionConfig,
    VisualizationConfig,
    ExperimentResult,
)
from services.space_experiments import ExperimentService
from services.space_visualization import VisualizationService
from services.combinatorial_generator import generate_combinatorial_chords
from services.population_filter import filter_dataframe
from tools.population_utils import dedupe_population
from tools import data_access

class ExperimentController:
    def __init__(self, log_queue: queue.Queue):
        self.log_queue = log_queue
        self.experiment_service = ExperimentService()
        self.visualization_service = VisualizationService()
        self._temp_payloads: List[Path] = []

        # State
        self.final_population_df: Optional[pd.DataFrame] = None
        self.temporal_population_df: Optional[pd.DataFrame] = None

        self.last_experiment_result: Optional[ExperimentResult] = None
        self.last_report_path: Optional[Path] = None

    def generate_combinatorial_population(
        self,
        alphabet: List[int],
        octave_min: int,
        octave_max: int,
        cardinalities: List[int],
        structural_mode: bool,
        filters: Optional[data_access.ChordFilters] = None
    ) -> pd.DataFrame:
        """Generates a population using the combinatorial generator."""
        df = generate_combinatorial_chords(
            alphabet, octave_min, octave_max, cardinalities, structural_mode
        )

        if filters:
             df = filter_dataframe(df, filters)

        self.temporal_population_df = df
        return df

    def add_temporal_to_final(self) -> Tuple[int, int]:
        """Appends temporal population to final population. Returns (added_count, total_count)."""
        if self.temporal_population_df is None or self.temporal_population_df.empty:
            return 0, len(self.final_population_df) if self.final_population_df is not None else 0

        if self.final_population_df is None:
            self.final_population_df = self.temporal_population_df.copy()
        else:
            self.final_population_df = pd.concat(
                [self.final_population_df, self.temporal_population_df], ignore_index=True
            )

        self.final_population_df, _ = dedupe_population(self.final_population_df)
        self.final_population_df.reset_index(drop=True, inplace=True)

        return len(self.temporal_population_df), len(self.final_population_df)

    def clear_final_population(self) -> None:
        self.final_population_df = None

    def run_experiment_async(
        self,
        config: ExperimentConfig,
        vis_config: VisualizationConfig
    ) -> threading.Thread:
        """Starts the experiment in a background thread."""
        thread = threading.Thread(
            target=self._run_service_thread,
            args=(config, vis_config),
            daemon=True
        )
        thread.start()
        return thread

    def _run_service_thread(self, exp_config: ExperimentConfig, vis_config: VisualizationConfig) -> None:
        try:
            self._log("compare_log", f"Iniciando experimento: {exp_config.name}\n")
            self._log("compare_log", f"[resumen] proposals={len(exp_config.roughness.proposals)} · métricas={len(exp_config.roughness.metrics)} · reducciones={len(exp_config.reduction.methods)} · seeds={len(exp_config.execution.seeds)}\n")
            self._log("progress", (10.0, "Ejecutando pipeline..."))

            def _service_logger(msg: str) -> None:
                self._log("compare_log", msg if msg.endswith("\n") else msg + "\n")

            result = self.experiment_service.run_experiment(exp_config, logger=_service_logger)
            self.last_experiment_result = result

            self._log("compare_log", f"[población] {len(result.population_df)} acordes\n")
            for stage, secs in (result.timing or []):
                self._log("compare_log", f"[tiempo] {stage}: {secs:.2f}s\n")

            if vis_config.skip_html_report:
                self._log("compare_log", "Pipeline completado. Exportando plots/tablas separados (sin HTML unificado)...\n")
            else:
                self._log("compare_log", "Pipeline completado. Generando reporte...\n")
            self._log("progress", (80.0, "Generando visualizaciones..."))

            report_path = self.visualization_service.generate_report_full(result, vis_config, logger=_service_logger)
            self.last_report_path = report_path

            if vis_config.skip_html_report:
                self._log("compare_log", f"Artefactos generados: {report_path}\n")
            else:
                self._log("compare_log", f"Reporte generado: {report_path}\n")
            self._log("compare_status", "Completado")
            self._log("progress", (100.0, "Listo"))

        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._log("compare_error", str(exc))
            self._log("error", str(exc))
        finally:
            self._log("done", None)

    def _log(self, kind: str, payload: Any) -> None:
        self.log_queue.put((kind, payload))

    def write_population_json(self, df: pd.DataFrame) -> str:
        tmp = tempfile.NamedTemporaryFile(prefix="chordspace_population_", suffix=".jsonl", delete=False)
        try:
            df.to_json(tmp.name, orient="records", lines=True, date_format="iso")
        finally:
            tmp.close()
        path = Path(tmp.name)
        self._temp_payloads.append(path)
        return str(path)

    def cleanup_temp_files(self) -> None:
        for p in self._temp_payloads:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        self._temp_payloads.clear()

    def get_query_registry(self) -> Dict[str, Any]:
        from tools.query_registry import get_all_queries
        registry = get_all_queries()
        # Ensure preset names are included even if not in DB registry (mocked behavior from app_new)
        for name in data_access.list_preset_names():
            if name not in registry:
                registry[name] = {"sql": "<generated by data_access>", "source": "preset"}
        return dict(sorted(registry.items()))

    def get_default_output_dir(self) -> Path:
        return self.build_run_output_dir(self.get_default_output_root())

    def get_default_output_root(self) -> Path:
        return Path("outputs") / "gui_runs"

    def build_run_output_dir(self, base_dir: Path | str | None = None) -> Path:
        root = Path(base_dir) if base_dir is not None else self.get_default_output_root()
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = root / timestamp
        suffix = 2
        while candidate.exists():
            candidate = root / f"{timestamp}_{suffix:02d}"
            suffix += 1
        return candidate
