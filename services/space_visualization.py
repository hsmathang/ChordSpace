"""
Visualization service for ChordSpace.
Generates plots and HTML reports from ExperimentResult.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import shutil
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from services.domain import ExperimentResult, VisualizationConfig

# Logic for figure generation
from tools.proposals_pipeline.figures import (
    ColorSettings,
    HighlightSettings,
    generate_figures,
    _generate_compact_labels
)
from tools.proposals_pipeline.population import load_chords, stack_hist
from tools.compare_proposals import (
    COLOR_PER_PAIR_SUBTRACT, COLOR_PER_NOTE_SUBTRACT, COLOR_PER_EXISTING_SUBTRACT,
    COLOR_EXISTING_THRESHOLD, COLOR_DEN_EXPONENT, COLOR_OUTPUT_EXPONENT, COLOR_EXPONENTS,
    FAMILY_HIGHLIGHT_THRESHOLD, FAMILY_HIGHLIGHT_SIZE_SCALE, FAMILY_HIGHLIGHT_SIZE_DELTA,
    FAMILY_HIGHLIGHT_SELECTED_OPACITY, FAMILY_HIGHLIGHT_UNSELECTED_OPACITY_FACTOR,
    METRIC_INFO
)
from tools.reporting import render_report_html

class VisualizationService:

    def generate_report(self, result: ExperimentResult, config: VisualizationConfig) -> Path:
        """
        Generates the HTML report for the given experiment result.
        Returns the path to the generated report.
        """
        if not result.output_path:
             raise ValueError("ExperimentResult must have an output_path to save the report.")

        report_path = result.output_path / "report.html"

        # This method is now redundant but kept for interface compatibility if needed.
        # It just delegates to generate_report_full logic essentially.
        return self.generate_report_full(result, config)

    def _generate_figures(
        self,
        result: ExperimentResult,
        sections_enabled: Dict[str, bool] | None = None,
        *,
        logger: Optional[Callable[[str], None]] = None,
    ) -> List[Tuple[str, go.Figure]]:
        # Configure settings based on global constants
        color_settings = ColorSettings(
            per_pair_subtract=COLOR_PER_PAIR_SUBTRACT,
            per_note_subtract=COLOR_PER_NOTE_SUBTRACT,
            per_existing_subtract=COLOR_PER_EXISTING_SUBTRACT,
            existing_threshold=COLOR_EXISTING_THRESHOLD,
            denominator_exponent=COLOR_DEN_EXPONENT,
            output_exponent=COLOR_OUTPUT_EXPONENT,
            exponents=COLOR_EXPONENTS,
        )
        highlight_settings = HighlightSettings(
            threshold=FAMILY_HIGHLIGHT_THRESHOLD,
            size_scale=FAMILY_HIGHLIGHT_SIZE_SCALE,
            size_delta=FAMILY_HIGHLIGHT_SIZE_DELTA,
            selected_opacity=FAMILY_HIGHLIGHT_SELECTED_OPACITY,
            fade_factor=FAMILY_HIGHLIGHT_UNSELECTED_OPACITY_FACTOR,
        )

        return generate_figures(
            result.visualization_payloads,
            result.entries,
            result.totals,
            result.pairs,
            result.preproc_cache,
            result.dist_simplex_cache,
            result.dist_matrices,
            color_settings=color_settings,
            highlight_settings=highlight_settings,
            sections_enabled=sections_enabled,
            logger=logger,
        )

    def generate_report_full(
        self,
        result: ExperimentResult,
        config: VisualizationConfig,
        *,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Path:
        # Full implementation
        if logger:
            logger("[visualizacion] usando resultados cacheados en ExperimentResult (idempotente)")

        # 1. Verify required data
        if not result.entries or result.totals is None:
             raise ValueError("ExperimentResult missing cached entries/totals. Re-run experiment.")

        # 2. Secciones activas según VisualizationConfig
        sections_enabled = {s: True for s in config.sections}
        if config.sections == ["all"]:
            sections_enabled = {k: True for k in ["scatter", "heatmap", "shepard", "table", "secondary_metrics", "metadata"]}
        if logger:
            enabled_txt = ", ".join(k for k, enabled in sections_enabled.items() if enabled)
            logger(f"[visualizacion] secciones activas: {enabled_txt}")
        # 3. Generate Figures (respeta secciones)
        figures = self._generate_figures(
            result,
            sections_enabled=sections_enabled,
            logger=logger,
        )

        # 4. Prepare Heatmap Data for JS (if eligible)
        heatmap_data: Optional[Dict[str, Any]] = None
        HEATMAP_LIMIT = 1500  # Hard limit for browser performance

        def _slugify(name: str) -> str:
            slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
            return slug or "scenario"

        if (
            sections_enabled.get("heatmap", False)
            and len(result.entries) <= HEATMAP_LIMIT
            and result.output_path is not None
        ):
            if logger:
                logger(f"[visualizacion] generando payload de heatmaps dinámicos (N={len(result.entries)})...")

            labels = _generate_compact_labels(result.entries)
            cardinalities = [
                int(getattr(entry, "n_notes", 0) or 0)
                for entry in result.entries
            ]

            heatmap_dir = result.output_path / "heatmaps"
            if heatmap_dir.exists():
                shutil.rmtree(heatmap_dir)
            heatmap_dir.mkdir(parents=True, exist_ok=True)

            heatmap_files: Dict[str, str] = {}
            slug_counts: Dict[str, int] = {}

            for payload in result.visualization_payloads:
                scenario_name = payload["scenario"]
                preproc_id = payload["preproc_id"]
                metric = payload["metric"]

                dist_condensed = result.dist_matrices.get((preproc_id, metric))
                if dist_condensed is None:
                    continue
                if scenario_name in heatmap_files:
                    continue

                condensed_list = (
                    dist_condensed.tolist()
                    if isinstance(dist_condensed, np.ndarray)
                    else list(dist_condensed)
                )

                base_slug = _slugify(scenario_name)
                count = slug_counts.get(base_slug, 0)
                slug_counts[base_slug] = count + 1
                slug = base_slug if count == 0 else f"{base_slug}_{count+1}"

                script_path = heatmap_dir / f"{slug}.js"
                payload_json = json.dumps({"condensed": condensed_list}, ensure_ascii=False)
                script_content = (
                    "(function(){window.__HEATMAP_PAYLOADS=window.__HEATMAP_PAYLOADS||{};"
                    f"window.__HEATMAP_PAYLOADS[{json.dumps(scenario_name)}]={payload_json};"
                    "})();"
                )
                script_path.write_text(script_content, encoding="utf-8")
                relative_path = script_path.relative_to(result.output_path).as_posix()
                heatmap_files[scenario_name] = relative_path

            if heatmap_files:
                heatmap_data = {
                    "metadata": {
                        "labels": labels,
                        "cardinalities": cardinalities,
                    },
                    "files": heatmap_files,
                }

        # Build minimal metadata if none provided, so the section is visible when selected.
        run_metadata = result.config.population.metadata or {}
        if not run_metadata:
            df = result.population_df
            total_rows = len(df)
            try:
                card_counts = df["n"].value_counts().sort_index()
                card_summary = ", ".join(f"{k}n:{v}" for k, v in card_counts.items())
                cards_list = sorted(card_counts.index.tolist())
            except Exception:
                card_summary = ""
                cards_list = []
            try:
                if "span_semitones" in df:
                    span_min = int(df["span_semitones"].min())
                    span_max = int(df["span_semitones"].max())
                else:
                    span_min = span_max = None
            except Exception:
                span_min = span_max = None

            filters_label_parts = []
            if card_summary:
                filters_label_parts.append(f"cardinalidades: {card_summary}")
            if span_min is not None and span_max is not None:
                filters_label_parts.append(f"span={span_min}-{span_max}")
            filters_label = " | ".join(filters_label_parts) if filters_label_parts else ""

            meta_src = result.config.population.metadata if isinstance(result.config.population.metadata, dict) else {}
            run_metadata = {
                "selection": {
                    "rows_selected": total_rows,
                    "rows_available": total_rows,
                    "mode": "archivo",
                    "payload_path": None,
                },
                "population": {
                    "descriptors": [
                        {
                            "label": "Población generada",
                            "rows": total_rows,
                            "mode": "combinatorial",
                            "combinatorial": {
                                "alphabet": meta_src.get("alphabet") or [],
                                "cardinalities": cards_list,
                                "octave_min": meta_src.get("octave_min"),
                                "octave_max": meta_src.get("octave_max"),
                                "structural_mode": meta_src.get("structural_mode"),
                            },
                            "database": {},
                            "filters": {"label": filters_label} if filters_label else {},
                        }
                    ]
                }
            }

        if logger:
            logger("[visualizacion] renderizando reporte HTML…")
        render_report_html(
            metrics_df=result.metrics_df,
            figures=figures,
            output_path=result.output_path / "report.html",
            seeds=result.config.execution.seeds,
            run_metadata=run_metadata,
            metric_info=METRIC_INFO,
            highlight_threshold=config.highlight_threshold,
            sections_enabled=sections_enabled,
            heatmap_data=heatmap_data  # Pass the extra payload
        )

        return result.output_path / "report.html"
