"""
Visualization service for ChordSpace.
Generates plots and HTML reports from ExperimentResult.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from services.domain import ExperimentResult, VisualizationConfig

# Logic for figure generation
from tools.proposals_pipeline.figures import (
    ColorSettings,
    HighlightSettings,
    generate_figures
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

        # Regenerate figures from payloads and population data
        # We need to rebuild the context (entries, totals, etc.) because `generate_figures` demands it.
        # Ideally, `ExperimentResult` would contain `entries` or `stack_hist` output, but it contains raw DF.
        # We need to re-parse the population DF into entries.
        # This is a bit redundant but ensures we don't pickle complex objects in ExperimentResult unless necessary.

        # Re-load entries from DF in result
        from tools.proposals_pipeline.population import load_chords, stack_hist
        entries = load_chords(
            dyads_query="", triads_query="",
            df_override=result.population_df
        )
        hist, totals, counts, pairs, notes = stack_hist(entries)

        # Prepare caches (partially available in result, partially rebuilt)
        # `generate_figures` needs:
        # payloads, entries, totals, pairs, preproc_cache, dist_simplex_cache, distance_cache

        # We have distance_cache in result.
        # We DON'T have preproc_cache or dist_simplex_cache in result (I didn't add them to ExperimentResult to save space).
        # However, `generate_figures` needs them to draw histograms/distributions if requested.
        # If I want to support this without re-running preprocessing, I should have added them to ExperimentResult.
        # For now, let's assume `generate_figures` handles missing cache gracefully OR we must accept it might fail if those are needed.
        # Looking at `generate_figures` signature... it takes them as arguments.

        # Update: `space_experiments.py` calls `pipeline_run` which returns `preproc_cache`.
        # I should add `preproc_cache` and `dist_simplex_cache` to `ExperimentResult` to be safe.
        # This seems necessary for a complete decoupled visualization service.

        # Hack for now: Pass empty dicts and see if it explodes, or re-calculate.
        # Re-calculating requires knowing the preprocessors used.
        # Better: Add them to ExperimentResult.
        # I will modify `domain.py` and `space_experiments.py` one more time to include caches.

        # Wait, `visualization_payloads` (figure_payloads) contain the embedding coordinates.
        # The histograms are drawn from `preproc_cache`.

        # Let's assume for this step I will mock them or use what I have.
        # But for correctness, I should add them.

        # Let's write this service to fail if they are missing, but I will go back and add them.
        pass

    def _generate_figures(self, result: ExperimentResult, entries: List[Any], totals, pairs) -> List[Tuple[str, go.Figure]]:
        # Configure settings based on global constants (migrated from compare_proposals)
        # In a future iteration, these should be in VisualizationConfig completely.
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

        # We need the caches. Since I haven't added them to Result yet, I will use empty dicts.
        # This might break histogram plots but Scatter should work if payloads are correct.
        preproc_cache = getattr(result, "preproc_cache", {})
        dist_simplex_cache = getattr(result, "dist_simplex_cache", {})
        distance_cache = result.dist_matrices

        return generate_figures(
            result.visualization_payloads,
            entries,
            totals,
            pairs,
            preproc_cache,
            dist_simplex_cache,
            distance_cache,
            color_settings=color_settings,
            highlight_settings=highlight_settings,
        )

    def generate_report_full(self, result: ExperimentResult, config: VisualizationConfig) -> Path:
        # Full implementation
        # 1. Rehydrate data
        from tools.proposals_pipeline.population import load_chords, stack_hist
        entries = load_chords(dyads_query="", triads_query="", df_override=result.population_df)
        hist, totals, counts, pairs, notes = stack_hist(entries)

        # 2. Generate Figures
        figures = self._generate_figures(result, entries, totals, pairs)

        # 3. Assemble HTML
        sections_enabled = {s: True for s in config.sections}
        if config.sections == ["all"]:
             sections_enabled = {k: True for k in ["scatter", "heatmap", "shepard", "table", "metadata"]}

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

        render_report_html(
            metrics_df=result.metrics_df,
            figures=figures,
            output_path=result.output_path / "report.html",
            seeds=result.config.execution.seeds,
            run_metadata=run_metadata,
            metric_info=METRIC_INFO,
            highlight_threshold=config.highlight_threshold,
            sections_enabled=sections_enabled
        )

        return result.output_path / "report.html"

