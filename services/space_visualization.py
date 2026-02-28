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
from tools.reporting.utils import compute_rank


def midi_to_freqs(notes: List[int]) -> List[float]:
    """Converts a list of MIDI note numbers to frequencies (Hz)."""
    return [440.0 * (2.0 ** ((n - 69) / 12.0)) for n in notes]


NOTE_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def _midi_to_note_name(note: int) -> str:
    midi = int(note)
    return f"{NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"


def _entry_axis_label(entry: Any, index: int) -> str:
    acorde = getattr(entry, "acorde", None)
    base_name = getattr(acorde, "name", None) or getattr(entry, "identity_name", None) or f"Chord {index}"
    intervals = getattr(acorde, "intervals", []) if acorde is not None else []
    try:
        interval_txt = "[" + ",".join(str(int(i)) for i in intervals) + "]" if intervals else ""
    except Exception:
        interval_txt = ""

    notes_abs = getattr(acorde, "notes_abs", None) if acorde is not None else None
    note_names: List[str] = []
    if isinstance(notes_abs, (list, tuple, np.ndarray)):
        for note in list(notes_abs):
            try:
                note_names.append(_midi_to_note_name(int(round(float(note)))))
            except Exception:
                continue
    notes_txt = f" ({','.join(note_names)})" if note_names else ""
    return f"{base_name}{(' ' + interval_txt) if interval_txt else ''}{notes_txt}"


def _build_heatmap_axis_labels(entries: List[Any]) -> List[str]:
    return [_entry_axis_label(entry, idx) for idx, entry in enumerate(entries)]


def build_audio_descriptors(result: ExperimentResult) -> Dict[str, Any]:
    """Builds a dictionary mapping chord index to audio properties (frequencies, label)."""
    chords: Dict[int, Dict[str, Any]] = {}
    for idx, entry in enumerate(result.entries):
        acorde = getattr(entry, "acorde", None)
        if acorde is None:
            continue

        freqs = getattr(acorde, "frequencies", None)
        if not freqs:
            notes_abs = getattr(acorde, "notes_abs", None) or []
            freqs = midi_to_freqs(notes_abs)

        try:
            freqs_list = [float(f) for f in freqs]
        except Exception:
            continue

        chords[idx] = {
            "freqs": freqs_list,
            "label": entry.identity_name or getattr(acorde, "name", "") or f"Chord {idx}",
            "n_notes": int(getattr(entry, "n_notes", len(freqs_list))),
        }

    return {
        "chords": chords,
        "config": {
            "sampleRate": 44100,
        },
    }


class VisualizationService:
    @staticmethod
    def _slugify_filename(name: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        return slug or "figure"

    def _export_split_artifacts(
        self,
        result: ExperimentResult,
        figures: List[Tuple[str, go.Figure]],
        *,
        sections_enabled: Dict[str, bool],
        logger: Optional[Callable[[str], None]] = None,
    ) -> Path:
        if result.output_path is None:
            raise ValueError("ExperimentResult must have an output_path to export artifacts.")

        output_dir = result.output_path
        plots_dir = output_dir / "plots"
        tables_dir = output_dir / "tables"
        plots_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        include_section = lambda key, default=True: bool(sections_enabled.get(key, default))
        manifest: Dict[str, Any] = {"plots": [], "tables": []}

        if include_section("table", True) and result.metrics_df is not None and not result.metrics_df.empty:
            ranked_df = result.metrics_df.copy()
            ranked_df["rank"] = compute_rank(ranked_df)
            ranked_df.sort_values(by=["rank"], inplace=True)
            ranked_csv_path = tables_dir / "metrics_ranked.csv"
            ranked_json_path = tables_dir / "metrics_ranked.json"
            ranked_df.to_csv(ranked_csv_path, index=False)
            ranked_df.to_json(ranked_json_path, orient="records", indent=2)
            manifest["tables"].append({"name": "metrics_ranked", "csv": str(ranked_csv_path.name), "json": str(ranked_json_path.name)})

        run_metadata = result.config.population.metadata or {}
        if include_section("metadata", True) and isinstance(run_metadata, dict) and run_metadata:
            metadata_path = tables_dir / "run_metadata.json"
            metadata_path.write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            manifest["tables"].append({"name": "run_metadata", "json": str(metadata_path.name)})

        exported = 0
        skipped_png = 0
        seen: Dict[str, int] = {}
        scatter_prefixes = ("raw_total", "pair_exp", "types_exp")
        for title, fig in figures:
            if fig is None:
                continue

            if "||" in title:
                _, suffix = title.split("||", 1)
            else:
                suffix = "raw_total"
            suffix_l = suffix.lower()

            should_export = False
            if suffix_l.startswith(scatter_prefixes):
                should_export = include_section("scatter", True)
            elif suffix_l == "heatmap":
                should_export = include_section("heatmap", True)
            elif suffix_l == "shepard":
                should_export = include_section("shepard", True)
            else:
                should_export = include_section("scatter", True)

            if not should_export:
                continue

            base_slug = self._slugify_filename(title)
            seen[base_slug] = seen.get(base_slug, 0) + 1
            slug = base_slug if seen[base_slug] == 1 else f"{base_slug}_{seen[base_slug]}"
            html_path = plots_dir / f"{slug}.html"
            png_path = plots_dir / f"{slug}.png"

            fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
            plot_entry: Dict[str, Any] = {"title": title, "html": str(Path("plots") / html_path.name)}

            try:
                fig.write_image(str(png_path))
                plot_entry["png"] = str(Path("plots") / png_path.name)
            except Exception:
                skipped_png += 1

            manifest["plots"].append(plot_entry)
            exported += 1

        manifest_path = output_dir / "plots_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        if logger:
            logger(f"[visualizacion] modo separado: {exported} plots exportados en {plots_dir}")
            logger(f"[visualizacion] tablas exportadas en {tables_dir}")
            if skipped_png:
                logger(
                    "[visualizacion] algunas imagenes PNG no se exportaron "
                    "(falta dependencia de exportacion estatica); se mantuvieron los HTML."
                )
            logger(f"[visualizacion] manifiesto: {manifest_path}")

        return manifest_path

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

        if config.skip_html_report:
            if logger:
                logger("[visualizacion] opcion activa: sin unificar reporte HTML.")
            return self._export_split_artifacts(
                result,
                figures,
                sections_enabled=sections_enabled,
                logger=logger,
            )

        # 4. Prepare Heatmap Data for JS (if eligible)
        heatmap_data: Optional[Dict[str, Any]] = None

        def _slugify(name: str) -> str:
            slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
            return slug or "scenario"

        if sections_enabled.get("heatmap", False) and result.output_path is not None:
            if logger:
                logger(f"[visualizacion] generando payload de heatmaps dinámicos (N={len(result.entries)})...")

            labels = _build_heatmap_axis_labels(result.entries)
            compact_labels = _generate_compact_labels(result.entries)
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
                        "axis_labels": labels,
                        "labels": labels,
                        "compact_labels": compact_labels,
                        "cardinalities": cardinalities,
                    },
                    "files": heatmap_files,
                }

        # 5. Build Audio Data (if enabled)
        audio_data: Optional[Dict[str, Any]] = None
        if config.audio.enabled:
            if logger:
                logger(f"[visualizacion] generando datos de audio (N={len(result.entries)})...")
            audio_data = build_audio_descriptors(result)

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
        try:
            render_report_html(
                metrics_df=result.metrics_df,
                figures=figures,
                output_path=result.output_path / "report.html",
                seeds=result.config.execution.seeds,
                run_metadata=run_metadata,
                metric_info=METRIC_INFO,
                highlight_threshold=config.highlight_threshold,
                sections_enabled=sections_enabled,
                heatmap_data=heatmap_data,  # Pass the extra payload
                audio_data=audio_data,      # Pass the audio payload
                instructional_mode=config.instructional_mode
            )
        except MemoryError:
            if logger:
                logger(
                    "[visualizacion] memoria insuficiente para reporte completo; "
                    "reintentando sin heatmap/shepard."
                )
            try:
                fallback_sections = {
                    "scatter": bool(sections_enabled.get("scatter", True)),
                    "heatmap": False,
                    "shepard": False,
                    "table": bool(sections_enabled.get("table", True)),
                    "secondary_metrics": bool(sections_enabled.get("secondary_metrics", False)),
                    "metadata": bool(sections_enabled.get("metadata", True)),
                }
                render_report_html(
                    metrics_df=result.metrics_df,
                    figures=figures,
                    output_path=result.output_path / "report.html",
                    seeds=result.config.execution.seeds,
                    run_metadata=run_metadata,
                    metric_info=METRIC_INFO,
                    highlight_threshold=config.highlight_threshold,
                    sections_enabled=fallback_sections,
                    heatmap_data=None,
                    audio_data=audio_data,
                )
                if logger:
                    logger(
                        "[visualizacion] reporte parcial generado "
                        "(scatter activo, heatmap/shepard desactivados)."
                    )
            except MemoryError:
                if logger:
                    logger(
                        "[visualizacion] memoria insuficiente incluso en modo parcial; "
                        "generando version liviana (sin figuras)."
                    )
                fallback_sections = {
                    "scatter": False,
                    "heatmap": False,
                    "shepard": False,
                    "table": bool(sections_enabled.get("table", True)),
                    "secondary_metrics": bool(sections_enabled.get("secondary_metrics", False)),
                    "metadata": bool(sections_enabled.get("metadata", True)),
                }
                render_report_html(
                    metrics_df=result.metrics_df,
                    figures=[],
                    output_path=result.output_path / "report.html",
                    seeds=result.config.execution.seeds,
                    run_metadata=run_metadata,
                    metric_info=METRIC_INFO,
                    highlight_threshold=config.highlight_threshold,
                    sections_enabled=fallback_sections,
                    heatmap_data=None,
                    audio_data=None,
                )
                if logger:
                    logger("[visualizacion] reporte liviano generado correctamente.")

        return result.output_path / "report.html"
