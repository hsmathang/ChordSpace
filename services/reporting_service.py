"""Servicio unificado para generar reportes HTML a partir de figuras Plotly.

Pensado para reemplazar los reportes "ligeros" dispersos y evitar
duplicación de plantillas. Usa los assets comunes en tools/report_assets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
from plotly.io import to_html


ASSETS_DIR = Path(__file__).resolve().parent.parent / "tools" / "report_assets"


def _load_asset(name: str) -> str:
    path = ASSETS_DIR / name
    return path.read_text(encoding="utf-8")


try:
    _TEMPLATE = _load_asset("template.html")
    _CSS = _load_asset("styles.css")
    _JS = _load_asset("script.js")
except FileNotFoundError:  # pragma: no cover
    _TEMPLATE = ""
    _CSS = ""
    _JS = ""


def render_figures_report(
    *,
    title: str,
    figures: Sequence[tuple[str, "go.Figure"]],
    output_path: Path,
    sections_enabled: Mapping[str, bool] | None = None,
    metrics: pd.DataFrame | None = None,
    metadata_rows: Iterable[tuple[str, str]] | None = None,
) -> None:
    """Renderiza un único report.html con las figuras solicitadas.

    Args:
        title: Título principal del reporte.
        figures: Lista (titulo, figura).
        output_path: Ruta de salida del HTML.
        sections_enabled: llaves para incluir/excluir secciones (scatter/heatmap/shepard/table/meta).
        metrics: dataframe opcional con métricas para tabla.
        metadata_rows: pares (label, valor) para tabla de metadatos.
    """
    sections = sections_enabled or {}
    include = lambda key, default=True: bool(sections.get(key, default))

    include_js = True
    html_sections = []
    for fig_title, fig in figures:
        key = fig_title.lower()
        if key.startswith("scatter") and not include("scatter", True):
            continue
        if key.startswith("heatmap") and not include("heatmap", True):
            continue
        if key.startswith("shepard") and not include("shepard", True):
            continue
        snippet = to_html(fig, include_plotlyjs="cdn" if include_js else False, full_html=False)
        include_js = False
        html_sections.append(f"<section><h2>{fig_title}</h2>{snippet}</section>")

    table_html = ""
    if include("table", True) and metrics is not None and not metrics.empty:
        table_html = metrics.to_html(index=False, float_format=lambda x: f"{x:.4f}")

    meta_html = ""
    rows = list(metadata_rows or [])
    if include("metadata", True) and rows:
        body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
        meta_html = "<table class='meta-table'><tbody>" + body + "</tbody></table>"

    if _TEMPLATE:
        html_content = (
            _TEMPLATE.replace("__CSS__", _CSS)
            .replace("__TABLE_HTML__", table_html)
            .replace("__METADATA_HTML__", meta_html)
            .replace("__TABS_HTML__", "".join(html_sections))
            .replace("__SCRIPT_JS__", _JS)
            .replace("<title>Comparacion de Propuestas de Rugosidad</title>", f"<title>{title}</title>")
        )
    else:  # pragma: no cover
        html_content = (
            "<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'/>"
            f"<title>{title}</title></head><body><h1>{title}</h1>"
            + table_html
            + meta_html
            + "".join(html_sections)
            + "</body></html>"
        )

    output_path.write_text(html_content, encoding="utf-8")


__all__ = ["render_figures_report"]
