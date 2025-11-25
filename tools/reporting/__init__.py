"""Paquete con helpers para la construcción del reporte HTML y sus contratos mínimos."""

from .report_builder import render_report_html
from .contracts import (
    FilterDataset,
    SubstitutionProfiles,
    FigureMeta,
    validate_scatter_payload_meta,
    validate_run_metadata,
)

__all__ = [
    "render_report_html",
    "FilterDataset",
    "SubstitutionProfiles",
    "FigureMeta",
    "validate_scatter_payload_meta",
    "validate_run_metadata",
]
