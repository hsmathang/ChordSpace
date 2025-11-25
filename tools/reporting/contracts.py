"""Especificaciones ligeras y validadores para los metadatos del reporte/figuras.

Se usan para documentar y asegurar los contratos entre Python y el JS
(`tools/report_assets/script.js`), sin agregar dependencias pesadas.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, TypedDict


class FilterDataset(TypedDict, total=False):
    traceSources: Sequence[Mapping[str, Any]]
    fields: Mapping[str, Any]


class SubstitutionProfiles(TypedDict, total=False):
    default: str
    profiles: Mapping[str, Mapping[str, str]]


class FigureMeta(TypedDict, total=False):
    colorTitle: str
    isProposal: bool
    filterDataset: FilterDataset
    substitutionNeighbors: Mapping[str, Any]
    substitutionProfiles: SubstitutionProfiles
    familyHighlight: Mapping[str, Any]


REQUIRED_META_KEYS = {"colorTitle", "isProposal", "filterDataset"}


def validate_scatter_payload_meta(meta: Mapping[str, Any]) -> None:
    """Valida la forma mínima de los metadatos que consumen los scripts del reporte.

    Se busca atrapar cambios de contrato (renombres o ausencias) de manera temprana
    en tests. No se intenta validar contenido semántico, solo presencia y tipos básicos.
    """

    missing = [key for key in REQUIRED_META_KEYS if key not in meta]
    if missing:
        raise ValueError(f"Meta de figura incompleta: faltan {missing}")

    # filterDataset debe contener traceSources y fields
    fd = meta.get("filterDataset")
    if not isinstance(fd, Mapping):
        raise ValueError("filterDataset debe ser un mapeo")
    if "traceSources" not in fd or "fields" not in fd:
        raise ValueError("filterDataset requiere traceSources y fields")

    # Si existen perfiles de sustitución, validar que haya perfil default
    subs_profiles = meta.get("substitutionProfiles")
    if subs_profiles is not None:
        if not isinstance(subs_profiles, Mapping):
            raise ValueError("substitutionProfiles debe ser un mapeo si está presente")
        if "default" not in subs_profiles:
            raise ValueError("substitutionProfiles debe declarar una clave 'default'")


def validate_run_metadata(meta: Mapping[str, Any]) -> None:
    """Valida la forma mínima del objeto run_metadata que se muestra en el reporte."""

    if not isinstance(meta, Mapping):
        raise ValueError("run_metadata debe ser un mapeo")
    selection = meta.get("selection", {})
    if not isinstance(selection, Mapping):
        raise ValueError("run_metadata.selection debe ser un mapeo")
    # Claves mínimas: filas seleccionadas/disponibles si existen
    for key in ("rows_selected", "rows_available"):
        if key in selection and selection[key] is not None:
            try:
                int(selection[key])
            except Exception as exc:
                raise ValueError(f"run_metadata.selection.{key} debe ser entero") from exc

    population = meta.get("population", {})
    if not isinstance(population, Mapping):
        raise ValueError("run_metadata.population debe ser un mapeo")
    descriptors = population.get("descriptors", [])
    if descriptors is None:
        descriptors = []
    if not isinstance(descriptors, (list, tuple)):
        raise ValueError("run_metadata.population.descriptors debe ser lista")
    for desc in descriptors:
        if not isinstance(desc, Mapping):
            raise ValueError("Cada descriptor de población debe ser un mapeo")
        if "mode" not in desc:
            raise ValueError("Cada descriptor de población requiere la clave 'mode'")
        mode_val = desc.get("mode")
        if mode_val is not None and not isinstance(mode_val, str):
            raise ValueError("run_metadata.population.descriptors.mode debe ser string")
        rows_val = desc.get("rows")
        if rows_val is not None:
            try:
                int(rows_val)
            except Exception as exc:
                raise ValueError("run_metadata.population.descriptors.rows debe ser entero") from exc
        if "combinatorial" in desc and desc["combinatorial"] is not None and not isinstance(desc["combinatorial"], Mapping):
            raise ValueError("run_metadata.population.descriptors.combinatorial debe ser un mapeo")
        if "database" in desc and desc["database"] is not None and not isinstance(desc["database"], Mapping):
            raise ValueError("run_metadata.population.descriptors.database debe ser un mapeo")


__all__ = [
    "FilterDataset",
    "SubstitutionProfiles",
    "FigureMeta",
    "validate_scatter_payload_meta",
    "validate_run_metadata",
]
