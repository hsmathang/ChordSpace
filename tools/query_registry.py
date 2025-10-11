"""
Utility helpers to manage SQL query definitions coming from config.py and user-defined entries.

This module centralizes the discovery of available queries and allows persisting new ones
without touching the main configuration file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import importlib

CONFIG_MODULE = "config"
CUSTOM_QUERIES_PATH = Path(__file__).resolve().parent / "custom_queries.json"


def _load_config_module():
    return importlib.import_module(CONFIG_MODULE)


def _load_custom_queries() -> Dict[str, str]:
    if not CUSTOM_QUERIES_PATH.exists():
        return {}
    try:
        data = json.loads(CUSTOM_QUERIES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Archivo corrupto: mejor ignorar y empezar limpio.
        return {}
    return {str(k).upper(): str(v) for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}


def _save_custom_queries(data: Dict[str, str]) -> None:
    CUSTOM_QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_QUERIES_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_all_queries() -> Dict[str, Dict[str, str]]:
    """
    Returns a dictionary mapping query names to {"sql": ..., "source": ...}.
    """
    cfg = _load_config_module()
    registry: Dict[str, Dict[str, str]] = {}
    for name in dir(cfg):
        if not name.startswith("QUERY_"):
            continue
        value = getattr(cfg, name)
        if isinstance(value, str):
            registry[name] = {"sql": value, "source": "config"}
    for name, sql in _load_custom_queries().items():
        if name not in registry:
            registry[name] = {"sql": sql, "source": "custom"}
        else:
            # Preferir mantener la versión de config si hay conflicto.
            registry[f"CUSTOM_{name}"] = {"sql": sql, "source": "custom"}
    return dict(sorted(registry.items()))


def resolve_query_sql(query_ref: str) -> str:
    """
    Resolves a query reference to its SQL string.

    Accepts:
      - Name of a constant defined in config.py (e.g. QUERY_CHORDS_WITH_NAME)
      - Name of a saved custom query
      - Raw SQL text (starts with SELECT/WITH/--//*)
    """
    if not query_ref:
        raise ValueError("No se proporcionó una consulta.")
    trimmed = query_ref.strip()
    upper = trimmed.upper()
    if upper.startswith(("SELECT", "WITH", "/*", "--")):
        return trimmed

    queries = get_all_queries()
    if upper in queries:
        return queries[upper]["sql"]
    # Permitir nombres ya normalizados (sin upper) por conveniencia.
    if trimmed in queries:
        return queries[trimmed]["sql"]

    raise KeyError(f"No se encontró una consulta llamada '{query_ref}'.")


def add_custom_query(name: str, sql: str) -> str:
    """
    Stores a new custom query and returns the normalized name.
    """
    if not name or not name.strip():
        raise ValueError("El nombre de la consulta no puede estar vacío.")
    if not sql or not sql.strip():
        raise ValueError("El SQL no puede estar vacío.")

    normalized = name.strip().upper()
    if not normalized.startswith("QUERY_"):
        normalized = f"QUERY_{normalized}"

    if not sql.lstrip().upper().startswith(("SELECT", "WITH")):
        raise ValueError("La consulta debe empezar con SELECT o WITH.")

    queries = get_all_queries()
    if normalized in queries and queries[normalized]["source"] == "config":
        raise ValueError(f"Ya existe una consulta llamada '{normalized}' en config.py.")

    custom = _load_custom_queries()
    custom[normalized] = sql.strip()
    _save_custom_queries(custom)
    return normalized


def available_query_names() -> Iterable[str]:
    return get_all_queries().keys()
