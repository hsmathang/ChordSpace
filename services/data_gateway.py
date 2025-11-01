"""Experiment data access layer for CLI/GUI tools.

This module centralises how experiment runners access chord populations.
It provides a registry-driven gateway interface that abstracts the
underlying data source (database, CSV exports, etc.), ensures consistent
SQL resolution, handles deduplication policies and exposes the shared
chord template catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple

import pandas as pd

from config import (
    CHORD_TEMPLATES_METADATA,
    QUERY_DYADS_REFERENCE,
    QUERY_TRIADS_CORE,
    config_db,
)
from tools.population_utils import dedupe_population
from tools.query_registry import get_all_queries, resolve_query_sql

try:  # pragma: no cover - prefer packaged executor when available
    from chordcodex.model import QueryExecutor  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from synth_tools import QueryExecutor  # type: ignore


# Keep references to ensure the canonical config constants are loaded.
_ = QUERY_DYADS_REFERENCE, QUERY_TRIADS_CORE

DEFAULT_DYADS_QUERY = "QUERY_DYADS_REFERENCE"
DEFAULT_TRIADS_QUERY = "QUERY_TRIADS_CORE"
DEFAULT_GATEWAY_NAME = "database"


@dataclass
class PopulationResult:
    """Container with population data returned by a gateway."""

    dataframe: pd.DataFrame
    dedupe_key: Optional[str] = None
    sources: Tuple[str, ...] = ()
    stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataframe = self.dataframe.reset_index(drop=True)


class ExperimentDataGateway(Protocol):
    """Protocol implemented by experiment data gateways."""

    def resolve_sql(self, query_or_alias: str) -> str:
        """Resolve a query reference to SQL using the shared registry."""

    def fetch_population(self, sources: Sequence[str], *, dedupe: bool = True) -> PopulationResult:
        """Fetch and combine populations identified by ``sources``."""

    def ingest_population(
        self,
        frame: pd.DataFrame,
        *,
        dedupe: bool = True,
        source: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PopulationResult:
        """Ingest an already materialised population frame."""

    def get_templates(self) -> Sequence[Mapping[str, Any]]:
        """Return chord template metadata for consumers that need it."""

    def available_queries(self) -> Mapping[str, Dict[str, str]]:
        """Expose the discoverable SQL registry (name -> {sql, source})."""


class BaseExperimentDataGateway(ExperimentDataGateway):
    """Base implementation with shared helpers for concrete gateways."""

    def resolve_sql(self, query_or_alias: str) -> str:
        return resolve_query_sql(query_or_alias)

    def available_queries(self) -> Mapping[str, Dict[str, str]]:
        return get_all_queries()

    def get_templates(self) -> Sequence[Mapping[str, Any]]:
        # Return shallow copies to avoid accidental mutation of globals.
        return tuple(dict(template) for template in CHORD_TEMPLATES_METADATA)

    def ingest_population(
        self,
        frame: pd.DataFrame,
        *,
        dedupe: bool = True,
        source: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PopulationResult:
        df = frame.copy()
        stats: Dict[str, Any] = dict(metadata or {})

        if source and "__source__" not in df.columns:
            df = df.copy()
            df["__source__"] = source

        raw_count = len(df)
        stats.setdefault("raw_count", raw_count)

        dedupe_key: Optional[str] = None
        if dedupe:
            df, dedupe_key = dedupe_population(df)
            stats["removed"] = raw_count - len(df)
        else:
            df = df.copy()
            stats["removed"] = 0

        final_count = len(df)
        stats["final_count"] = final_count

        if "__source__" in df.columns:
            source_series = df["__source__"].astype(str).fillna("")
            stats["source_counts_after"] = source_series.value_counts().to_dict()
            sources = tuple(source_series.unique())
        else:
            sources = tuple()

        return PopulationResult(df, dedupe_key, sources, stats)

    def fetch_population(self, sources: Sequence[str], *, dedupe: bool = True) -> PopulationResult:  # pragma: no cover - abstract
        raise NotImplementedError


class DatabaseQueryGateway(BaseExperimentDataGateway):
    """Gateway that loads chord populations from the configured database."""

    def __init__(
        self,
        *,
        db_config: Optional[Mapping[str, Any]] = None,
        executor: Optional[QueryExecutor] = None,
    ) -> None:
        self._db_config: MutableMapping[str, Any] = dict(db_config or config_db)
        self._executor: Optional[QueryExecutor] = executor

    @property
    def executor(self) -> QueryExecutor:
        if self._executor is None:
            self._executor = QueryExecutor(**self._db_config)
        return self._executor

    def fetch_population(self, sources: Sequence[str], *, dedupe: bool = True) -> PopulationResult:
        frames = []
        counts_before: Dict[str, int] = {}
        used_sources: Tuple[str, ...] = tuple()
        ordered_sources = []

        for reference in sources:
            if not reference:
                continue
            sql = self.resolve_sql(reference)
            df = self.executor.as_pandas(sql)
            counts_before[reference] = len(df)
            df = df.copy()
            if "__source__" not in df.columns:
                df["__source__"] = reference
            else:
                df["__source__"] = df["__source__"].fillna(reference)
            frames.append(df)
            ordered_sources.append(reference)

        if not frames:
            raise ValueError("No se proporcionaron consultas válidas para construir la población.")

        combined = pd.concat(frames, ignore_index=True)
        metadata = {
            "source_counts_before": counts_before,
            "query_order": tuple(ordered_sources),
        }
        result = self.ingest_population(combined, dedupe=dedupe, metadata=metadata)
        return result


class CSVPopulationGateway(BaseExperimentDataGateway):
    """Gateway that reads pre-exported populations from CSV/JSON files."""

    def __init__(
        self,
        *,
        default_path: Optional[str] = None,
        file_format: Optional[str] = None,
    ) -> None:
        self._default_path = Path(default_path).expanduser() if default_path else None
        self._explicit_format = file_format.lower() if file_format else None

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute() and self._default_path is not None:
            return (self._default_path.parent / path).resolve()
        return path

    def _detect_format(self, path: Path) -> str:
        if self._explicit_format:
            return self._explicit_format
        suffix = path.suffix.lower().lstrip(".")
        return suffix or "csv"

    def _load_file(self, path: Path) -> pd.DataFrame:
        fmt = self._detect_format(path)
        if fmt in {"json", "ndjson", "jsonl"}:
            return pd.read_json(path, orient="records", lines=True)
        if fmt in {"parquet"}:
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def fetch_population(self, sources: Sequence[str], *, dedupe: bool = True) -> PopulationResult:
        paths = [self._resolve_path(src) for src in sources if src]
        if not paths and self._default_path is not None:
            paths = [self._default_path]

        if not paths:
            raise ValueError("No se proporcionaron archivos de población para el gateway CSV.")

        frames = []
        counts_before: Dict[str, int] = {}
        ordered_paths = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"No existe el archivo de población: {path}")
            df = self._load_file(path)
            counts_before[str(path)] = len(df)
            df = df.copy()
            if "__source__" not in df.columns:
                df["__source__"] = str(path)
            frames.append(df)
            ordered_paths.append(str(path))

        combined = pd.concat(frames, ignore_index=True)
        metadata = {
            "source_counts_before": counts_before,
            "file_paths": tuple(ordered_paths),
        }
        return self.ingest_population(combined, dedupe=dedupe, metadata=metadata)


GatewayFactory = Callable[..., ExperimentDataGateway]
_GATEWAY_REGISTRY: Dict[str, GatewayFactory] = {}


def register_data_gateway(name: str, factory: GatewayFactory) -> None:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("El nombre del gateway no puede estar vacío.")
    _GATEWAY_REGISTRY[normalized] = factory


def get_registered_gateways() -> Tuple[str, ...]:
    return tuple(sorted(_GATEWAY_REGISTRY.keys()))


def create_data_gateway(name: str, **kwargs: Any) -> ExperimentDataGateway:
    normalized = name.strip().lower()
    if normalized not in _GATEWAY_REGISTRY:
        available = ", ".join(sorted(_GATEWAY_REGISTRY.keys()))
        raise KeyError(f"Gateway desconocido '{name}'. Disponibles: {available}")
    factory = _GATEWAY_REGISTRY[normalized]
    return factory(**kwargs)


register_data_gateway("database", DatabaseQueryGateway)
register_data_gateway("csv", CSVPopulationGateway)

__all__ = [
    "DEFAULT_DYADS_QUERY",
    "DEFAULT_TRIADS_QUERY",
    "DEFAULT_GATEWAY_NAME",
    "PopulationResult",
    "ExperimentDataGateway",
    "DatabaseQueryGateway",
    "CSVPopulationGateway",
    "register_data_gateway",
    "get_registered_gateways",
    "create_data_gateway",
]
