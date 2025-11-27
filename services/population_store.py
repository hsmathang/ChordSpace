"""
Population storage and access layer.
Handles loading/saving chord populations from various sources (Files, DB).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import pandas as pd
import numpy as np

from services.domain import PopulationConfig
from tools.population_utils import dedupe_population
from config import config_db

# Try to import QueryExecutor, but handle if it's not available (file-only mode)
try:
    from chordcodex.model import QueryExecutor
except ModuleNotFoundError:
    try:
        from synth_tools import QueryExecutor
    except ImportError:
        QueryExecutor = None

@dataclass
class PopulationData:
    """Container for population dataframe and metadata."""
    dataframe: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.dataframe = self.dataframe.reset_index(drop=True)

class PopulationStore(Protocol):
    """Interface for population storage/retrieval."""

    def load(self, config: PopulationConfig) -> PopulationData:
        """Loads a population based on the configuration."""
        ...

    def save(self, data: PopulationData, path: Union[str, Path]) -> None:
        """Saves a population to disk."""
        ...

class FilePopulationStore:
    """File-based implementation of PopulationStore (JSONL/CSV)."""

    def load(self, config: PopulationConfig) -> PopulationData:
        if not config.file_path:
            raise ValueError("File path required for FilePopulationStore")

        path = Path(config.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Population file not found: {path}")

        df = self._read_file(path)

        # Merge config metadata with any metadata found in the file?
        # For now, just use what's in config
        return PopulationData(
            dataframe=df,
            metadata=config.metadata.copy()
        )

    def _read_file(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in {'.json', '.jsonl', '.ndjson'}:
            return pd.read_json(path, orient='records', lines=True)
        elif suffix == '.csv':
            return pd.read_csv(path)
        elif suffix == '.parquet':
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def save(self, data: PopulationData, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save dataframe
        if path.suffix.lower() in {'.json', '.jsonl'}:
            data.dataframe.to_json(path, orient='records', lines=True)
        elif path.suffix.lower() == '.csv':
            data.dataframe.to_csv(path, index=False)
        else:
            # Default to jsonl if unknown or directory
            if path.suffix == '':
                path = path / "population.jsonl"
            data.dataframe.to_json(path, orient='records', lines=True)

class DatabasePopulationStore:
    """Database-backed implementation (Postgres)."""

    def __init__(self, connection_config: Dict[str, Any] = config_db):
        self.config_db = connection_config
        self._executor = None

    @property
    def executor(self):
        if self._executor is None:
            if QueryExecutor is None:
                raise ImportError("Database support dependencies not installed (chordcodex/psycopg2)")
            self._executor = QueryExecutor(**self.config_db)
        return self._executor

    def load(self, config: PopulationConfig) -> PopulationData:
        if QueryExecutor is None:
             raise RuntimeError("Database support is not available.")

        frames = []
        sources = []

        # Resolve queries from config
        # We handle dyads, triads, sevenths as specific query fields
        # But we could also have a generic list of queries in metadata if needed
        queries = [
            ("dyads", config.dyads_query),
            ("triads", config.triads_query),
            ("sevenths", config.sevenths_query)
        ]

        for label, query in queries:
            if not query:
                continue

            # Simple SQL resolution if it starts with explicit SQL or just a table name
            # Real resolution logic from data_gateway/query_registry could be imported here
            # For now, assume it's valid SQL or a key handled by existing tools if we integrate them.
            # But the requirement says "clean architecture". Let's assume raw SQL or keys.
            # If we want to use the registry, we should import `resolve_query_sql` from tools.

            try:
                from tools.query_registry import resolve_query_sql
                if query.upper().startswith("QUERY_") or not " " in query:
                     sql = resolve_query_sql(query)
                else:
                     sql = query
            except ImportError:
                 sql = query

            df = self.executor.as_pandas(sql)
            if "__source__" not in df.columns:
                df["__source__"] = f"DB:{label}"
            frames.append(df)
            sources.append(label)

        if not frames:
            raise ValueError("No queries provided for DatabasePopulationStore")

        combined = pd.concat(frames, ignore_index=True)

        # Dedupe logic similar to current pipeline
        combined, _ = dedupe_population(combined)

        return PopulationData(
            dataframe=combined,
            metadata={
                **config.metadata,
                "db_sources": sources
            }
        )

    def save(self, data: PopulationData, path: Union[str, Path]) -> None:
        raise NotImplementedError("Saving to DB not implemented directly. Use FileStore for artifacts.")

def get_population_store(config: PopulationConfig) -> PopulationStore:
    """Factory to get the appropriate store based on config."""
    if config.source_type == 'db':
        return DatabasePopulationStore()
    else:
        return FilePopulationStore()
