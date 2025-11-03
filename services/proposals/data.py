"""Data loading helpers for proposal comparison services."""

from __future__ import annotations

import ast
import numbers
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import config_db
from pre_process import Acorde, ChordAdapter, ModeloSetharesVec, get_chord_type_from_intervals
from tools.query_registry import resolve_query_sql

try:  # pragma: no cover - optional dependency
    from chordcodex.model import QueryExecutor  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for local envs
    from synth_tools import QueryExecutor  # type: ignore


@dataclass(frozen=True)
class ChordEntry:
    """Materialised representation of a chord in the comparison population."""

    acorde: object
    hist: np.ndarray
    total: float
    counts: np.ndarray
    total_pairs: float
    n_notes: int
    dyad_bin: Optional[int]
    identity_name: str
    identity_aliases: Tuple[str, ...]
    is_named: bool
    is_inversion: bool = False
    family_id: Optional[object] = None
    inversion_rotation: Optional[int] = None


class PopulationLoader:
    """Factory for :class:`ChordEntry` objects from SQL or dataframes."""

    def __init__(self, executor: Optional[QueryExecutor] = None) -> None:
        self._executor = executor
        self._modelo = ModeloSetharesVec(config={})
        self._roughness_cache: Dict[
            Tuple[Tuple[int, ...], Optional[Tuple[Any, ...]]], Tuple[np.ndarray, float]
        ] = {}
        self._interval_cache: Dict[
            Tuple[int, ...], Tuple[np.ndarray, float, int, Optional[int]]
        ] = {}

    @property
    def executor(self) -> QueryExecutor:
        if self._executor is None:
            self._executor = QueryExecutor(**config_db)
        return self._executor

    @staticmethod
    def _safe_literal_eval(value: Any, fallback_factory: Callable[[], Any]) -> Any:
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return fallback_factory()
        if value is None:
            return fallback_factory()
        try:
            if pd.isna(value):
                return fallback_factory()
        except (TypeError, ValueError):
            pass
        return value

    @staticmethod
    def _prepare_literal_column(
        dataframe: pd.DataFrame,
        column: str,
        fallback_factory: Callable[[], Any],
    ) -> List[Any]:
        if column not in dataframe.columns:
            return [fallback_factory() for _ in range(len(dataframe))]
        series = dataframe[column]
        return [
            PopulationLoader._safe_literal_eval(value, fallback_factory)
            for value in series
        ]

    @staticmethod
    def _prepare_scalar_column(
        dataframe: pd.DataFrame, column: str, default: Any
    ) -> List[Any]:
        if column not in dataframe.columns:
            return [default for _ in range(len(dataframe))]
        values: List[Any] = []
        series = dataframe[column]
        for value in series:
            if value is None:
                values.append(default)
                continue
            try:
                if pd.isna(value):
                    values.append(default)
                    continue
            except (TypeError, ValueError):
                pass
            values.append(value)
        return values

    @staticmethod
    def _flatten_iterable(value: Any) -> Iterable[Any]:
        if isinstance(value, np.ndarray):
            for item in value.flat:
                yield item
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from PopulationLoader._flatten_iterable(item)
        else:
            yield value

    def _roughness_from_cache(self, acorde: "Acorde") -> Tuple[np.ndarray, float]:
        interval_key = tuple(int(i) for i in acorde.intervals)
        freq_key: Optional[Tuple[Any, ...]] = None
        if acorde.frequencies is not None:
            freq_key = tuple(
                float(item) if isinstance(item, numbers.Number) else item
                for item in self._flatten_iterable(acorde.frequencies)
            )
        cache_key = (interval_key, freq_key)
        cached = self._roughness_cache.get(cache_key)
        if cached is not None:
            hist, total = cached
            return hist.copy(), total
        hist, total = self._modelo.calcular(acorde)
        hist = np.asarray(hist, dtype=float)
        result = (hist, float(total))
        self._roughness_cache[cache_key] = result
        return hist.copy(), float(total)

    def _interval_metadata(
        self, intervals: Sequence[int]
    ) -> Tuple[np.ndarray, float, int, Optional[int]]:
        key = tuple(int(i) for i in intervals)
        cached = self._interval_cache.get(key)
        if cached is not None:
            counts, total_pairs, n_notes, dyad_bin = cached
            return counts.copy(), total_pairs, n_notes, dyad_bin
        counts = compute_interval_counts(intervals)
        total_pairs = float(np.sum(counts))
        n_notes = len(intervals) + 1
        dyad_bin = determine_dyad_bin(intervals) if n_notes == 2 else None
        self._interval_cache[key] = (counts, total_pairs, n_notes, dyad_bin)
        return counts.copy(), total_pairs, n_notes, dyad_bin

    def from_queries(
        self,
        dyads_query: str,
        triads_query: str,
        sevenths_query: Optional[str] = None,
    ) -> List[ChordEntry]:
        frames: List[pd.DataFrame] = []
        for query in (dyads_query, triads_query, sevenths_query):
            if not query:
                continue
            sql = resolve_query_sql(query) if query.upper().startswith("QUERY_") else query
            frames.append(self.executor.as_pandas(sql))
        if not frames:
            raise ValueError("No se proporcionaron consultas válidas ni población precombinada.")
        df_all = pd.concat(frames, ignore_index=True)
        return self.from_dataframe(df_all)

    def from_dataframe(self, dataframe: pd.DataFrame) -> List[ChordEntry]:
        has_family = "__family_id" in dataframe.columns
        has_inv_flag = "__inv_flag" in dataframe.columns
        has_inv_source = "__inv_source_id" in dataframe.columns
        has_inv_rotation = "__inv_rotation" in dataframe.columns

        dataframe = dataframe.copy()
        if dataframe.empty:
            return []

        intervals_col = self._prepare_literal_column(
            dataframe, "interval", lambda: []
        )
        chroma_col = self._prepare_literal_column(
            dataframe, "chroma", lambda: [0] * 12
        )
        frequencies_col = self._prepare_literal_column(
            dataframe, "frequencies", lambda: None
        )
        notes_col = self._prepare_literal_column(dataframe, "notes", lambda: None)
        codes = self._prepare_scalar_column(dataframe, "code", "Sin nombre")
        inv_flags = self._prepare_scalar_column(dataframe, "__inv_flag", False)
        family_ids = self._prepare_scalar_column(dataframe, "__family_id", None)
        inv_sources = self._prepare_scalar_column(dataframe, "__inv_source_id", None)
        inv_rotations = self._prepare_scalar_column(dataframe, "__inv_rotation", None)
        ids = self._prepare_scalar_column(dataframe, "id", None)

        entries: List[ChordEntry] = []
        for idx in range(len(dataframe)):
            row_payload = {
                "code": codes[idx],
                "interval": intervals_col[idx],
                "chroma": chroma_col[idx],
                "frequencies": frequencies_col[idx],
                "notes": notes_col[idx],
            }
            acorde = ChordAdapter.from_csv_row(row_payload)
            identity_obj = get_chord_type_from_intervals(acorde.intervals, with_alias=True)
            identity_name = getattr(identity_obj, "name", str(identity_obj))
            identity_aliases = tuple(getattr(identity_obj, "aliases", ()))
            is_named = bool(identity_name and identity_name != "Unknown")
            hist, total = self._roughness_from_cache(acorde)
            counts, total_pairs, n_notes, dyad_bin = self._interval_metadata(
                acorde.intervals
            )
            inv_flag = bool(inv_flags[idx]) if has_inv_flag else False

            family_id: Optional[object] = None
            if has_family:
                raw_family = family_ids[idx]
                if pd.notna(raw_family):
                    try:
                        family_id = int(raw_family)
                    except (TypeError, ValueError):
                        family_id = str(raw_family)
            if family_id is None and has_inv_source:
                raw_family = inv_sources[idx]
                if pd.notna(raw_family):
                    try:
                        family_id = int(raw_family)
                    except (TypeError, ValueError):
                        family_id = str(raw_family)
            if family_id is None:
                raw_id = ids[idx]
                if pd.notna(raw_id):
                    try:
                        family_id = int(raw_id)
                    except (TypeError, ValueError):
                        family_id = str(raw_id)

            inv_rotation: Optional[int] = None
            if has_inv_rotation:
                raw_rot = inv_rotations[idx]
                if pd.notna(raw_rot):
                    try:
                        inv_rotation = int(raw_rot)
                    except (TypeError, ValueError):
                        inv_rotation = None

            entries.append(
                ChordEntry(
                    acorde=acorde,
                    hist=hist,
                    total=float(total),
                    counts=counts,
                    total_pairs=total_pairs if total_pairs > 0 else 1.0,
                    n_notes=n_notes,
                    dyad_bin=dyad_bin,
                    identity_name=identity_name,
                    identity_aliases=identity_aliases,
                    is_named=is_named,
                    is_inversion=inv_flag,
                    family_id=family_id,
                    inversion_rotation=inv_rotation,
                )
            )
        return entries


def compute_interval_counts(intervals: Sequence[int]) -> np.ndarray:
    """Count pairs per interval class using UI bin order."""

    semitonos = [0]
    for step in intervals:
        semitonos.append((semitonos[-1] + int(step)) % 12)
    counts = np.zeros(12, dtype=float)
    for i in range(len(semitonos) - 1):
        for j in range(i + 1, len(semitonos)):
            intervalo = (semitonos[j] - semitonos[i]) % 12
            bin_idx = (intervalo - 1) % 12
            counts[bin_idx] += 1.0
    return counts


def determine_dyad_bin(intervals: Sequence[int]) -> Optional[int]:
    if not intervals:
        return None
    intervalo = int(intervals[0]) % 12
    return (intervalo - 1) % 12


def stack_hist(entries: Iterable[ChordEntry]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    entries = list(entries)
    hist = np.stack([e.hist for e in entries], axis=0)
    totals = np.array([e.total for e in entries], dtype=float)
    counts = np.stack([e.counts for e in entries], axis=0)
    pairs = np.array([e.total_pairs for e in entries], dtype=float)
    notes = np.array([float(e.n_notes) for e in entries], dtype=float)
    return hist, totals, counts, pairs, notes


__all__ = [
    "ChordEntry",
    "PopulationLoader",
    "compute_interval_counts",
    "determine_dyad_bin",
    "stack_hist",
]
