"""Data loading helpers for proposal comparison services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import config_db
from pre_process import ChordAdapter, ModeloSetharesVec, get_chord_type_from_intervals
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

    @property
    def executor(self) -> QueryExecutor:
        if self._executor is None:
            self._executor = QueryExecutor(**config_db)
        return self._executor

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

        entries: List[ChordEntry] = []
        for _, row in dataframe.iterrows():
            acorde = ChordAdapter.from_csv_row(row)
            identity_obj = get_chord_type_from_intervals(acorde.intervals, with_alias=True)
            identity_name = getattr(identity_obj, "name", str(identity_obj))
            identity_aliases = tuple(getattr(identity_obj, "aliases", ()))
            is_named = bool(identity_name and identity_name != "Unknown")
            hist, total = self._modelo.calcular(acorde)
            hist = np.asarray(hist, dtype=float)
            counts = compute_interval_counts(acorde.intervals)
            total_pairs = float(np.sum(counts))
            n_notes = len(acorde.intervals) + 1
            dyad_bin = determine_dyad_bin(acorde.intervals) if n_notes == 2 else None
            inv_flag = bool(row.get("__inv_flag")) if has_inv_flag else False

            family_id: Optional[object] = None
            if has_family:
                raw_family = row.get("__family_id")
                if pd.notna(raw_family):
                    try:
                        family_id = int(raw_family)
                    except (TypeError, ValueError):
                        family_id = str(raw_family)
            if family_id is None and has_inv_source:
                raw_family = row.get("__inv_source_id")
                if pd.notna(raw_family):
                    try:
                        family_id = int(raw_family)
                    except (TypeError, ValueError):
                        family_id = str(raw_family)
            if family_id is None:
                raw_id = row.get("id")
                if pd.notna(raw_id):
                    try:
                        family_id = int(raw_id)
                    except (TypeError, ValueError):
                        family_id = str(raw_id)

            inv_rotation: Optional[int] = None
            if has_inv_rotation:
                raw_rot = row.get("__inv_rotation")
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
