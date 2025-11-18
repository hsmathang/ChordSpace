"""Construcción de poblaciones y estructuras `ChordEntry`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import config_db
from pre_process import ChordAdapter, ModeloSetharesVec, get_chord_type_from_intervals
from tools.query_registry import resolve_query_sql

try:  # pragma: no cover - mismo shim que en compare_proposals
    from chordcodex.model import QueryExecutor  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from synth_tools import QueryExecutor  # type: ignore


@dataclass
class ChordEntry:
    """Representa un acorde con toda la metadata usada en el pipeline."""

    acorde: object  # pre_process.Acorde
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
    musical_inversion_ids: List[Any] = field(default_factory=list)
    structural_inversion_ids: List[Any] = field(default_factory=list)


def load_chords(
    dyads_query: str,
    triads_query: str,
    sevenths_query: Optional[str] = None,
    *,
    df_override: Optional[pd.DataFrame] = None,
) -> List[ChordEntry]:
    """Carga acordes desde SQL o un DataFrame externo."""

    if df_override is not None:
        df_all = df_override.copy()
    else:
        executor = QueryExecutor(**config_db)
        frames: List[pd.DataFrame] = []
        for query in (dyads_query, triads_query, sevenths_query):
            if not query:
                continue
            sql = resolve_query_sql(query) if query.upper().startswith("QUERY_") else query
            frames.append(executor.as_pandas(sql))
        if not frames:
            raise SystemExit("No se proporcionaron consultas válidas ni población precombinada.")
        df_all = pd.concat(frames, ignore_index=True)

    has_family = "__family_id" in df_all.columns
    has_family_size = "__family_size" in df_all.columns
    has_inv_flag = "__inv_flag" in df_all.columns
    has_inv_source = "__inv_source_id" in df_all.columns
    has_inv_rotation = "__inv_rotation" in df_all.columns

    modelo = ModeloSetharesVec(config={})
    entries: List[ChordEntry] = []

    for _, row in df_all.iterrows():
        acorde = ChordAdapter.from_csv_row(row)
        if "notes_abs_json" in row and isinstance(row["notes_abs_json"], str) and row["notes_abs_json"]:
            acorde.notes_abs = json.loads(row["notes_abs_json"])
        elif "notes_abs_json" in row and isinstance(row["notes_abs_json"], (list, tuple, np.ndarray)):
            try:
                acorde.notes_abs = [int(n) for n in list(row["notes_abs_json"])]
            except Exception:  # pragma: no cover - defensivo
                acorde.notes_abs = None
        else:
            base_freq = 440.0
            semitonos_rel = np.cumsum([0] + acorde.intervals)
            root_midi = row.get("__root_midi", 60)
            acorde.notes_abs = [int(root_midi + s) for s in semitonos_rel]

        identity_obj = get_chord_type_from_intervals(acorde.intervals, with_alias=True)
        identity_name = getattr(identity_obj, "name", str(identity_obj))
        identity_aliases = tuple(getattr(identity_obj, "aliases", ()))
        is_named = bool(identity_name and identity_name != "Unknown")
        hist, total = modelo.calcular(acorde)
        hist = np.asarray(hist, dtype=float)
        counts = compute_interval_counts(acorde.intervals)
        total_pairs = float(np.sum(counts))
        n_notes = len(acorde.intervals) + 1
        dyad_bin = determine_dyad_bin(acorde.intervals) if n_notes == 2 else None
        inv_flag = False
        family_id: Optional[object] = None
        inv_rotation: Optional[int] = None

        if has_family:
            raw_family = row.get("__family_id")
            if pd.notna(raw_family):
                try:
                    family_id = int(raw_family)
                except (TypeError, ValueError):
                    family_id = str(raw_family)

        if has_inv_flag:
            raw_flag = row.get("__inv_flag")
            inv_flag = bool(raw_flag) if pd.notna(raw_flag) else False

        if has_inv_source and family_id is None:
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

    musical_inversion_map: dict[Tuple[int, ...], List[int]] = {}
    structural_inversion_map: dict[Tuple[int, ...], List[int]] = {}
    for i, entry in enumerate(entries):
        notes_abs = entry.acorde.notes_abs
        musical_key = tuple(notes_abs)
        musical_inversion_map.setdefault(musical_key, []).append(i)
        pcs = [note % 12 for note in notes_abs]
        structural_key = tuple(norm_0(pcs))
        structural_inversion_map.setdefault(structural_key, []).append(i)

    for i, entry in enumerate(entries):
        notes_abs = entry.acorde.notes_abs
        musical_inversions = get_musical_inversions(notes_abs)
        for inv in musical_inversions:
            key = tuple(inv)
            if key in musical_inversion_map:
                entry.musical_inversion_ids.extend(musical_inversion_map[key])

        structural_inversions = get_structural_inversions(notes_abs)
        for inv in structural_inversions:
            key = tuple(inv)
            if key in structural_inversion_map:
                entry.structural_inversion_ids.extend(structural_inversion_map[key])

        entry.musical_inversion_ids = sorted(set(entry.musical_inversion_ids))
        entry.structural_inversion_ids = sorted(set(entry.structural_inversion_ids))

    return entries


def compute_interval_counts(intervals: Sequence[int]) -> np.ndarray:
    """Cuenta pares por clase de intervalo usando el orden UI."""

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


def norm_0(pcs: List[int]) -> List[int]:
    """Normaliza un conjunto de pitch classes anclándolo en 0."""

    if not pcs:
        return []
    base = pcs[0]
    return sorted([(pc - base) % 12 for pc in pcs])


def get_musical_inversions(notes_abs: List[int]) -> List[List[int]]:
    """Calcula las inversiones musicales de un acorde."""

    inversions = [notes_abs]
    current_notes = list(notes_abs)
    for _ in range(len(notes_abs) - 1):
        new_notes = sorted(current_notes[1:] + [current_notes[0] + 12])
        inversions.append(new_notes)
        current_notes = new_notes
    return inversions


def get_structural_inversions(notes_abs: List[int]) -> List[List[int]]:
    """Calcula las inversiones estructurales de un acorde."""

    musical_inversions = get_musical_inversions(notes_abs)
    structural_inversions = []
    for inv in musical_inversions:
        pcs = [note % 12 for note in inv]
        structural_inversions.append(norm_0(pcs))
    return structural_inversions


def stack_hist(entries: List[ChordEntry]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apila histogramas, totales y conteos en matrices numpy."""

    hist = np.stack([e.hist for e in entries], axis=0)
    totals = np.array([e.total for e in entries], dtype=float)
    counts = np.stack([e.counts for e in entries], axis=0)
    pairs = np.array([e.total_pairs for e in entries], dtype=float)
    notes = np.array([float(e.n_notes) for e in entries], dtype=float)
    return hist, totals, counts, pairs, notes


def l1_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normaliza cada fila al simplex L1."""

    sums = np.sum(matrix, axis=1, keepdims=True)
    sums[np.isclose(sums, 0.0)] = 1.0
    return matrix / sums


__all__ = [
    "ChordEntry",
    "load_chords",
    "compute_interval_counts",
    "determine_dyad_bin",
    "norm_0",
    "get_musical_inversions",
    "get_structural_inversions",
    "stack_hist",
    "l1_normalize",
]

