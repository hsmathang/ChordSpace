"""Utilities for synthesising chord populations beyond direct DB filtering.

This module centralises reusable builders that operate on chord DataFrames
produced by :mod:`tools.data_access`.  The goal is to keep the GUI/controller
layer light while offering musically meaningful transformations (scale-based
aggregations, transpositions, etc.) that can also be reused from tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence

import pandas as pd

from synth_tools import transpose_row


@dataclass(frozen=True)
class ScaleConstraint:
    """Describe a scale for population generation.

    Attributes
    ----------
    pitch_classes:
        Collection of allowed pitch classes (0-11).  They define the scale and
        also act as default transposition steps.
    transposition_steps:
        Optional explicit steps (in semitones) to explore.  When omitted we use
        the pitch classes themselves as offsets relative to the DB-anchored
        representation (i.e. chords stored with root 0).
    """

    pitch_classes: Sequence[int]
    transposition_steps: Sequence[int] | None = None

    def normalised_pitch_classes(self) -> set[int]:
        return {int(pc) % 12 for pc in self.pitch_classes}

    def iter_steps(self) -> Iterator[int]:
        if self.transposition_steps is not None:
            steps = {int(step) % 12 for step in self.transposition_steps}
        else:
            steps = self.normalised_pitch_classes()
        # Preserve deterministic ordering for reproducibility/tests
        for value in sorted(steps):
            yield value


def _parse_interval_sequence(raw: object) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    if isinstance(raw, str):
        stripped = raw.strip().strip("{}[]")
        if not stripped:
            return []
        parts = [part.strip() for part in stripped.split(",") if part.strip()]
        return [int(part) for part in parts]
    raise TypeError(f"Unsupported interval format: {type(raw)!r}")


def _extract_notes_abs(row: Mapping[str, object]) -> list[int]:
    raw_json = row.get("notes_abs_json")
    if isinstance(raw_json, str) and raw_json:
        try:
            data = json.loads(raw_json)
            return [int(val) for val in data]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return _intervals_to_absolute(_parse_interval_sequence(row.get("interval")))


def _intervals_to_absolute(intervals: Iterable[int]) -> list[int]:
    notes = [0]
    for step in intervals:
        notes.append(notes[-1] + int(step))
    return notes


def _ensure_columns(df: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    if template is None or template.empty:
        return df
    for column in template.columns:
        if column not in df.columns:
            df[column] = None
    return df


def generate_scale_population(
    df_source: pd.DataFrame,
    constraint: ScaleConstraint,
    *,
    source_label: str | None = None,
) -> pd.DataFrame:
    """Expand a base population so that chords respect a scale definition.

    Parameters
    ----------
    df_source:
        DataFrame with chords anchored at pitch-class 0 (as retrieved from the
        database).  Must include ``interval`` and/or ``notes_abs_json``.
    constraint:
        :class:`ScaleConstraint` describing the allowed pitch classes and the
        transpositions to explore.
    source_label:
        Optional label injected into the ``__source__`` column for the generated
        chords.  When omitted, any pre-existing value is preserved.

    Returns
    -------
    pandas.DataFrame
        New DataFrame containing the chords whose pitch classes are a subset of
        the provided scale.  The output always contains the same columns as the
        input (plus metadata columns ``__scale_parent_id``,
        ``__scale_transposition`` and ``__scale_generated``).
    """

    if df_source is None or df_source.empty:
        return df_source.copy()

    allowed = constraint.normalised_pitch_classes()
    if not allowed:
        # Nothing to constrain against → return copy to avoid accidental mutation
        return df_source.copy()

    records: list[dict[str, object]] = []
    for _, row in df_source.iterrows():
        row_dict = row.to_dict()
        base_notes = _extract_notes_abs(row_dict)
        if not base_notes:
            continue
        base_pc_set = {note % 12 for note in base_notes}
        for step in constraint.iter_steps():
            step = int(step) % 12
            transposed_notes = [note + step for note in base_notes]
            transposed_pc_set = {note % 12 for note in transposed_notes}
            if not transposed_pc_set.issubset(allowed):
                continue
            if step == 0 and base_pc_set.issubset(allowed):
                record = dict(row_dict)
            else:
                try:
                    record = transpose_row(row_dict, step, tag=row_dict.get("tag"))
                except ValueError:
                    continue
            if source_label is not None:
                record["__source__"] = source_label
            elif "__source__" not in record and "__source__" in row_dict:
                record["__source__"] = row_dict.get("__source__")
            record["__scale_parent_id"] = row_dict.get("id")
            record["__scale_transposition"] = step
            record["__scale_generated"] = step != 0 or not base_pc_set.issubset(allowed)
            records.append(record)

    if not records:
        # Preserve column set for downstream consumers
        empty = pd.DataFrame(columns=df_source.columns)
        for col in ["__scale_parent_id", "__scale_transposition", "__scale_generated"]:
            empty[col] = pd.Series(dtype="Int64" if col != "__scale_generated" else "boolean")
        return empty

    result = pd.DataFrame.from_records(records)
    result = _ensure_columns(result, df_source)
    ordered_cols = list(df_source.columns)
    for extra in ["__scale_parent_id", "__scale_transposition", "__scale_generated"]:
        if extra not in ordered_cols:
            ordered_cols.append(extra)
    for col in result.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)
    return result.loc[:, ordered_cols]


__all__ = ["ScaleConstraint", "generate_scale_population"]

