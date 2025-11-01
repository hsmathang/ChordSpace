"""
Centralized data-access helpers for building and executing chord population queries.

This module abstracts the SQL generation that was previously spread across the GUI
and other utilities.  It exposes composable filters, sampling strategies and column
profiles so that callers (GUI, CLI, tests) can request data without crafting raw SQL.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

import config as cfg
from tools.query_registry import get_all_queries, resolve_query_sql
from tools.experiment_inversions import _parse_pop_spec, _build_population

try:  # pragma: no cover - prefer packaged executor
    from chordcodex.model import QueryExecutor  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from synth_tools import QueryExecutor  # type: ignore

# --------------------------------------------------------------------------- #
# Column profiles
# --------------------------------------------------------------------------- #


class ColumnProfile(str, Enum):
    """Named column selections for different UI contexts."""

    MINIMAL = "minimal"
    AUDIO = "audio"
    VISUAL = "visual"
    FULL = "full"

    @property
    def columns(self) -> Tuple[str, ...]:
        base_minimal = ("id", "n", "interval", "notes")
        base_audio = base_minimal + ("frequencies", "octave")
        base_visual = base_audio + (
            "tag",
            "code",
            "chroma",
            "span_semitones",
            "abs_mask_int",
            "abs_mask_hex",
            "notes_abs_json",
        )
        mapping: Dict[ColumnProfile, Tuple[str, ...]] = {
            ColumnProfile.MINIMAL: base_minimal,
            ColumnProfile.AUDIO: base_audio,
            ColumnProfile.VISUAL: base_visual,
            ColumnProfile.FULL: tuple(),
        }
        return mapping[self]


# --------------------------------------------------------------------------- #
# Sampling strategies and filters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SampleStrategy:
    mode: str
    percent: Optional[float] = None
    fraction: Optional[float] = None
    limit: Optional[int] = None

    @staticmethod
    def table_sample(percent: float) -> SampleStrategy:
        return SampleStrategy(mode="table_sample", percent=percent)

    @staticmethod
    def random_fraction(fraction: float, limit: Optional[int] = None) -> SampleStrategy:
        return SampleStrategy(mode="random_fraction", fraction=fraction, limit=limit)


@dataclass
class ChordFilters:
    cardinalities: Optional[List[int]] = None
    min_cardinality: Optional[int] = None
    # Legacy: exact sets list (kept for compatibility)
    interval_sets: Optional[List[List[int]]] = None
    root_values: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    max_interval_sum: Optional[int] = None
    include_ids: Optional[List[int]] = None
    exclude_ids: Optional[List[int]] = None
    limit: Optional[int] = None
    order_by: Optional[str] = "id"
    sample_strategy: Optional[SampleStrategy] = None
    span_min: Optional[int] = None
    span_max: Optional[int] = None
    include_pitch_classes: Optional[List[int]] = None
    exclude_pitch_classes: Optional[List[int]] = None
    # Legacy: single exact interval vector
    interval_exact: Optional[List[int]] = None
    codes: Optional[List[str]] = None

    # New flexible filters
    include_pc_mode: Optional[str] = None  # one of: contains_all, contains_any, subset_of
    interval_mode: Optional[str] = None  # one of: exact, subseq, any_value
    interval_patterns: Optional[List[List[int]]] = None  # multiple patterns
    interval_values: Optional[List[int]] = None  # for any_value mode


@dataclass
class PopulationPreset:
    filters: Optional[ChordFilters] = None
    raw_sql: Optional[str] = None  # e.g. "config:QUERY_TRIADS_WITH_INVERSIONS"


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #


def get_executor() -> QueryExecutor:
    return QueryExecutor(**cfg.config_db)


def _normalize_columns(profile: ColumnProfile, df: pd.DataFrame) -> pd.DataFrame:
    columns = profile.columns
    if not columns:
        return df
    existing = [col for col in columns if col in df.columns]
    missing = [col for col in columns if col not in df.columns]
    for col in missing:
        df[col] = None
    ordered = list(columns) + [col for col in df.columns if col not in columns]
    return df.loc[:, ordered]


def _rehydrate_profile_columns(
    df: pd.DataFrame,
    profile: ColumnProfile,
    exec_obj,
) -> pd.DataFrame:
    """Ensure all profile columns are populated when the upstream query omitted them."""

    columns = profile.columns
    if not columns or "id" not in df.columns:
        return df

    try:
        id_series = pd.to_numeric(df["id"], errors="coerce")
    except Exception:
        return df

    try:
        valid_ids_list = [int(v) for v in id_series.dropna().astype(int).unique().tolist()]
    except Exception:
        valid_ids_list = []
    if not valid_ids_list:
        return df

    nullish_cols: List[str] = []
    for col in columns:
        if col == "id":
            continue
        if col not in df.columns:
            nullish_cols.append(col)
        else:
            series = df[col]
            if series.isna().all():
                nullish_cols.append(col)

    if not nullish_cols:
        return df

    cols_sql = ", ".join(["id"] + nullish_cols)
    try:
        extra = exec_obj.as_pandas(
            f"SELECT {cols_sql} FROM chords WHERE id = ANY(%s)",
            (valid_ids_list,),
        )
    except Exception:
        return df

    if extra.empty:
        return df

    extra = extra.drop_duplicates(subset=["id"]).set_index("id")
    id_int = id_series.astype("Int64")

    for col in nullish_cols:
        if col not in extra.columns:
            continue
        if col not in df.columns:
            df[col] = None
        replacement = id_int.map(extra[col])
        if replacement is not None:
            df[col] = df[col].where(df[col].notna(), replacement)

    return df


def _format_int_list(values: Iterable[int]) -> str:
    return ", ".join(str(int(v)) for v in values)


def _format_str_list(values: Iterable[str]) -> str:
    return ", ".join(f"'{str(v)}'" for v in values)


def _format_interval(interval: Iterable[int]) -> str:
    return f"ARRAY[{_format_int_list(interval)}]::integer[]"


def build_sql(filters: ChordFilters, profile: ColumnProfile) -> str:
    select_cols = profile.columns
    select_expr = "*" if not select_cols else ", ".join(select_cols)

    sample_clause = ""
    conditions: List[str] = []
    order_by = filters.order_by
    limit = filters.limit

    if filters.sample_strategy:
        if filters.sample_strategy.mode == "table_sample" and filters.sample_strategy.percent:
            percent = max(0.01, min(100.0, filters.sample_strategy.percent))
            sample_clause = f" TABLESAMPLE SYSTEM ({percent})"
        elif filters.sample_strategy.mode == "random_fraction" and filters.sample_strategy.fraction:
            frac = max(0.0, min(1.0, filters.sample_strategy.fraction))
            conditions.append(f"random() < {frac}")
            if filters.sample_strategy.limit is not None and limit is None:
                limit = filters.sample_strategy.limit

    if filters.cardinalities:
        cond = f"n = ANY(ARRAY[{_format_int_list(filters.cardinalities)}]::integer[])"
        conditions.append(cond)
    if filters.min_cardinality:
        conditions.append(f"n >= {int(filters.min_cardinality)}")
    # Interval filters (new modes first)
    if filters.interval_mode:
        mode = (filters.interval_mode or "").strip().lower()
        if mode == "exact":
            patterns: List[List[int]] = []
            if filters.interval_patterns:
                patterns.extend([list(map(int, p)) for p in filters.interval_patterns if p])
            # Back-compat fallbacks
            if not patterns and filters.interval_sets:
                patterns.extend([list(map(int, p)) for p in filters.interval_sets if p])
            if not patterns and filters.interval_exact:
                patterns.append(list(map(int, filters.interval_exact)))
            if patterns:
                interval_conditions = [f"interval = {_format_interval(p)}" for p in patterns]
                conditions.append("(" + " OR ".join(interval_conditions) + ")")
        elif mode == "subseq":
            patterns = [list(map(int, p)) for p in (filters.interval_patterns or []) if p]
            subseq_conds: List[str] = []
            for p in patterns:
                m = len(p)
                if m <= 0:
                    continue
                arr = _format_interval(p)
                # exists subarray slice equals pattern
                subseq_conds.append(
                    "EXISTS ("
                    "SELECT 1 FROM generate_subscripts(interval, 1) AS s "
                    f"WHERE s >= 1 AND s + {m} - 1 <= array_length(interval, 1) "
                    f"AND interval[s:s+{m}-1] = {arr}"
                    ")"
                )
            if subseq_conds:
                conditions.append("(" + " OR ".join(subseq_conds) + ")")
        elif mode == "any_value":
            values = list(dict.fromkeys(int(v) for v in (filters.interval_values or [])))
            if not values and filters.interval_patterns:
                # Interpret single-length patterns as values
                for p in filters.interval_patterns:
                    if len(p) == 1:
                        values.append(int(p[0]))
            if values:
                arr_vals = _format_int_list(values)
                conditions.append(
                    "EXISTS (SELECT 1 FROM unnest(interval) AS x WHERE x = ANY(ARRAY[" + arr_vals + "]::integer[]))"
                )
    else:
        # Back-compat legacy behavior
        if filters.interval_sets:
            interval_conditions = [f"interval = {_format_interval(interval)}" for interval in filters.interval_sets]
            conditions.append("(" + " OR ".join(interval_conditions) + ")")
    if filters.root_values:
        cond = f"notes[1] = ANY(ARRAY[{_format_str_list(filters.root_values)}])"
        conditions.append(cond)
    if filters.tags:
        cond = f"tag = ANY(ARRAY[{_format_str_list(filters.tags)}])"
        conditions.append(cond)
    if filters.max_interval_sum is not None:
        conditions.append(
            "(SELECT COALESCE(SUM(i), 0) FROM unnest(interval) AS i) <= " + str(int(filters.max_interval_sum))
        )
    if filters.span_min is not None:
        conditions.append(f"span_semitones >= {int(filters.span_min)}")
    if filters.span_max is not None:
        conditions.append(f"span_semitones <= {int(filters.span_max)}")
    if filters.interval_exact:
        conditions.append(f"interval = {_format_interval(filters.interval_exact)}")
    if filters.codes:
        codes = ", ".join(f"'{str(code)}'" for code in filters.codes)
        conditions.append(f"code = ANY(ARRAY[{codes}]::varchar[])")
    if filters.include_ids:
        cond = f"id = ANY(ARRAY[{_format_int_list(filters.include_ids)}]::integer[])"
        conditions.append(cond)
    if filters.exclude_ids:
        cond = f"id <> ALL(ARRAY[{_format_int_list(filters.exclude_ids)}]::integer[])"
        conditions.append(cond)
    # Pitch-class filters with modes
    if filters.include_pitch_classes:
        pcs = [int(pc) for pc in filters.include_pitch_classes]
        mode = (filters.include_pc_mode or "contains_all").strip().lower()
        if mode == "contains_any":
            any_clause = []
            for pc in pcs:
                idx = int(pc) + 1
                any_clause.append(f"chroma[{idx}] = 1")
            if any_clause:
                conditions.append("(" + " OR ".join(any_clause) + ")")
        elif mode == "subset_of":
            # All positions not in P must be zero
            allowed = set(pcs)
            subset_clauses = []
            for j in range(12):
                if j not in allowed:
                    subset_clauses.append(f"chroma[{j+1}] = 0")
            if subset_clauses:
                conditions.append("(" + " AND ".join(subset_clauses) + ")")
        else:  # contains_all (default)
            for pc in pcs:
                idx = int(pc) + 1
                conditions.append(f"chroma[{idx}] = 1")
    if filters.exclude_pitch_classes:
        for pc in filters.exclude_pitch_classes:
            idx = int(pc) + 1
            conditions.append(f"chroma[{idx}] = 0")

    sql_lines = [f"SELECT {select_expr}", f"FROM chords{sample_clause}"]
    if conditions:
        sql_lines.append("WHERE " + "\n  AND ".join(conditions))
    if order_by:
        sql_lines.append(f"ORDER BY {order_by}")
    if limit is not None:
        sql_lines.append(f"LIMIT {int(limit)}")
    return "\n".join(sql_lines)


# --------------------------------------------------------------------------- #
# Preset registry
# --------------------------------------------------------------------------- #


PRESETS: Dict[str, PopulationPreset] = {
    "QUERY_CHORDS_3_NOTES": PopulationPreset(
        filters=ChordFilters(cardinalities=[3], order_by="id", limit=60)
    ),
    "QUERY_CHORDS_3_NOTES_ALL": PopulationPreset(
        filters=ChordFilters(cardinalities=[3], order_by="id")
    ),
    "QUERY_CHORDS_3_NOTES_SAMPLE_25": PopulationPreset(
        filters=ChordFilters(cardinalities=[3], sample_strategy=SampleStrategy.table_sample(25.0), order_by="id")
    ),
    "QUERY_CHORDS_3_NOTES_SAMPLE_50": PopulationPreset(
        filters=ChordFilters(cardinalities=[3], sample_strategy=SampleStrategy.table_sample(50.0), order_by="id")
    ),
    "QUERY_CHORDS_3_NOTES_SAMPLE_75": PopulationPreset(
        filters=ChordFilters(cardinalities=[3], sample_strategy=SampleStrategy.table_sample(75.0), order_by="id")
    ),
    "QUERY_CHORDS_4_NOTES_ALL": PopulationPreset(
        filters=ChordFilters(cardinalities=[4], order_by="id")
    ),
    "QUERY_CHORDS_4_NOTES_SAMPLE_25": PopulationPreset(
        filters=ChordFilters(cardinalities=[4], sample_strategy=SampleStrategy.table_sample(25.0), order_by="id")
    ),
    "QUERY_CHORDS_4_NOTES_SAMPLE_50": PopulationPreset(
        filters=ChordFilters(cardinalities=[4], sample_strategy=SampleStrategy.table_sample(50.0), order_by="id")
    ),
    "QUERY_CHORDS_4_NOTES_SAMPLE_75": PopulationPreset(
        filters=ChordFilters(cardinalities=[4], sample_strategy=SampleStrategy.table_sample(75.0), order_by="id")
    ),
    "QUERY_CHORDS_5_NOTES_ALL": PopulationPreset(
        filters=ChordFilters(cardinalities=[5], order_by="id")
    ),
    "QUERY_CHORDS_5_NOTES_SAMPLE_25": PopulationPreset(
        filters=ChordFilters(cardinalities=[5], sample_strategy=SampleStrategy.table_sample(25.0), order_by="id")
    ),
    "QUERY_CHORDS_5_NOTES_SAMPLE_50": PopulationPreset(
        filters=ChordFilters(cardinalities=[5], sample_strategy=SampleStrategy.table_sample(50.0), order_by="id")
    ),
    "QUERY_CHORDS_5_NOTES_SAMPLE_75": PopulationPreset(
        filters=ChordFilters(cardinalities=[5], sample_strategy=SampleStrategy.table_sample(75.0), order_by="id")
    ),
    "QUERY_CHORDS_WITH_NAME": PopulationPreset(
        filters=ChordFilters(
            interval_sets=[
                [2, 5], [3, 3], [3, 4], [4, 3],
                [4, 4], [5, 2], [3, 3, 3], [3, 3, 4],
                [3, 4, 2], [3, 4, 3], [3, 4, 4], [4, 3, 2],
                [4, 3, 3], [4, 3, 4], [4, 3, 7], [4, 4, 2],
                [4, 4, 3], [3, 4, 3, 4], [4, 3, 2, 5], [4, 3, 3, 3],
                [4, 3, 3, 4], [4, 3, 3, 5], [4, 3, 3, 8], [4, 3, 4, 3],
                [4, 3, 4, 7], [3, 4, 3, 4, 3], [4, 3, 3, 4, 3], [4, 3, 3, 4, 7],
                [4, 3, 4, 3, 7], [3, 4, 3, 4, 3, 4],
            ],
            order_by="id",
        )
    ),
    # For more complex queries we fall back to raw SQL in config.py
    "QUERY_TRIADS_WITH_INVERSIONS": PopulationPreset(raw_sql="config:QUERY_TRIADS_WITH_INVERSIONS"),
    "QUERY_TRIADS_ROOT_ONLY_MOBIUS_MAZZOLA": PopulationPreset(raw_sql="config:QUERY_TRIADS_ROOT_ONLY_MOBIUS_MAZZOLA"),
    "QUERY_CHORDS_SPECIFIC_INTERVALS_AND_RANDOM_SAME_OCTAVE": PopulationPreset(
        raw_sql="config:QUERY_CHORDS_SPECIFIC_INTERVALS_AND_RANDOM_SAME_OCTAVE"
    ),
    "QUERY_DYADS_RANDOM_UNIQUE": PopulationPreset(raw_sql="config:QUERY_DYADS_RANDOM_UNIQUE"),
    "QUERY_DYADS_REFERENCE": PopulationPreset(raw_sql="config:QUERY_DYADS_REFERENCE"),
    "QUERY_DYADS_TT_P5_PLUS_RANDOM": PopulationPreset(raw_sql="custom:QUERY_DYADS_TT_P5_PLUS_RANDOM"),
    "QUERY_EXTREME_CLUSTER_10_NOTES": PopulationPreset(raw_sql="config:QUERY_EXTREME_CLUSTER_10_NOTES"),
    "QUERY_RANDOM_CHORD_MORE_THAN_2_NOTES": PopulationPreset(raw_sql="config:QUERY_RANDOM_CHORD_MORE_THAN_2_NOTES"),
    "QUERY_RANDOM_CHORD_MORE_THAN_2_NOTES_CLOSE_OCTAVE": PopulationPreset(
        raw_sql="config:QUERY_RANDOM_CHORD_MORE_THAN_2_NOTES_CLOSE_OCTAVE"
    ),
    "QUERY_SEVENTHS_CORE": PopulationPreset(raw_sql="custom:QUERY_SEVENTHS_CORE"),
    "QUERY_TRIADS_WITH_REPEATED_NOTES": PopulationPreset(raw_sql="config:QUERY_TRIADS_WITH_REPEATED_NOTES"),
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def list_preset_names() -> List[str]:
    return sorted(PRESETS.keys())


def _resolve_raw_sql(identifier: str) -> str:
    if identifier.startswith("config:"):
        attr = identifier.split(":", 1)[1]
        return getattr(cfg, attr)
    if identifier.startswith("custom:"):
        name = identifier.split(":", 1)[1]
        registry = get_all_queries()
        if name in registry:
            return registry[name]["sql"]
        raise KeyError(f"No custom query named '{name}'")
    return identifier


def fetch_population(
    filters: ChordFilters,
    profile: ColumnProfile = ColumnProfile.VISUAL,
    *,
    executor: Optional[QueryExecutor] = None,
) -> pd.DataFrame:
    sql = build_sql(filters, profile)
    exec_obj = executor or get_executor()
    df = exec_obj.as_pandas(sql)
    df = _normalize_columns(profile, df)
    return _rehydrate_profile_columns(df, profile, exec_obj)


def fetch_population_by_name(
    name: str,
    profile: ColumnProfile = ColumnProfile.VISUAL,
    *,
    executor: Optional[QueryExecutor] = None,
) -> pd.DataFrame:
    preset = PRESETS.get(name)
    if preset and preset.filters:
        return fetch_population(preset.filters, profile=profile, executor=executor)

    # Fallback to raw SQL (preset raw or registry)
    if preset and preset.raw_sql:
        sql = _resolve_raw_sql(preset.raw_sql)
    else:
        sql = resolve_query_sql(name)
    exec_obj = executor or get_executor()
    df = exec_obj.as_pandas(sql)
    df = _normalize_columns(profile, df)
    return _rehydrate_profile_columns(df, profile, exec_obj)


def fetch_population_spec(
    spec: str,
    profile: ColumnProfile = ColumnProfile.VISUAL,
    *,
    executor: Optional[QueryExecutor] = None,
) -> pd.DataFrame:
    if ":" in spec:
        ptype, qname = _parse_pop_spec(spec)
        df = _build_population(ptype, qname)
        df = _normalize_columns(profile, df)
        exec_obj = executor or get_executor()
        return _rehydrate_profile_columns(df, profile, exec_obj)
    return fetch_population_by_name(spec, profile=profile, executor=executor)


def fetch_population_with_source(
    spec: str,
    profile: ColumnProfile,
    source_label: str,
    *,
    executor: Optional[QueryExecutor] = None,
) -> pd.DataFrame:
    df = fetch_population_spec(spec, profile=profile, executor=executor).copy()
    df["__source__"] = source_label
    return df


def build_sql_for_ids(ids: Iterable[int], profile: ColumnProfile = ColumnProfile.FULL) -> str:
    filters = ChordFilters(include_ids=[int(i) for i in ids], order_by="id")
    return build_sql(filters, profile)


# --------------------------------------------------------------------------- #
# Transformations: apply A/B/C population mode to arbitrary DataFrames
# --------------------------------------------------------------------------- #


def apply_population_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Apply population mode A/B/C to an arbitrary chord DataFrame.

    - A: return df unchanged
    - B: anchored inversions intersected with DB (include original)
    - C: synthetic inversions (include original)
    """
    mode = (mode or "A").strip().upper()
    if mode == "A":
        return df
    if mode == "B":
        try:
            from tools import experiment_inversions as ei  # lazy import to avoid cycles
            return ei.build_inversions_anchored_db(df, include_original=True)
        except Exception:
            return df
    if mode == "C":
        try:
            from synth_tools import make_inversions_df, DB_TAG  # type: ignore
            return make_inversions_df(df, tag=DB_TAG, include_original=True)
        except Exception:
            return df
    return df
