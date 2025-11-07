"""
Servicio para aplicar filtros a un DataFrame de acordes en memoria.
"""
from __future__ import annotations
import pandas as pd
import json
from tools.data_access import ChordFilters

def _parse_interval(interval_str):
    if isinstance(interval_str, list):
        return [int(v) for v in interval_str]
    if interval_str is None or (isinstance(interval_str, float) and pd.isna(interval_str)):
        return []
    text = str(interval_str).strip("{}[]() ")
    if not text:
        return []
    return [int(i) for i in text.split(',') if i.strip()]


def _contains_subsequence(sequence, pattern):
    if not pattern:
        return True
    seq_len = len(sequence)
    pat_len = len(pattern)
    if pat_len > seq_len:
        return False
    for idx in range(seq_len - pat_len + 1):
        if tuple(sequence[idx : idx + pat_len]) == tuple(pattern):
            return True
    return False

def filter_dataframe(df: pd.DataFrame, filters: ChordFilters) -> pd.DataFrame:
    """
    Aplica los filtros especificados a un DataFrame de acordes.
    """
    if df.empty:
        return df

    filtered_df = df.copy()

    # 1. Filtro por Cardinalidad
    if filters.cardinalities:
        filtered_df = filtered_df[filtered_df['n'].isin(filters.cardinalities)]

    # 2. Filtro por Span
    if filters.span_min is not None:
        filtered_df = filtered_df[filtered_df['span_semitones'] >= filters.span_min]
    if filters.span_max is not None:
        filtered_df = filtered_df[filtered_df['span_semitones'] <= filters.span_max]

    # 3. Filtro por Máximo Intervalo Interno
    if filters.max_internal_interval is not None:
        # La columna 'interval' puede ser una lista o un string tipo '{3,4}'
        intervals = filtered_df['interval'].apply(_parse_interval)
        max_intervals = intervals.apply(lambda x: max(x) if x else 0)
        filtered_df = filtered_df[max_intervals <= filters.max_internal_interval]

    # 4. Filtro por Pitch Classes
    if filters.include_pitch_classes:
        include_pcs = set(filters.include_pitch_classes)
        # 'notes_abs_json' es la fuente más fiable de las notas MIDI absolutas
        pitch_classes_set = filtered_df['notes_abs_json'].apply(lambda x: set(n % 12 for n in json.loads(x)))

        mode = filters.include_pc_mode or 'contains_all'
        if mode == 'contains_all':
            mask = pitch_classes_set.apply(lambda pcs: include_pcs.issubset(pcs))
        elif mode == 'contains_any':
            mask = pitch_classes_set.apply(lambda pcs: not include_pcs.isdisjoint(pcs))
        elif mode == 'subset_of':
            mask = pitch_classes_set.apply(lambda pcs: pcs.issubset(include_pcs))

        filtered_df = filtered_df[mask]

    if filters.exclude_pitch_classes:
        exclude_pcs = set(filters.exclude_pitch_classes)
        pitch_classes_set = filtered_df['notes_abs_json'].apply(lambda x: set(n % 12 for n in json.loads(x)))
        mask = pitch_classes_set.apply(lambda pcs: pcs.isdisjoint(exclude_pcs))
        filtered_df = filtered_df[mask]

    intervals_series = filtered_df['interval'].apply(_parse_interval)

    # 5. Filtro por patrones de intervalos
    patterns: list[list[int]] = []
    if filters.interval_exact:
        patterns.append([int(x) for x in filters.interval_exact])
    if filters.interval_patterns:
        patterns.extend([[int(x) for x in pattern] for pattern in filters.interval_patterns])

    if filters.interval_mode == "exact" and patterns:
        mask = intervals_series.apply(
            lambda seq: any(tuple(seq) == tuple(pattern) for pattern in patterns)
        )
        filtered_df = filtered_df[mask]
        intervals_series = intervals_series[mask]
    elif filters.interval_mode == "subseq" and patterns:
        mask = intervals_series.apply(
            lambda seq: any(_contains_subsequence(seq, pattern) for pattern in patterns)
        )
        filtered_df = filtered_df[mask]
        intervals_series = intervals_series[mask]

    # 6. Filtro por valores individuales de intervalos
    if filters.interval_mode == "any_value" and filters.interval_values:
        values = set(int(v) for v in filters.interval_values)
        mask = intervals_series.apply(lambda seq: any(val in values for val in seq))
        filtered_df = filtered_df[mask]
        intervals_series = intervals_series[mask]

    # 7. Filtro por códigos absolutos
    if filters.codes:
        codes = {code.upper() for code in filters.codes}
        code_series = filtered_df['code'].astype(str).str.upper()
        mask = code_series.apply(lambda c: c in codes)
        filtered_df = filtered_df[mask]

    return filtered_df
