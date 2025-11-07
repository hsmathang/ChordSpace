"""
Servicio para aplicar filtros a un DataFrame de acordes en memoria.
"""
from __future__ import annotations
import pandas as pd
import json
from tools.data_access import ChordFilters

def _parse_interval(interval_str):
    if isinstance(interval_str, list):
        return interval_str
    return [int(i) for i in interval_str.strip('{}').split(',') if i]

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

    return filtered_df
