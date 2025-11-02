from __future__ import annotations

import json

import pandas as pd

from synth_tools import calculate_row
from tools.population_builders import ScaleConstraint, generate_scale_population


def _record_from_notes(notes: list[int], chord_id: int) -> dict:
    (
        n,
        intervals,
        pitch_classes,
        bass,
        octave,
        frequencies,
        chroma,
        tag,
        code,
        span_semitones,
        abs_mask_int,
        abs_mask_hex,
        notes_abs_json,
    ) = calculate_row(tuple(notes))
    return {
        "id": chord_id,
        "n": n,
        "interval": intervals,
        "notes": pitch_classes,
        "bass": bass,
        "octave": octave,
        "frequencies": frequencies,
        "chroma": chroma,
        "tag": tag,
        "code": code,
        "span_semitones": span_semitones,
        "abs_mask_int": abs_mask_int,
        "abs_mask_hex": abs_mask_hex,
        "notes_abs_json": notes_abs_json,
    }


def test_generate_scale_population_diatonic_triads():
    # Base chords anchored at 0 representing maj, min, dim triads
    records = [
        _record_from_notes([0, 4, 7], chord_id=1),
        _record_from_notes([0, 3, 7], chord_id=2),
        _record_from_notes([0, 3, 6], chord_id=3),
    ]
    df_source = pd.DataFrame(records)

    constraint = ScaleConstraint((0, 2, 4, 5, 7, 9, 11))
    df_scale = generate_scale_population(df_source, constraint, source_label="custom")

    assert len(df_scale) == 7

    pcs_sets = {
        frozenset(int(val) % 12 for val in json.loads(str(row["notes_abs_json"])))
        for _, row in df_scale.iterrows()
    }
    expected = {
        frozenset({0, 4, 7}),
        frozenset({2, 5, 9}),
        frozenset({4, 7, 11}),
        frozenset({5, 9, 0}),
        frozenset({7, 11, 2}),
        frozenset({9, 0, 4}),
        frozenset({11, 2, 5}),
    }
    assert pcs_sets == expected

    # Ensure metadata tracks origin
    assert set(df_scale["__scale_parent_id"].dropna().astype(int).unique()) == {1, 2, 3}
    # At least one original chord (shift 0) should be flagged as not generated
    assert not df_scale.loc[df_scale["__scale_transposition"] == 0, "__scale_generated"].any()
