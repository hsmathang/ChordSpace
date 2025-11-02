import json

import numpy as np
import pandas as pd

from synth_tools import make_inversions_df


def _build_source_dataframe() -> pd.DataFrame:
    base_notes = [0, 10, 15, 22]
    return pd.DataFrame(
        [
            {
                "id": 101,
                "n": len(base_notes),
                "interval": [10, 5, 7],
                "notes": [str(n % 12) for n in base_notes],
                "bass": str(base_notes[0] % 12),
                "octave": 4,
                "frequencies": [],
                "chroma": [],
                "tag": "SRC",
                "code": "TEST",
                "span_semitones": max(base_notes) - min(base_notes),
                "abs_mask_int": int(np.sum(1 << np.array(base_notes))),
                "abs_mask_hex": "",
                "notes_abs_json": json.dumps(base_notes),
            }
        ]
    )


def test_make_inversions_df_tracks_families_and_rotations():
    df_src = _build_source_dataframe()
    result = make_inversions_df(df_src, tag="SRC", include_original=True, allow_out_of_range=True)

    assert {"__inv_flag", "__family_id", "__family_size", "__inv_rotation"}.issubset(result.columns)

    assert result["__family_id"].nunique() == 1
    assert int(result["__family_size"].iloc[0]) == len(result)

    base_rows = result[~result["__inv_flag"]]
    assert len(base_rows) == 1
    assert int(base_rows["__inv_source_id"].iloc[0]) == 101

    generated = result[result["__inv_flag"]]
    rotations = sorted(int(rot) for rot in generated["__inv_rotation"].dropna())
    assert rotations == [1, 2, 3]

    # Notas > 24 deberían estar presentes cuando allow_out_of_range=True
    max_note = max(max(json.loads(row)) for row in generated["notes_abs_json"])
    assert max_note > 24


def test_make_inversions_df_default_range_limits_notes():
    df_src = _build_source_dataframe()
    restricted = make_inversions_df(df_src, tag="SRC", include_original=True)

    generated = restricted[restricted["__inv_flag"]]
    assert not generated.empty
    max_note = max(max(json.loads(row)) for row in generated["notes_abs_json"])
    assert max_note <= 24
