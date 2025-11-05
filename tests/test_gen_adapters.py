from __future__ import annotations

from typing import Sequence

import pytest

from gen import GeneratedChord, generated_chord_to_record
from synth_tools import calculate_row


_EXPECTED_KEYS = (
    "n",
    "interval",
    "notes",
    "bass",
    "octave",
    "frequencies",
    "chroma",
    "tag",
    "code",
    "span_semitones",
    "abs_mask_int",
    "abs_mask_hex",
    "notes_abs_json",
)


def _expected_from_calculate_row(midi: Sequence[int]) -> dict:
    result = calculate_row(tuple(sorted(int(m) for m in midi)))
    return dict(zip(_EXPECTED_KEYS, result))


@pytest.mark.parametrize(
    "midi",
    [
        (0, 7),
        (60, 67),
    ],
)
def test_generated_chord_adapter_matches_dyads(midi: Sequence[int]) -> None:
    chord = GeneratedChord(tuple(midi), meta={"origin": "TEST"})
    record = generated_chord_to_record(chord)
    expected = _expected_from_calculate_row(midi)
    for key in _EXPECTED_KEYS:
        assert record[key] == expected[key], key
    assert record["id"] is None
    assert record["origin"] == "TEST"


@pytest.mark.parametrize(
    "midi",
    [
        (0, 4, 7),
        (62, 65, 69),
    ],
)
def test_generated_chord_adapter_matches_triads(midi: Sequence[int]) -> None:
    chord = GeneratedChord(tuple(midi), meta={})
    record = generated_chord_to_record(chord)
    expected = _expected_from_calculate_row(midi)
    for key in _EXPECTED_KEYS:
        assert record[key] == expected[key], key
    assert record["tag"] == "ABS_V2"
