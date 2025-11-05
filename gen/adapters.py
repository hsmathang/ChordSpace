"""Adapters that bridge generated chords with historical DataFrame schemas."""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional

from core import chord_dataframe_payload

from .generate import GeneratedChord

_DEFAULT_TAG = "ABS_V2"


def generated_chord_to_record(
    chord: GeneratedChord,
    *,
    chord_id: Optional[int] = None,
    tag: Optional[str] = None,
) -> Dict[str, object]:
    """Return a DB-aligned dictionary for ``GeneratedChord`` instances."""

    payload = chord_dataframe_payload(
        chord.midi,
        tag=tag or chord.meta.get("tag") or _DEFAULT_TAG,
    )
    record: Dict[str, object] = {"id": chord_id, **payload}
    for key, value in chord.meta.items():
        if key not in record:
            record[key] = value
    return record


def iter_chord_records(
    chords: Iterable[GeneratedChord],
    *,
    start_id: int = 1,
    tag: Optional[str] = None,
) -> Iterator[Dict[str, object]]:
    """Yield DB-aligned dictionaries for an iterable of chords."""

    current_id = start_id
    for chord in chords:
        yield generated_chord_to_record(chord, chord_id=current_id, tag=tag)
        current_id += 1


__all__ = ["generated_chord_to_record", "iter_chord_records"]
