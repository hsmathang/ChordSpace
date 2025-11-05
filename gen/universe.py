"""Utilities to construct MIDI universes for chord generation."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Set


def _validate_pitch_classes(pcs: Iterable[int]) -> List[int]:
    unique = {int(pc) % 12 for pc in pcs}
    if not unique:
        raise ValueError("Pitch-class alphabet cannot be empty")
    return sorted(unique)


def build_midi_universe(
    pitch_classes: Set[int], o_min: int, o_max: int, edge_pc0: bool = False
) -> List[int]:
    """Return the sorted MIDI universe within the provided constraints."""

    if o_min > o_max:
        raise ValueError("o_min must be <= o_max")
    pcs = _validate_pitch_classes(pitch_classes)
    midi: List[int] = []
    for octave in range(o_min, o_max + 1):
        for pc in pcs:
            midi.append(12 * (octave + 1) + pc)
    if edge_pc0 and 0 in pcs:
        midi.append(12 * (o_max + 2) + 0)
    midi.sort()
    return midi
