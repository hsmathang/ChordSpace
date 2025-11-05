"""Filtering primitives used by generation routines."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Set, Tuple

from core import adjacent_intervals_semitones, pc_tuple


def passes_span(midi_list: Sequence[int], max_span: Optional[int]) -> bool:
    """Return whether the voicing span does not exceed ``max_span``."""

    if max_span is None:
        return True
    if not midi_list:
        return True
    return midi_list[-1] - midi_list[0] <= max_span


def passes_pc_requirements(
    midi_list: Sequence[int],
    must_have: Optional[Set[int]] = None,
    must_avoid: Optional[Set[int]] = None,
) -> bool:
    pcs = set(pc_tuple(midi_list))
    if must_have is not None and not must_have.issubset(pcs):
        return False
    if must_avoid is not None and pcs.intersection(must_avoid):
        return False
    return True


def matches_interval_pattern(
    midi_list: Sequence[int], pattern: Optional[Sequence[int]]
) -> bool:
    if pattern is None:
        return True
    if len(midi_list) < 2:
        return False
    intervals = tuple(i % 12 for i in adjacent_intervals_semitones(midi_list))
    pattern_mod = tuple(int(p) % 12 for p in pattern)
    return intervals == pattern_mod
