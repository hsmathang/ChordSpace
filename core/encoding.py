"""Core encoding utilities for chord generation outputs.

This module centralises the representation logic described in
``docs/modelo_computacional_de_generacion_y_tratamiento_de_acordes_v_1.md``.
The functions operate on sorted MIDI lists that contain at least two
entries and no duplicated MIDI numbers (unisons).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Iterable, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class StructuralSignature:
    """Canonical structural identifier for a chord.

    Attributes
    ----------
    pc_mask_canon0:
        Integer bitmask describing the pitch-class set after anchoring the
        lowest pitch class at zero.
    interval_signature:
        Tuple of adjacent pitch-class intervals (cyclic) for the canonical
        representative.
    """

    pc_mask_canon0: int
    interval_signature: Tuple[int, ...]

    def as_key(self) -> Tuple[int, Tuple[int, ...]]:
        """Return tuple suitable for dictionary keys."""

        return (self.pc_mask_canon0, self.interval_signature)


# ---------------------------------------------------------------------------
# Pitch-class projections
# ---------------------------------------------------------------------------

def _validate_midi_list(midi_list: Sequence[int]) -> Tuple[int, ...]:
    if len(midi_list) < 2:
        raise ValueError("A chord requires at least two MIDI entries")
    ordered = tuple(sorted(int(m) for m in midi_list))
    for i in range(1, len(ordered)):
        if ordered[i] == ordered[i - 1]:
            raise ValueError("Chord contains unisons (duplicated MIDI numbers)")
    return ordered


def pc_tuple(midi_list: Sequence[int]) -> Tuple[int, ...]:
    """Return the pitch classes in voicing order.

    The computation keeps the incoming order – it is up to callers to provide
    the voicing they care about. Values are always mapped to ``[0, 11]``.
    """

    return tuple(int(m) % 12 for m in midi_list)


def pc_set_sorted(midi_list: Sequence[int]) -> Tuple[int, ...]:
    """Return a sorted tuple with unique pitch classes."""

    if not midi_list:
        raise ValueError("midi_list cannot be empty")
    pcs = sorted({int(m) % 12 for m in midi_list})
    return tuple(pcs)


def pc_tuple_canon0(midi_list: Sequence[int]) -> Tuple[int, ...]:
    """Return the canonical pitch-class tuple with minimum set to zero."""

    ordered = _validate_midi_list(midi_list)
    base = ordered[0]
    pcs: list[int] = []
    seen: set[int] = set()
    for m in ordered:
        pc = m % 12
        if pc in seen:
            continue
        seen.add(pc)
        pcs.append((m - base) % 12)
    if not pcs:
        return tuple()
    if len(pcs) == 1:
        return (0,)

    intervals = []
    for i in range(len(pcs)):
        nxt = pcs[(i + 1) % len(pcs)]
        diff = (nxt - pcs[i]) % 12
        if diff == 0:
            diff = 12
        intervals.append(diff)

    best_rot = 0
    best_signature = tuple(intervals)
    for i in range(1, len(pcs)):
        rotated_sig = tuple(intervals[i:] + intervals[:i])
        if rotated_sig < best_signature:
            best_signature = rotated_sig
            best_rot = i

    ref = pcs[best_rot]
    rotated = [((pcs[(best_rot + i) % len(pcs)] - ref) % 12) for i in range(len(pcs))]
    rotated.sort()
    return tuple(rotated)


# ---------------------------------------------------------------------------
# Chroma vectors
# ---------------------------------------------------------------------------

def chroma01(midi_list: Sequence[int]) -> np.ndarray:
    """Binary chroma vector (12-D) describing the pitch-class set."""

    mask = np.zeros(12, dtype=int)
    for pc in pc_set_sorted(midi_list):
        mask[pc] = 1
    return mask


def chroma_count(midi_list: Sequence[int]) -> np.ndarray:
    """Chroma vector counting occurrences of each pitch class."""

    counts = np.zeros(12, dtype=int)
    for pc in pc_tuple(midi_list):
        counts[pc] += 1
    return counts


# ---------------------------------------------------------------------------
# Interval analysis
# ---------------------------------------------------------------------------

def adjacent_intervals_semitones(midi_list: Sequence[int]) -> Tuple[int, ...]:
    """Return consecutive semitone distances within the voicing."""

    ordered = _validate_midi_list(midi_list)
    return tuple(ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1))


def pairwise_dist_list_semitones(midi_list: Sequence[int]) -> Tuple[int, ...]:
    """Return all pairwise semitone distances (j > i)."""

    ordered = _validate_midi_list(midi_list)
    distances = [b - a for a, b in combinations(ordered, 2)]
    return tuple(distances)


def pairwise_dist_hist_semitones(
    midi_list: Sequence[int], max_bins: int | None = None
) -> np.ndarray:
    """Histogram of pairwise semitone distances.

    Parameters
    ----------
    midi_list:
        Input voicing.
    max_bins:
        Optional fixed length for the histogram. When ``None`` the histogram
        length equals the span of the chord.
    """

    distances = pairwise_dist_list_semitones(midi_list)
    if not distances:
        return np.zeros(0, dtype=int)
    span = max(distances)
    length = span if max_bins is None else max_bins
    hist = np.zeros(length, dtype=int)
    for d in distances:
        if d <= 0:
            continue
        idx = min(d - 1, length - 1)
        hist[idx] += 1
    return hist


def pairwise_dist_hist_mod12(midi_list: Sequence[int]) -> np.ndarray:
    """Histogram of pairwise distances modulo 12.

    The bin at index 0 corresponds to a distance of one semitone; the last
    bin collects distances congruent to zero modulo twelve (octaves), which
    are allowed because the input is free from unisons but may span octaves.
    """

    distances = pairwise_dist_list_semitones(midi_list)
    hist = np.zeros(12, dtype=int)
    for d in distances:
        hist[d % 12] += 1
    hist[0] = 0  # policy: no unisons are expected
    return hist


# ---------------------------------------------------------------------------
# Signatures and metadata
# ---------------------------------------------------------------------------

def span_of(midi_list: Sequence[int]) -> int:
    ordered = _validate_midi_list(midi_list)
    return ordered[-1] - ordered[0]


def pc_mask_of(midi_list: Sequence[int]) -> int:
    mask = 0
    for pc in pc_set_sorted(midi_list):
        mask |= 1 << pc
    return mask


def abs_mask_bigint_of(midi_list: Sequence[int]) -> int:
    ordered = _validate_midi_list(midi_list)
    mask = 0
    for m in ordered:
        mask |= 1 << m
    return mask


@lru_cache(maxsize=8192)
def _cyclic_interval_signature(canon: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(canon) < 2:
        return tuple()
    diffs = []
    for i in range(len(canon)):
        nxt = canon[(i + 1) % len(canon)]
        diff = (nxt - canon[i]) % 12
        if diff == 0:
            diff = 12
        diffs.append(diff)
    candidates = [tuple(diffs[i:] + diffs[:i]) for i in range(len(diffs))]
    return min(candidates)


def struct_id_of(midi_list: Sequence[int]) -> StructuralSignature:
    canon = pc_tuple_canon0(midi_list)
    mask = pc_mask_of(canon)
    signature = _cyclic_interval_signature(canon)
    return StructuralSignature(mask, signature)


def octave_vector(midi_list: Sequence[int]) -> Tuple[int, ...]:
    """Return the octave indices of the MIDI list."""

    ordered = _validate_midi_list(midi_list)
    return tuple(m // 12 - 1 for m in ordered)


def rotation_bass_up(midi_list: Sequence[int]) -> Tuple[int, ...]:
    """Rotate the bass to the top octave-preserving, keeping order sorted."""

    ordered = _validate_midi_list(midi_list)
    rotated = list(ordered[1:]) + [ordered[0] + 12]
    rotated.sort()
    return tuple(rotated)
