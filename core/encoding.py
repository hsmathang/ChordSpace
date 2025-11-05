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

import json

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


def interval_to_ui_bin(interval_semitones: int) -> int:
    """Map semitone distances to the 12-bin UI convention."""

    return (int(interval_semitones) - 1) % 12


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


# ---------------------------------------------------------------------------
# DB-aligned encodings
# ---------------------------------------------------------------------------

def midi_tuple_sorted(midi_list: Sequence[int]) -> Tuple[int, ...]:
    """Return a sorted MIDI tuple with unique entries."""

    ordered = tuple(sorted(int(m) for m in midi_list))
    if not ordered:
        raise ValueError("midi_list cannot be empty")
    for i in range(1, len(ordered)):
        if ordered[i] == ordered[i - 1]:
            raise ValueError("Chord contains unisons (duplicated MIDI numbers)")
    return ordered


def pitch_class_strings(midi_list: Sequence[int]) -> Tuple[str, ...]:
    """Return pitch-class strings in ascending MIDI order."""

    ordered = midi_tuple_sorted(midi_list)
    return tuple(str(pc) for pc in pc_tuple(ordered))


def chord_code_hex(midi_list: Sequence[int]) -> str:
    """Return the hexadecimal pitch-class code used by the historical DB."""

    ordered = midi_tuple_sorted(midi_list)
    pcs = pc_tuple(ordered)
    return "".join(HEX_PITCH_CLASS[int(pc) % 12] for pc in pcs)


def bass_pitch_class(midi_list: Sequence[int]) -> str:
    """Return the bass pitch class as a string."""

    pcs = pitch_class_strings(midi_list)
    return pcs[0] if pcs else "0"


def base_octave_number(midi_list: Sequence[int]) -> int:
    """Return the DB-style octave number (C4 == 4)."""

    ordered = midi_tuple_sorted(midi_list)
    if not ordered:
        return 4
    return 4 + ordered[0] // 12


def abs_mask_hex_of(midi_list: Sequence[int]) -> str:
    """Return the hexadecimal absolute mask representation."""

    mask = abs_mask_bigint_of(midi_list)
    return format(mask, "07X")


def notes_abs_json(midi_list: Sequence[int]) -> str:
    """Return the JSON representation of the absolute MIDI notes."""

    ordered = midi_tuple_sorted(midi_list)
    return json.dumps(list(ordered))


def fundamental_frequencies(midi_list: Sequence[int]) -> Tuple[float, ...]:
    """Return the fundamental frequencies mirroring ``calculate_row``."""

    ordered = midi_tuple_sorted(midi_list)
    freqs = []
    for midi in ordered:
        pc = midi % 12
        base = PITCH_CLASS_BASE_FREQUENCIES.get(pc)
        if base is None:
            raise ValueError(f"Unsupported pitch class: {pc}")
        octave = midi // 12
        freqs.append(base * (2 ** octave))
    return tuple(freqs)


def chord_dataframe_payload(
    midi_list: Sequence[int],
    *,
    tag: str = "ABS_V2",
) -> dict:
    """Return a dictionary aligned with the historical chord DataFrame."""

    ordered = midi_tuple_sorted(midi_list)
    intervals = (
        list(adjacent_intervals_semitones(ordered))
        if len(ordered) > 1
        else []
    )
    return {
        "n": len(ordered),
        "interval": intervals,
        "notes": list(pitch_class_strings(ordered)),
        "bass": bass_pitch_class(ordered),
        "octave": base_octave_number(ordered),
        "frequencies": list(fundamental_frequencies(ordered)),
        "chroma": chroma01(ordered).astype(int).tolist(),
        "tag": tag,
        "code": chord_code_hex(ordered),
        "span_semitones": ordered[-1] - ordered[0] if len(ordered) > 1 else 0,
        "abs_mask_int": abs_mask_bigint_of(ordered),
        "abs_mask_hex": abs_mask_hex_of(ordered),
        "notes_abs_json": notes_abs_json(ordered),
    }
# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

HEX_PITCH_CLASS = "0123456789AB"

# Fundamental frequencies for pitch classes anchored at octave 4.  These
# values mirror the fallback implementation used by ``synth_tools.calculate_row``
# so that adapters relying on :mod:`core.encoding` stay aligned with historical
# chord codification.
PITCH_CLASS_BASE_FREQUENCIES = {
    0: 261.63,
    1: 277.18,
    2: 293.66,
    3: 311.13,
    4: 329.63,
    5: 349.23,
    6: 369.99,
    7: 391.99,
    8: 415.30,
    9: 440.00,
    10: 466.16,
    11: 493.88,
}


