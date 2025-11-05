"""Chord generation algorithms."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from core import (
    abs_mask_bigint_of,
    adjacent_intervals_semitones,
    chroma01,
    octave_vector,
    pc_mask_of,
    pc_tuple,
    pc_tuple_canon0,
    span_of,
    struct_id_of,
)

from .universe import build_midi_universe


@dataclass(frozen=True)
class GeneratedChord:
    midi: Tuple[int, ...]
    meta: Dict[str, object]


def _passes_filters(
    midi_tuple: Tuple[int, ...],
    *,
    max_span: Optional[int] = None,
    must_have_pcs: Optional[set[int]] = None,
    must_avoid_pcs: Optional[set[int]] = None,
    interval_pattern: Optional[Tuple[int, ...]] = None,
) -> bool:
    if max_span is not None and span_of(midi_tuple) > max_span:
        return False
    pcs = set(pc_tuple(midi_tuple))
    if must_have_pcs is not None and not must_have_pcs.issubset(pcs):
        return False
    if must_avoid_pcs is not None and must_avoid_pcs.intersection(pcs):
        return False
    if interval_pattern is not None:
        intervals = tuple(i % 12 for i in adjacent_intervals_semitones(midi_tuple))
        if intervals != tuple(p % 12 for p in interval_pattern):
            return False
    return True


def _metadata_for_total(midi_tuple: Tuple[int, ...], origin: str) -> Dict[str, object]:
    struct = struct_id_of(midi_tuple)
    metadata: Dict[str, object] = {
        "abs_mask_bigint": abs_mask_bigint_of(midi_tuple),
        "pc_mask": pc_mask_of(midi_tuple),
        "n": len(midi_tuple),
        "span": span_of(midi_tuple),
        "octave_vector": octave_vector(midi_tuple),
        "pc_tuple_canon0": pc_tuple_canon0(midi_tuple),
        "struct_id": struct.as_key(),
        "origin": origin,
        "chroma01": chroma01(midi_tuple),
    }
    return metadata


def gen_total(
    pitch_classes: set[int],
    o_min: int,
    o_max: int,
    N: Iterable[int],
    *,
    edge_pc0: bool = False,
    early_filters: Optional[Dict[str, object]] = None,
) -> Iterator[GeneratedChord]:
    """Generate all absolute chords inside the requested universe."""

    universe = build_midi_universe(pitch_classes, o_min, o_max, edge_pc0=edge_pc0)
    filters = early_filters or {}
    origin = f"GEN_TOTAL(S={sorted({pc % 12 for pc in pitch_classes})},O=[{o_min},{o_max}],N={sorted({int(n) for n in N})},edge_pc0={edge_pc0})"
    allowed_sizes = sorted({int(n) for n in N if int(n) >= 2})
    for size in allowed_sizes:
        for combo in combinations(universe, size):
            midi_tuple = tuple(combo)
            if not _passes_filters(
                midi_tuple,
                max_span=filters.get("max_span"),
                must_have_pcs=filters.get("must_have_pcs"),
                must_avoid_pcs=filters.get("must_avoid_pcs"),
                interval_pattern=filters.get("interval_pattern"),
            ):
                continue
            yield GeneratedChord(midi_tuple, _metadata_for_total(midi_tuple, origin))


def _canonical_options(pitch_classes: set[int]) -> List[Tuple[int, ...]]:
    pcs_sorted = sorted({int(pc) % 12 for pc in pitch_classes})
    options = []
    for base in pcs_sorted:
        canon = tuple((pc - base) % 12 for pc in pcs_sorted)
        if canon not in options:
            options.append(canon)
    return options


def _structural_backtrack(
    canonical_set: Tuple[int, ...],
    size: int,
    max_span_struct: int,
) -> Iterator[Tuple[int, ...]]:
    values = sorted(set(canonical_set))

    def recurse(prefix: List[int]) -> Iterator[Tuple[int, ...]]:
        if len(prefix) == size:
            yield tuple(prefix)
            return
        last = prefix[-1]
        for pc in values:
            candidate = pc
            while candidate <= max_span_struct:
                if candidate > last:
                    prefix.append(candidate)
                    yield from recurse(prefix)
                    prefix.pop()
                candidate += 12

    yield from recurse([0])


def gen_struct(
    pitch_classes: set[int],
    N: Iterable[int],
    *,
    max_span_struct: int = 12,
) -> Iterator[GeneratedChord]:
    """Generate structural chords anchored at zero."""

    if max_span_struct not in (12, 24):
        raise ValueError("max_span_struct must be either 12 or 24")
    allowed_sizes = sorted({int(n) for n in N if int(n) >= 2})
    seen: set[Tuple[int, ...]] = set()
    for canonical_set in _canonical_options(pitch_classes):
        for size in allowed_sizes:
            for pattern in _structural_backtrack(canonical_set, size, max_span_struct):
                if pattern in seen:
                    continue
                seen.add(pattern)
                origin = f"GEN_STRUCT(S={sorted({pc % 12 for pc in pitch_classes})},N={allowed_sizes},max_span_struct={max_span_struct})"
                struct_sig = struct_id_of(pattern)
                meta = {
                    "pc_mask": pc_mask_of(pattern),
                    "n": len(pattern),
                    "span": pattern[-1] if pattern else 0,
                    "struct_id": struct_sig.as_key(),
                    "origin": origin,
                }
                yield GeneratedChord(pattern, meta)
