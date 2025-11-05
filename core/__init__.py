"""Core utilities for ChordSpace chord generation."""

from .encoding import (
    StructuralSignature,
    abs_mask_bigint_of,
    adjacent_intervals_semitones,
    chroma01,
    chroma_count,
    octave_vector,
    pairwise_dist_hist_mod12,
    pairwise_dist_hist_semitones,
    pairwise_dist_list_semitones,
    pc_mask_of,
    pc_set_sorted,
    pc_tuple,
    pc_tuple_canon0,
    rotation_bass_up,
    span_of,
    struct_id_of,
)

__all__ = [
    "StructuralSignature",
    "abs_mask_bigint_of",
    "adjacent_intervals_semitones",
    "chroma01",
    "chroma_count",
    "octave_vector",
    "pairwise_dist_hist_mod12",
    "pairwise_dist_hist_semitones",
    "pairwise_dist_list_semitones",
    "pc_mask_of",
    "pc_set_sorted",
    "pc_tuple",
    "pc_tuple_canon0",
    "rotation_bass_up",
    "span_of",
    "struct_id_of",
]
