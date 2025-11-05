import numpy as np

from core import (
    abs_mask_bigint_of,
    chroma01,
    pairwise_dist_hist_mod12,
    pc_mask_of,
    rotation_bass_up,
    struct_id_of,
)


def test_abs_mask_bigint_injective_on_examples():
    chord_a = (60, 64, 67)
    chord_b = (60, 64, 68)
    mask_a = abs_mask_bigint_of(chord_a)
    mask_b = abs_mask_bigint_of(chord_b)
    assert mask_a != mask_b
    assert abs_mask_bigint_of(chord_a) == mask_a


def test_pairwise_hist_mod12_transposition_invariant():
    base = (60, 64, 67, 72)
    transposed = tuple(m + 1 for m in base)
    hist_base = pairwise_dist_hist_mod12(base)
    hist_transposed = pairwise_dist_hist_mod12(transposed)
    np.testing.assert_array_equal(hist_base, hist_transposed)
    assert struct_id_of(base).as_key() == struct_id_of(transposed).as_key()


def test_pc_mask_matches_chroma01():
    chord = (60, 64, 67, 72)
    mask = pc_mask_of(chord)
    chroma = chroma01(chord)
    reconstructed_mask = 0
    for idx, value in enumerate(chroma):
        if value:
            reconstructed_mask |= 1 << idx
    assert mask == reconstructed_mask


def test_rotation_bass_up_preserves_structure():
    chord = (60, 64, 67)
    rotated = rotation_bass_up(chord)
    assert rotated == (64, 67, 72)
    assert struct_id_of(chord).as_key() == struct_id_of(rotated).as_key()
