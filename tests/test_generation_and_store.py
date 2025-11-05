from core import abs_mask_bigint_of, struct_id_of
from gen import gen_struct, gen_total
from store import ChordStore


def test_gen_struct_dyads_cover_all_intervals():
    pcs = set(range(12))
    chords = list(gen_struct(pcs, {2}))
    assert len(chords) == 12
    struct_ids = {ch.meta["struct_id"] for ch in chords}
    assert len(struct_ids) == 7
    spans = sorted(ch.meta["span"] for ch in chords)
    assert spans[0] == 1
    assert spans[-1] == 12


def test_gen_total_contains_diatonic_triads():
    pcs = {0, 2, 4, 5, 7, 9, 11}
    chords = list(gen_total(pcs, 4, 5, {3}))
    produced = {ch.meta["struct_id"] for ch in chords}
    expected_roots = [
        (60, 64, 67),
        (62, 65, 69),
        (64, 67, 71),
        (65, 69, 72),
        (67, 71, 74),
        (69, 72, 76),
        (71, 74, 77),
    ]
    expected = {struct_id_of(ch).as_key() for ch in expected_roots}
    assert expected.issubset(produced)


def test_chord_store_memory_backend_roundtrip():
    store = ChordStore()
    chord = (60, 64, 67)
    generated = next(
        ch
        for ch in gen_total({0, 2, 4, 5, 7, 9, 11}, 4, 4, {3})
        if ch.midi == chord
    )
    stored = store.get_or_add_abs(chord, generated.meta)
    assert store.has_abs_mask(abs_mask_bigint_of(chord))
    again = store.get_or_add_abs(chord, generated.meta)
    assert stored.chord_id == again.chord_id
    fetched = store.get_by_struct_id(generated.meta["struct_id"])
    assert fetched and fetched[0].midi == chord


def test_chord_store_sqlite_backend_roundtrip(tmp_path):
    path = tmp_path / "chords.db"
    store = ChordStore(backend="sqlite", path=str(path))
    chord = (60, 64, 67)
    generated = next(
        ch
        for ch in gen_total({0, 2, 4, 5, 7, 9, 11}, 4, 4, {3})
        if ch.midi == chord
    )
    stored = store.get_or_add_abs(chord, generated.meta)
    assert store.has_abs_mask(abs_mask_bigint_of(chord))
    again = store.get_or_add_abs(chord, generated.meta)
    assert stored.chord_id == again.chord_id
    fetched = store.get_by_struct_id(generated.meta["struct_id"])
    assert fetched and fetched[0].meta["struct_id"] == generated.meta["struct_id"]
