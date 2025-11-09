"""
Servicio para la generación combinatoria de acordes.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from synth_tools import _build_record_from_notes

A4_MIDI = 69
A4_FREQ = 440.0


def _midi_to_freq(note: int) -> float:
    """Convierte un número MIDI en Hz usando 12-TET."""
    return A4_FREQ * (2.0 ** ((note - A4_MIDI) / 12.0))


def _stable_id(notes_abs: List[int]) -> int:
    """Genera un identificador estable de 64 bits a partir de la secuencia absoluta."""
    digest = hashlib.blake2b(
        json.dumps(notes_abs).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")


def _actual_mask(notes_abs: List[int]) -> str:
    """Calcula la máscara absoluta (respecto a MIDI real) y la devuelve como hexadecimal."""
    mask = 0
    for note in notes_abs:
        mask |= 1 << int(note)
    return format(mask, "x")


def _process_chord_record(notes_abs: List[int], tag: str) -> Dict[str, Any]:
    """Procesa una lista de notas MIDI y devuelve un diccionario con los datos del acorde."""
    root_midi = notes_abs[0]
    normalized = [note - root_midi for note in notes_abs]

    record = _build_record_from_notes(normalized, tag=tag)
    # Guardar la vista normalizada (anclada) para la UI opcional
    record['__norm_interval'] = record.get('interval')
    record['__norm_notes'] = record.get('notes')
    record['__norm_code'] = record.get('code')
    record['__norm_bass'] = record.get('bass')

    # Reemplazar por vista REAL (no anclada): PCs y diferencias en MIDI
    record['notes_abs_json'] = json.dumps(notes_abs)
    record['octave'] = (root_midi // 12) - 1
    record['frequencies'] = [_midi_to_freq(note) for note in notes_abs]
    record['span_semitones'] = notes_abs[-1] - notes_abs[0]
    # PCs reales y bajo real
    pcs_real = [str(n % 12) for n in notes_abs]
    intervals_real = [int(notes_abs[i+1] - notes_abs[i]) for i in range(len(notes_abs)-1)]
    record['notes'] = pcs_real
    record['bass'] = str(root_midi % 12)
    # code real (0123456789AB)
    HEX12 = "0123456789AB"
    record['code'] = ''.join(HEX12[int(n) % 12] for n in notes_abs)
    record['interval'] = intervals_real
    record['__source__'] = "GENERATED:COMBINATORIAL"
    record['__transposition__'] = 0
    record['__root_midi'] = root_midi
    record['abs_mask_midi'] = _actual_mask(notes_abs)
    record['id'] = _stable_id(notes_abs)
    return record


# Columnas esperadas por el pipeline principal
EXPECTED_COLUMNS = [
    'id', 'n', 'interval', 'notes', 'code', 'bass', 'octave',
    'frequencies', 'chroma', 'tag',
    'span_semitones', 'abs_mask_int', 'abs_mask_hex', 'notes_abs_json',
    'source_id', 'rotation', 'family_id', 'family_size',
    '__source__', '__transposition__', '__root_midi', 'abs_mask_midi',
    '__norm_interval', '__norm_notes', '__norm_code', '__norm_bass'
]


def generate_combinatorial_chords(
    alphabet: List[int],
    octave_min: int,
    octave_max: int,
    cardinalities: List[int],
    structural_mode: bool = False,
) -> pd.DataFrame:
    """
    Genera un DataFrame de acordes usando un enfoque combinatorio.

    Args:
        alphabet: Lista de pitch classes (0-11) a usar.
        octave_min: Octava MIDI inicial (e.g., 3 para C3).
        octave_max: Octava MIDI final.
        cardinalities: Lista de tamaños de acorde a generar (e.g., [3, 4]).
        structural_mode: Si es True, la primera nota de cada acorde será DO.

    Returns:
        Un DataFrame de pandas con los acordes generados.
    """
    if not alphabet or not cardinalities:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    midi_universe: List[int] = []
    for octave in range(octave_min, octave_max + 1):
        for pc in alphabet:
            midi_note = 12 * (octave + 1) + pc
            if 0 <= midi_note <= 127:
                midi_universe.append(midi_note)

    midi_universe = sorted(set(midi_universe))
    if alphabet:
        boundary_pc = min(int(pc) for pc in alphabet)
        boundary_octave = octave_max + 1
        boundary_note = 12 * (boundary_octave + 1) + boundary_pc
        if 0 <= boundary_note <= 127 and boundary_note not in midi_universe:
            midi_universe.append(boundary_note)
            midi_universe.sort()
    all_chords_records: List[Dict[str, Any]] = []

    for k in cardinalities:
        if k <= 0 or k > len(midi_universe):
            continue

        if structural_mode:
            c_zero = 12 * (octave_min + 1)
            if c_zero not in midi_universe:
                continue

            other_notes = [note for note in midi_universe if note != c_zero]
            if k - 1 > len(other_notes):
                continue

            for chord_midi_tuple in itertools.combinations(other_notes, k - 1):
                notes_abs = sorted([c_zero] + list(chord_midi_tuple))
                record = _process_chord_record(notes_abs, "combinatorial_structural")
                all_chords_records.append(record)
        else:
            for chord_midi_tuple in itertools.combinations(midi_universe, k):
                notes_abs = list(chord_midi_tuple)
                record = _process_chord_record(notes_abs, "combinatorial")
                all_chords_records.append(record)

    if not all_chords_records:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = pd.DataFrame.from_records(all_chords_records)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df[EXPECTED_COLUMNS]


if __name__ == "__main__":
    print("Ejecutando prueba de validación del generador combinatorio...")
    test_alphabet = [0, 4, 7]
    test_df = generate_combinatorial_chords(test_alphabet, 4, 4, [3])
    print(test_df.head())
    print("Columnas:", test_df.columns.tolist())
