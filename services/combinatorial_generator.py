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


# Columnas esperadas por el pipeline principal
EXPECTED_COLUMNS = [
    'id', 'n', 'interval', 'notes', 'code', 'bass', 'octave', 'tag',
    'span_semitones', 'abs_mask_int', 'abs_mask_hex', 'notes_abs_json',
    'source_id', 'rotation', 'family_id', 'family_size',
    '__source__', '__transposition__', '__root_midi', 'abs_mask_midi'
]


def generate_combinatorial_chords(
    alphabet: List[int],
    octave_min: int,
    octave_max: int,
    cardinalities: List[int],
) -> pd.DataFrame:
    """
    Genera un DataFrame de acordes usando un enfoque combinatorio.

    Args:
        alphabet: Lista de pitch classes (0-11) a usar.
        octave_min: Octava MIDI inicial (e.g., 3 para C3).
        octave_max: Octava MIDI final.
        cardinalities: Lista de tamaños de acorde a generar (e.g., [3, 4]).

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
    all_chords_records: List[Dict[str, Any]] = []

    for k in cardinalities:
        if k <= 0 or k > len(midi_universe):
            continue

        for chord_midi_tuple in itertools.combinations(midi_universe, k):
            notes_abs = list(chord_midi_tuple)
            root_midi = notes_abs[0]
            normalized = [note - root_midi for note in notes_abs]

            record = _build_record_from_notes(normalized, tag="combinatorial")
            record['notes_abs_json'] = json.dumps(notes_abs)
            record['octave'] = (root_midi // 12) - 1
            record['frequencies'] = [_midi_to_freq(note) for note in notes_abs]
            record['span_semitones'] = notes_abs[-1] - notes_abs[0]
            record['__source__'] = "GENERATED:COMBINATORIAL"
            record['__transposition__'] = 0
            record['__root_midi'] = root_midi
            record['abs_mask_midi'] = _actual_mask(notes_abs)
            record['id'] = _stable_id(notes_abs)

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
