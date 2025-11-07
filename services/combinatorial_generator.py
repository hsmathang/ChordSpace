"""
Servicio para la generación combinatoria de acordes.
"""
from __future__ import annotations

import itertools
from typing import List, Dict, Any

import pandas as pd
import numpy as np

# Reutilizamos la lógica de construcción de registros para asegurar la compatibilidad
from synth_tools import _build_record_from_notes

# Columnas esperadas por el pipeline principal
EXPECTED_COLUMNS = [
    'id', 'n', 'interval', 'notes', 'code', 'bass', 'octave', 'tag',
    'span_semitones', 'abs_mask_int', 'abs_mask_hex', 'notes_abs_json',
    'source_id', 'rotation', 'family_id', 'family_size',
    '__source__', '__transposition__'
]

def generate_combinatorial_chords(
    alphabet: List[int],
    octave_min: int,
    octave_max: int,
    cardinalities: List[int]
) -> pd.DataFrame:
    """
    Genera un DataFrame de acordes usando un enfoque combinatorio.

    Args:
        alphabet: Lista de pitch classes (0-11) a usar.
        octave_min: Octava MIDI inicial (e.g., 3 para C3).
        octave_max: Octava MIDI final.
        cardinalities: Lista de tamaños de acorde a generar (e.g., [3, 4] para tríadas y cuatríadas).

    Returns:
        Un DataFrame de pandas con los acordes generados, compatible con el pipeline.
    """
    if not alphabet or not cardinalities:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    # 1. Crear el universo de notas MIDI
    midi_universe = []
    for octave in range(octave_min, octave_max + 1):
        for pc in alphabet:
            # Convención MIDI: C4 = 60. Octava -1 en MIDI es la octava 0 en la mayoría de DAWs.
            # C0 = 12, C1 = 24, ..., C4 = 60
            midi_note = 12 * (octave + 1) + pc
            if 0 <= midi_note <= 127:
                midi_universe.append(midi_note)

    # Eliminar duplicados si el rango de octavas y el alfabeto los generan
    midi_universe = sorted(list(set(midi_universe)))

    all_chords_records: List[Dict[str, Any]] = []

    # 2. Generar combinaciones y construir registros
    for k in cardinalities:
        if k <= 0 or k > len(midi_universe):
            continue

        for chord_midi_tuple in itertools.combinations(midi_universe, k):
            notes_abs = list(chord_midi_tuple)

            # 3. Llamar a la función reutilizada para generar el registro
            record = _build_record_from_notes(notes_abs, tag="combinatorial")

            # 4. Añadir metadatos adicionales para el flujo de la UI
            record['__source__'] = "GENERATED:COMBINATORIAL"
            record['__transposition__'] = 0

            # abs_mask_int es un buen candidato para ID único y eficiente en memoria
            # si no hay colisiones, lo cual es garantizado por la definición de la máscara.
            record['id'] = record['abs_mask_int']

            all_chords_records.append(record)

    if not all_chords_records:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    # 5. Crear el DataFrame final
    df = pd.DataFrame.from_records(all_chords_records)

    # Asegurar que todas las columnas esperadas existan, rellenando con Nulos si es necesario
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df[EXPECTED_COLUMNS]


if __name__ == '__main__':
    import json
    # --- Prueba de validación rápida ---
    print("Ejecutando prueba de validación del generador combinatorio...")

    # Parámetros de prueba: Tríada de Do Mayor (C, E, G) en la 4ª octava, generar díadas.
    test_alphabet = [0, 4, 7]  # C, E, G
    test_octave_min = 4
    test_octave_max = 4
    test_cardinalities = [2]  # Díadas

    # Universo MIDI esperado: C4=60, E4=64, G4=67
    # Combinaciones esperadas (díadas): (60, 64), (60, 67), (64, 67) -> 3 acordes

    generated_df = generate_combinatorial_chords(
        alphabet=test_alphabet,
        octave_min=test_octave_min,
        octave_max=test_octave_max,
        cardinalities=test_cardinalities
    )

    print(f"\nDataFrame generado ({len(generated_df)} filas):\n")
    print(generated_df.head())
    print("\nColumnas del DataFrame:")
    print(generated_df.columns.tolist())

    # --- Verificaciones ---
    print("\n--- Verificaciones ---")
    expected_rows = 3
    print(f"1. Número de filas: {'OK' if len(generated_df) == expected_rows else 'FALLO'} (Esperado: {expected_rows}, Obtenido: {len(generated_df)})")

    all_columns_ok = all(col in generated_df.columns for col in EXPECTED_COLUMNS)
    print(f"2. Todas las columnas esperadas están presentes: {'OK' if all_columns_ok else 'FALLO'}")

    notes_abs_list = generated_df['notes_abs_json'].apply(lambda x: tuple(json.loads(x))).tolist()
    expected_notes = [(60, 64), (60, 67), (64, 67)]
    notes_ok = sorted(notes_abs_list) == sorted(expected_notes)
    print(f"3. Contenido de acordes (notes_abs_json): {'OK' if notes_ok else 'FALLO'}")
    if not notes_ok:
        print(f"   - Esperado: {sorted(expected_notes)}")
        print(f"   - Obtenido: {sorted(notes_abs_list)}")

    print("\nPrueba finalizada.")
