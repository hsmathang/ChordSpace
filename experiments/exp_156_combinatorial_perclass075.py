"""
exp_156_combinatorial_perclass075.py
====================================
Réplica del ejercicio de 156 acordes estructurales generados de forma combinatoria
(n=3, span<=12, octavas=3-4) aplicando la propuesta 'perclass_alpha0_75' y MDS.
Incluye nombres de la "nueva simbología" (ej. May, Min, Maj7).
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_utils import build_entries, run_experiment
from services.combinatorial_generator import generate_combinatorial_chords
from services.population_filter import filter_dataframe
from tools.data_access import ChordFilters

OUTPUT_DIR = Path(__file__).parent / "outputs" / "156_combinatoria_perclass075"

def main():
    print("[INFO] Generando acordes estructurales combinatorios...")
    # Genera todas las combinaciones estructurales posibles
    df = generate_combinatorial_chords(
        alphabet=list(range(12)),
        octave_min=3,
        octave_max=4,
        cardinalities=[2, 3],
        structural_mode=True
    )
    
    print(f"[INFO] Total formas estructurales brutas: {len(df)}")
    
    # Aplicar filtros idénticos a los del reporte original:
    # Cardinalidades: 3
    # Intervalo interno máximo: 12
    # Modo de intervalos: exact (por defecto en el filtro vacío de patrones)
    # Modo pitch classes: contains_all ([]) 
    filters = ChordFilters(
        cardinalities=[2, 3],
        max_internal_interval=12,
        interval_mode="exact",
        include_pitch_classes=[],
    )
    
    df_filtered = filter_dataframe(df, filters)
    print(f"[INFO] Filtro aplicado -> {len(df_filtered)} acordes")
    
    # Reconstruimos chords_raw (name, intervals, freqs, notes_abs)
    # y anclamos en C3 (48) para consistencia estructural.
    BASE_MIDI = 48
    chords_raw = []
    
    for _, row in df_filtered.iterrows():
        semitones = row['__struct_semitones']
        notes_abs = [BASE_MIDI + s for s in semitones]
        intervals = [notes_abs[i+1] - notes_abs[i] for i in range(len(notes_abs)-1)]
        freqs = [440.0 * 2**((m - 69) / 12.0) for m in notes_abs]
        
        # El nombre genérico será reemplazado por build_entries internamente 
        # con "get_chord_type_from_intervals", que inyectará la "nueva simbología".
        struct_name = f"Struct_{row['__structure_id']}"
        chords_raw.append((struct_name, intervals, freqs, notes_abs))
        
    print("[INFO] Procesando vectores (proposal='perclass_alpha0_75')...")
    entries, X = build_entries(chords_raw, proposal="perclass_alpha0_75")
    
    # Preparamos las facetas basadas en la nueva simbología para mejorar la visualización
    facets = []
    symbols = []
    opacities = []
    
    # Las tríadas básicas estructurales que sabemos que existen son de estas categorías:
    # "Maj", "Min", "Dim", "Aug", "sus2", "sus4"
    # Sin embargo, `get_chord_type_from_intervals` devuelve a veces alias combinados.
    # Vamos a usar su patrón de intervalos para identificarlas con 100% de precisión 
    # estructural, independientemente del nombre.
    
    estructuras_triadas_basicas = {
        (4, 3): "Maj",
        (3, 4): "Min",
        (3, 3): "Dim",
        (4, 4): "Aug",
        (2, 5): "sus2",
        (5, 2): "sus4" 
    }
    
    diadas_count = 0
    triadas_count = 0
    tres_notas_count = 0
    
    # First pass to count
    for e in entries:
        if e.n_notes == 2:
            diadas_count += 1
        elif e.n_notes == 3:
            # Check if intervals correspond to basic triads
            intervals = tuple(e.acorde.intervals)
            if intervals in estructuras_triadas_basicas:
                triadas_count += 1
            else:
                tres_notas_count += 1
            
            
    for e in entries:
        if e.n_notes == 2:
            facets.append(f"Díadas ({diadas_count})")
            symbols.append("diamond-open")
            opacities.append(1.0)
        elif e.n_notes == 3:
            intervals = tuple(e.acorde.intervals)
            if intervals in estructuras_triadas_basicas:
                facets.append(f"Tríadas ({triadas_count})")
                symbols.append("triangle-down-open")
                opacities.append(1.0)
            else:
                facets.append(f"3 notas ({tres_notas_count})")
                symbols.append("circle")
                opacities.append(0.5)

    print("[INFO] Ejecutando experimento MDS...")
    run_experiment(
        entries=entries,
        X=X,
        metric="euclidean",
        output_dir=OUTPUT_DIR,
        experiment_name="combinatoria_perclass",
        scatter_title="Acordes Estructurales | perclass_075 | Nueva Simbología",
        scatter_mode="faceted",
        facet_labels=facets,
        facet_symbols=symbols,
        facet_opacities=opacities,
        facet_layout="single_overview",
        label_font_size=8,
        show_labels=False,
    )
    
    print(f"\n[ÉXITO] Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
