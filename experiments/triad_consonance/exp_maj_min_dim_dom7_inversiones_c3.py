"""
exp_maj_min_dim_dom7_inversiones_c3.py
======================================
Experimento enfocado en:
  - Triadas Maj, Min, Dim (todas sus inversiones)
  - Tetrada Dom7 (todas sus inversiones)

Todo anclado a C3 (MIDI 48) como nota mas grave, usando vectores 12-D
con la propuesta `perclass_alpha0_75`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_utils import build_entries, run_experiment

OUTPUT_DIR = Path(__file__).parent / "outputs" / "maj_min_dim_dom7_c3"
BASE_MIDI = 48  # C3


def midi_to_freq(midi_note: int) -> float:
    return 440.0 * 2 ** ((midi_note - 69) / 12.0)


def get_inversions_anchored(name: str, semitones: list[int], base_midi: int = 48):
    """
    Genera fundamental + inversiones, reanclando siempre la nota mas grave a base_midi.
    """
    variations = []
    n = len(semitones)
    for inv in range(n):
        if inv == 0:
            inv_name = f"{name}-fund"
            notes_abs = [base_midi + s for s in semitones]
        else:
            inv_name = f"{name}-inv{inv}"
            shifted = sorted(semitones[inv:] + [s + 12 for s in semitones[:inv]])
            lowest = shifted[0]
            notes_abs = [base_midi + (s - lowest) for s in shifted]
        variations.append((inv_name, notes_abs))
    return variations


def build_population(base_midi: int = BASE_MIDI):
    chords_raw = []

    triad_structures = {
        "Maj": [0, 4, 7],
        "Min": [0, 3, 7],
        "Dim": [0, 3, 6],
    }
    tetrad_structures = {
        "Dom7": [0, 4, 7, 10],
    }

    for name, semitones in triad_structures.items():
        for inv_name, notes_abs in get_inversions_anchored(name, semitones, base_midi):
            intervals = [notes_abs[i + 1] - notes_abs[i] for i in range(len(notes_abs) - 1)]
            freqs = [midi_to_freq(m) for m in notes_abs]
            chords_raw.append((inv_name, intervals, freqs, notes_abs))

    for name, semitones in tetrad_structures.items():
        for inv_name, notes_abs in get_inversions_anchored(name, semitones, base_midi):
            intervals = [notes_abs[i + 1] - notes_abs[i] for i in range(len(notes_abs) - 1)]
            freqs = [midi_to_freq(m) for m in notes_abs]
            chords_raw.append((inv_name, intervals, freqs, notes_abs))

    return chords_raw


def main():
    chords_raw = build_population()
    print(f"[INFO] Poblacion: {len(chords_raw)} acordes (Maj/Min/Dim + Dom7 con inversiones)")

    entries, X = build_entries(chords_raw, proposal="perclass_alpha0_75")

    facets = []
    symbols = []
    for e in entries:
        chord_type = e.identity_name.split("-")[0]
        if chord_type == "Maj":
            facets.append("Maj")
            symbols.append("triangle-up-open")
        elif chord_type == "Min":
            facets.append("Min")
            symbols.append("triangle-down-open")
        elif chord_type == "Dim":
            facets.append("Dim")
            symbols.append("circle-open")
        elif chord_type == "Dom7":
            facets.append("Dom7")
            symbols.append("square-open")
        else:
            facets.append(chord_type)
            symbols.append("circle")

    run_experiment(
        entries=entries,
        X=X,
        metric="euclidean",
        output_dir=OUTPUT_DIR,
        experiment_name="Maj_Min_Dim_Dom7_C3",
        scatter_title="Maj/Min/Dim + Dom7 (inversiones ancladas en C3)",
        n_init=8,
        scatter_mode="faceted",
        facet_labels=facets,
        facet_symbols=symbols,
        facet_layout="single_overview",
        label_font_size=9,
        show_labels=False,
    )

    print(f"[DONE] Resultados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
