import json
import sys
from pathlib import Path

BASE_MIDI = 48

def get_inversions_anchored(name, semitones, base_midi=48):
    variations = []
    n = len(semitones)
    for inv in range(n):
        if inv == 0:
            inv_name = f"{name}-fund"
            notas = [base_midi + s for s in semitones]
        else:
            inv_name = f"{name}-inv{inv}"
            shifted = sorted(semitones[inv:] + [s + 12 for s in semitones[:inv]])
            lowest = shifted[0]
            notas = [base_midi + (s - lowest) for s in shifted]
        variations.append((inv_name, notas))
    return variations

def main():
    chords_raw = []
    
    diadas_def = {
        "m2": [0, 1], "M2": [0, 2], "m3": [0, 3], "M3": [0, 4],
        "P4": [0, 5], "TT": [0, 6], "P5": [0, 7], "m6": [0, 8],
        "M6": [0, 9], "m7": [0, 10], "M7": [0, 11], "P8": [0, 12]
    }
    for name, semitones in diadas_def.items():
        # User requested: NO INVERSIONS for dyads
        invs = get_inversions_anchored(f"Diada_{name}", semitones, BASE_MIDI)
        # Solo tomamos la fundamental (el primer elemento que retorna)
        inv_name, notas = invs[0]
        intervals = [notas[i+1] - notas[i] for i in range(len(notas)-1)]
        freqs = [440.0 * 2**((m - 69) / 12.0) for m in notas]
        chords_raw.append((inv_name, intervals, freqs, notas))

    estructuras_triadas = {
        "Maj": [0, 4, 7], "Min": [0, 3, 7], "Dim": [0, 3, 6],
        "Aug": [0, 4, 8], "sus2": [0, 2, 7], "sus4": [0, 5, 7],
    }
    for name, semitones in estructuras_triadas.items():
        invs = get_inversions_anchored(name, semitones, BASE_MIDI)
        for inv_name, notas in invs:
            intervals = [notas[i+1] - notas[i] for i in range(len(notas)-1)]
            freqs = [440.0 * 2**((m - 69) / 12.0) for m in notas]
            chords_raw.append((inv_name, intervals, freqs, notas))

    estructuras_tetradas = {
        "Maj7": [0, 4, 7, 11], "Min7": [0, 3, 7, 10], "Dom7": [0, 4, 7, 10],
        "m7b5": [0, 3, 6, 10], "dim7": [0, 3, 6, 9], "mM7":  [0, 3, 7, 11],
    }
    for name, semitones in estructuras_tetradas.items():
        invs = get_inversions_anchored(name, semitones, BASE_MIDI)
        for inv_name, notas in invs:
            intervals = [notas[i+1] - notas[i] for i in range(len(notas)-1)]
            freqs = [440.0 * 2**((m - 69) / 12.0) for m in notas]
            chords_raw.append((inv_name, intervals, freqs, notas))

    estructuras_ext = {
        "Maj9": [0, 4, 7, 11, 14], "Dom9": [0, 4, 7, 10, 14], "Min9": [0, 3, 7, 10, 14],
    }
    for name, semitones in estructuras_ext.items():
        invs = get_inversions_anchored(name, semitones, BASE_MIDI)
        for inv_name, notas in invs:
            intervals = [notas[i+1] - notas[i] for i in range(len(notas)-1)]
            freqs = [440.0 * 2**((m - 69) / 12.0) for m in notas]
            chords_raw.append((inv_name, intervals, freqs, notas))

    out_path = Path(__file__).parent / "poblacion_extendida_c3.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, (name, intervals, freqs, notes_abs) in enumerate(chords_raw):
            row = {
                "id": idx + 1,
                "code": name,            
                "interval": intervals,
                "frequencies": freqs,
                "notes_abs_json": list(notes_abs),
                "__root_midi": notes_abs[0] if len(notes_abs) > 0 else 60
            }
            f.write(json.dumps(row) + "\n")
            
    print(f"[{len(chords_raw)} acordes exportados a {out_path}]")

if __name__ == "__main__":
    main()
