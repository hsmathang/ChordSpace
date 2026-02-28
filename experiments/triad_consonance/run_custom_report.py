import os
import sys
import pandas as pd
from pathlib import Path

# Fix sys.path to allow imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_proposals import main as compare_main
from tools.compare_proposals import parse_args

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

def build_custom_dataframe():
    chords_raw = []
    
    # ── 1. Las 12 Díadas (1 a 12 semitonos) ancladas a C3
    diadas_def = {
        "m2": [0, 1], "M2": [0, 2], "m3": [0, 3], "M3": [0, 4],
        "P4": [0, 5], "TT": [0, 6], "P5": [0, 7], "m6": [0, 8],
        "M6": [0, 9], "m7": [0, 10], "M7": [0, 11], "P8": [0, 12]
    }
    for name, semitones in diadas_def.items():
        invs = get_inversions_anchored(f"Diada_{name}", semitones, BASE_MIDI)
        for inv_name, notas in invs:
            intervals = [notas[i+1] - notas[i] for i in range(len(notas)-1)]
            freqs = [440.0 * 2**((m - 69) / 12.0) for m in notas]
            chords_raw.append((inv_name, intervals, freqs, notas))

    # ── 2. Tríadas (Básicas + Sus)
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

    # ── 3. Tétradas (Acordes de 7ma)
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

    # ── 4. Extensiones Jazzísticas comunes (9nas)
    estructuras_ext = {
        "Maj9": [0, 4, 7, 11, 14], "Dom9": [0, 4, 7, 10, 14], "Min9": [0, 3, 7, 10, 14],
    }
    for name, semitones in estructuras_ext.items():
        invs = get_inversions_anchored(name, semitones, BASE_MIDI)
        for inv_name, notas in invs:
            intervals = [notas[i+1] - notas[i] for i in range(len(notas)-1)]
            freqs = [440.0 * 2**((m - 69) / 12.0) for m in notas]
            chords_raw.append((inv_name, intervals, freqs, notas))

    rows = []
    for idx, (name, intervals, freqs, notes_abs) in enumerate(chords_raw):
        rows.append({
            "id": idx + 1,
            "code": name,            
            "interval": intervals,
            "frequencies": freqs,
            "notes_abs_json": list(notes_abs),
            "__root_midi": notes_abs[0] if len(notes_abs) > 0 else 60
        })
    return pd.DataFrame(rows)

def run():
    print("[INFO] Generando poblacion sintetica en DataFrame...")
    df = build_custom_dataframe()
    print(f"[INFO] Construidos {len(df)} acordes extendidos anclados a C3.")
    
    json_path = ROOT / "experiments" / "triad_consonance" / "poblacion_extendida_c3.jsonl"
    df.to_json(json_path, orient="records", lines=True)
    print(f"[INFO] Escrito correctamente a {json_path}")
    
    output_dir = ROOT / "experiments" / "triad_consonance" / "outputs" / "reporte_extendido_c3"
    
    print("[INFO] Lanzando compare_proposals API pipeline...")
    
    # Simulate the CLI arguments
    sys.argv = [
        "compare_proposals.py",
        "--population-json", str(json_path),
        "--proposals", "perclass_alpha0_75",
        "--metrics", "euclidean",
        "--output", str(output_dir)
    ]
    
    try:
        compare_main()
        print(f"\n[ÉXITO] Reporte HTML generado inyectando la población. Revisa la carpeta: {output_dir}")
    except SystemExit as exc:
        print(f"[EXIT] Proceso finalizado: {exc}")
    except Exception as exc:
        print(f"[ERROR] Falló la ejecución del pipeline: {exc}")

if __name__ == "__main__":
    run()
