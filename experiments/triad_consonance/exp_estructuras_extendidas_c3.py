"""
exp_estructuras_extendidas_c3.py
==============================
Experimento aislando la RUGOSIDAD ESTRUCTURAL para una población extendida de acordes.
Compara díadas, todas las tríadas (básicas, sus2, sus4), tétradas comunes (Maj7, Min7, Dom7, etc.)
y algunas extensiones de jazz (9nas).
CRÍTICO: Todos los acordes (incluso las inversiones) están "anclados" a C3 (MIDI 48) como nota más grave.
Esto elimina la variable de registro/tesitura, permitiendo comparar la estructura pura.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_utils import build_entries, run_experiment

OUTPUT_DIR = Path(__file__).parent / "outputs" / "estructuras_extendidas_c3"

BASE_MIDI = 48 # C3

def get_inversions_anchored(name, semitones, base_midi=48):
    """
    Genera la fundamental y todas las inversiones de una estructura interválica,
    asegurando que SIEMPRE la nota más grave sea `base_midi`.
    """
    variations = []
    n = len(semitones)
    
    for inv in range(n):
        if inv == 0:
            inv_name = f"{name}-fund"
            notas = [base_midi + s for s in semitones]
        else:
            inv_name = f"{name}-inv{inv}"
            # Subir las primeras 'inv' notas una octava y ORDENAR para evitar intervalos negativos
            shifted = sorted(semitones[inv:] + [s + 12 for s in semitones[:inv]])
            # Transponer todo para que la nota más grave coincida con base_midi
            lowest = shifted[0]
            notas = [base_midi + (s - lowest) for s in shifted]
        variations.append((inv_name, notas))
    return variations

chords_raw = []

# ── 1. Las 12 Díadas (1 a 12 semitonos) ancladas a C3 ────────────────────────
diadas_def = {
    "m2": [0, 1], "M2": [0, 2], "m3": [0, 3], "M3": [0, 4],
    "P4": [0, 5], "TT": [0, 6], "P5": [0, 7], "m6": [0, 8],
    "M6": [0, 9], "m7": [0, 10], "M7": [0, 11], "P8": [0, 12]
}
for name, semitones in diadas_def.items():
    invs = get_inversions_anchored(f"Diada_{name}", semitones, BASE_MIDI)
    # Solo tomamos la fundamental (el primer elemento) para díadas
    inv_name, notas = invs[0]
    intervals = [notas[i+1] - notas[i] for i in range(len(notas)-1)]
    freqs = [440.0 * 2**((m - 69) / 12.0) for m in notas]
    chords_raw.append((inv_name, intervals, freqs, notas))

# ── 2. Tríadas (Básicas + Sus) ───────────────────────────────────────────────
estructuras_triadas = {
    "Maj": [0, 4, 7],
    "Min": [0, 3, 7],
    "Dim": [0, 3, 6],
    "Aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}
for name, semitones in estructuras_triadas.items():
    invs = get_inversions_anchored(name, semitones, BASE_MIDI)
    for inv_name, notas in invs:
        intervals = [notas[i+1] - notas[i] for i in range(len(notas)-1)]
        freqs = [440.0 * 2**((m - 69) / 12.0) for m in notas]
        chords_raw.append((inv_name, intervals, freqs, notas))

# ── 3. Tétradas (Acordes de 7ma) ─────────────────────────────────────────────
estructuras_tetradas = {
    "Maj7": [0, 4, 7, 11],
    "Min7": [0, 3, 7, 10],
    "Dom7": [0, 4, 7, 10],
    "m7b5": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
    "mM7":  [0, 3, 7, 11],
}
for name, semitones in estructuras_tetradas.items():
    invs = get_inversions_anchored(name, semitones, BASE_MIDI)
    for inv_name, notas in invs:
        intervals = [notas[i+1] - notas[i] for i in range(len(notas)-1)]
        freqs = [440.0 * 2**((m - 69) / 12.0) for m in notas]
        chords_raw.append((inv_name, intervals, freqs, notas))

# ── 4. Extensiones Jazzísticas comunes (9nas) ────────────────────────────────
estructuras_ext = {
    "Maj9": [0, 4, 7, 11, 14],
    "Dom9": [0, 4, 7, 10, 14],
    "Min9": [0, 3, 7, 10, 14],
}
for name, semitones in estructuras_ext.items():
    invs = get_inversions_anchored(name, semitones, BASE_MIDI)
    for inv_name, notas in invs:
        intervals = [notas[i+1] - notas[i] for i in range(len(notas)-1)]
        freqs = [440.0 * 2**((m - 69) / 12.0) for m in notas]
        chords_raw.append((inv_name, intervals, freqs, notas))

print(f"[INFO] Población: {len(chords_raw)} estructuras en total (Díadas, Tríadas, Tétradas, 9nas)")

entries, X = build_entries(chords_raw)

# ── 5. Preparar Facetas y Símbolos ───────────────────────────────────────────
facets = []
symbols = []
for e in entries:
    nombre = e.identity_name # ej: "Maj7-inv2"
    tipo = nombre.split("-")[0]
    
    # Asignación de categorías (para la leyenda)
    if "Diada" in tipo:
        facets.append("Díadas")
        symbols.append("diamond-open")
    elif tipo == "Maj":
        facets.append("Maj")
        symbols.append("triangle-up-open")
    elif tipo == "Min":
        facets.append("Min")
        symbols.append("triangle-down-open")
    elif tipo == "Dim":
        facets.append("Dim")
        symbols.append("circle-open")
    elif tipo == "Aug":
        facets.append("Aug")
        symbols.append("cross-open")
    else:
        facets.append("Otros Acordes")
        symbols.append("circle")  # Punto CON relleno

# ── 6. Ejecutar experimento (SOLO VISTA GENERAL GRANDE) ──────────────────────
metrics_single = run_experiment(
    entries=entries,
    X=X,
    metric="euclidean",
    output_dir=OUTPUT_DIR,
    experiment_name="Acordes_Extendidos_C3",
    scatter_title="Estructuras Extendidas Ancladas en C3 (Díadas a 9nas)",
    n_init=8,
    scatter_mode="faceted",
    facet_labels=facets,
    facet_symbols=symbols,
    facet_layout="single_overview",
    label_font_size=9,
    show_labels=False,
)

print(f"\n[DONE] → {OUTPUT_DIR}")
