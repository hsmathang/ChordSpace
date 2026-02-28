"""
exp_mds_triadas_diatonicas.py
==============================
Experimento: 21 acordes diatónicos de Do Mayor
  7 tríadas básicas × 3 posiciones (fundamental + 2 inversiones)

Usa exp_utils.py → pipeline canónico con métricas completas + figuras del report.html.

Uso:
    .venv\\Scripts\\python.exe experiments\\triad_consonance\\exp_mds_triadas_diatonicas.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_utils import chords_from_midi, build_entries, run_experiment

OUTPUT_DIR = Path(__file__).parent / "outputs" / "21_triadas_diatonicas"

# ── Los 21 acordes diatónicos de Do Mayor ───────────────────────────────────
# Grados I-VII con sus 2 inversiones = 21 acordes
# MIDI: C4=60, D4=62, E4=64, F4=65, G4=67, A4=69, B4=71

TRIADAS_BASE = [
    ("I-CMaj",   [60, 64, 67]),   # Do Mayor
    ("II-DMin",  [62, 65, 69]),   # Re Menor
    ("III-EMin", [64, 67, 71]),   # Mi Menor
    ("IV-FMaj",  [65, 69, 72]),   # Fa Mayor
    ("V-GMaj",   [67, 71, 74]),   # Sol Mayor
    ("VI-AMin",  [69, 72, 76]),   # La Menor
    ("VII-BDim", [71, 74, 77]),   # Si Disminuida
]

DIADAS_BASE = [
    ("m2", [60, 61]),
    ("M2", [60, 62]),
    ("m3", [60, 63]),
    ("M3", [60, 64]),
    ("P4", [60, 65]),
    ("Tri", [60, 66]),
    ("P5", [60, 67]),
    ("m6", [60, 68]),
    ("M6", [60, 69]),
    ("m7", [60, 70]),
    ("M7", [60, 71]),
    ("P8", [60, 72]),
]


def inversiones(nombre, notas):
    n0 = sorted(notas)
    n1 = sorted(notas[1:] + [notas[0] + 12])
    n2 = sorted(notas[2:] + [notas[0] + 12, notas[1] + 12])
    return [
        (f"{nombre}-fund", n0),
        (f"{nombre}-inv1", n1),
        (f"{nombre}-inv2", n2),
    ]


todos_midi = []
for nombre, notas in TRIADAS_BASE:
    todos_midi.extend(inversiones(nombre, notas))
for nombre, notas in DIADAS_BASE:
    todos_midi.append((f"Dyad-{nombre}", notas))

# Convertir a (name, intervals, freqs, notes_abs) con nombres descriptivos
chords_raw = []
for nombre, notas in todos_midi:
    notas = sorted(notas)
    intervals = [notas[i+1] - notas[i] for i in range(len(notas)-1)]
    freqs = [440.0 * 2**((m - 69) / 12.0) for m in notas]
    chords_raw.append((nombre, intervals, freqs, notas))

print(f"[INFO] Población: {len(chords_raw)} acordes (7 tríadas × 3 posiciones)")

# ── Construir ChordEntry y vectores 12-D ─────────────────────────────────────
entries, X = build_entries(chords_raw, proposal="perclass_alpha0_75")

print(f"[INFO] X.shape={X.shape}")
for e in entries:
    print(f"  {e.identity_name:22s}  total_rug={e.total:.4f}")

# ── Ejecutar experimento completo (métricas + figuras) ───────────────────────
# Extraer el grado diatónico (I, II, III, etc.) del nombre (ej. "I-CMaj-fund" -> "I")
facets = [e.identity_name.split("-")[0] for e in entries]

metrics = run_experiment(
    entries=entries,
    X=X,
    metric="euclidean",
    output_dir=OUTPUT_DIR,
    experiment_name="21_triadas_y_12_diadas",
    n_init=8,
    scatter_mode="inset",
    zoom_window=[-1.15, -0.65, -0.75, -0.45], # Caja exacta del rincón inferior izquierdo
)

print(f"\n[DONE] → {OUTPUT_DIR}")
