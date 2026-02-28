"""
exp_diadas_triadas_octava.py
============================
Experimento: Todas las díadas y triadas dentro de una octava (C3–B3)
  Díadas : C(12,2) = 66  acordes
  Triadas: C(12,3) = 220 acordes
  Total  : 286 acordes

Métricas: Euclidiana  Y  Coseno

Usa exp_utils.py → pipeline canónico con métricas completas + figuras del report.html.

Uso:
    .venv\\Scripts\\python.exe experiments\\triad_consonance\\exp_diadas_triadas_octava.py
"""

import sys
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_utils import chords_from_midi, build_entries, run_experiment

OUTPUT_DIR = Path(__file__).parent / "outputs" / "diadas_triadas_octava"

# ── Generación combinatorial (C3–B3 = MIDI 48–59) ────────────────────────────
notas = list(range(48, 60))   # 12 notas de una octava

combos_midi = (
    list(combinations(notas, 2)) +   # 66 díadas
    list(combinations(notas, 3))     # 220 triadas
)

chords_raw = chords_from_midi(combos_midi)
n_diadas  = len(list(combinations(notas, 2)))
n_triadas = len(list(combinations(notas, 3)))
print(f"[INFO] Población: {n_diadas} díadas + {n_triadas} triadas = {len(chords_raw)} acordes")

# ── Construir ChordEntry y vectores 12-D ─────────────────────────────────────
entries, X = build_entries(chords_raw)
print(f"[INFO] X.shape={X.shape}")

# ── Ejecutar experimento para cada métrica ────────────────────────────────────
for metric in ("euclidean", "cosine"):
    run_experiment(
        entries=entries,
        X=X,
        metric=metric,
        output_dir=OUTPUT_DIR,
        experiment_name="diadas_triadas_octava",
        n_init=4,
    )

print(f"\n[DONE] → {OUTPUT_DIR}")
