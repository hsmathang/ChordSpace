"""
exp_triadas_diatonicas_normal.py
================================
Las 21 tríadas diatónicas de Do Mayor (7 grados × 3 posiciones):
  I-CMaj, II-DMin, III-EMin, IV-FMaj, V-GMaj, VI-AMin, VII-BDim
  cada una en fundamental + 1ª inversión + 2ª inversión.

Simbología por cualidad estructural (todos sin relleno, solo contorno):
  - Maj  → triángulo apuntando arriba   (triangle-up-open)
  - Min  → triángulo apuntando abajo    (triangle-down-open)
  - Dim  → círculo                      (circle-open)
  - Aug  → cruz                         (cross-open)

Sin etiquetas de texto visibles; el nombre aparece solo en el hover.

Uso:
    set PYTHONIOENCODING=utf-8
    .venv\Scripts\python.exe experiments\exp_triadas_diatonicas_normal.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_utils import build_entries, run_experiment

OUTPUT_DIR = Path(__file__).parent / "outputs" / "triadas_diatonicas_normal"

TRIADAS_BASE = [
    ("I-CMaj",   [60, 64, 67]),   # Do Mayor
    ("II-DMin",  [62, 65, 69]),   # Re Menor
    ("III-EMin", [64, 67, 71]),   # Mi Menor
    ("IV-FMaj",  [65, 69, 72]),   # Fa Mayor
    ("V-GMaj",   [67, 71, 74]),   # Sol Mayor
    ("VI-AMin",  [69, 72, 76]),   # La Menor
    ("VII-BDim", [71, 74, 77]),   # Si Disminuida
]

# Mapa: grado → cualidad → símbolo y grupo
CUALIDAD_MAP = {
    "CMaj":  ("Maj",  "triangle-up-open"),
    "DMin":  ("Min",  "triangle-down-open"),
    "EMin":  ("Min",  "triangle-down-open"),
    "FMaj":  ("Maj",  "triangle-up-open"),
    "GMaj":  ("Maj",  "triangle-up-open"),
    "AMin":  ("Min",  "triangle-down-open"),
    "BDim":  ("Dim",  "circle-open"),
}


def inversiones(nombre, notas):
    n0 = sorted(notas)
    n1 = sorted(notas[1:] + [notas[0] + 12])
    n2 = sorted(notas[2:] + [notas[0] + 12, notas[1] + 12])
    return [
        (f"{nombre}-fund", n0),
        (f"{nombre}-inv1", n1),
        (f"{nombre}-inv2", n2),
    ]


# ── Construir la lista de acordes ─────────────────────────────────────────────
chords_raw = []
chord_meta = []   # (cualidad_label, símbolo) alineado con chords_raw

for nombre, notas in TRIADAS_BASE:
    grado_code = nombre.split("-")[1]                      # "CMaj", "DMin", etc.
    cualidad, simbolo = CUALIDAD_MAP.get(grado_code, ("Otro", "square-open"))
    for tag, ns in inversiones(nombre, notas):
        ns = sorted(ns)
        intervals = [ns[i + 1] - ns[i] for i in range(len(ns) - 1)]
        freqs = [440.0 * 2 ** ((m - 69) / 12.0) for m in ns]
        chords_raw.append((tag, intervals, freqs, ns))
        chord_meta.append((cualidad, simbolo))

print(f"[INFO] Población: {len(chords_raw)} acordes (7 tríadas × 3 posiciones)")

# ── Vectores 12-D ─────────────────────────────────────────────────────────────
entries, X = build_entries(chords_raw, proposal="perclass_alpha0_75")

# ── Asignar facetas y símbolos ────────────────────────────────────────────────
facets  = [meta[0] for meta in chord_meta]
symbols = [meta[1] for meta in chord_meta]

print(f"[INFO] Grupos: { {f for f in facets} }")

# ── Experimento MDS — faceted single_overview, sin etiquetas visibles ─────────
metrics = run_experiment(
    entries=entries,
    X=X,
    metric="euclidean",
    output_dir=OUTPUT_DIR,
    experiment_name="TriadasDiatonicasC_Mayor",
    scatter_title="21 Tríadas Diatónicas de Do Mayor | perclass_alpha0_75",
    n_init=8,
    scatter_mode="faceted",
    facet_labels=facets,
    facet_symbols=symbols,
    facet_layout="single_overview",
    show_labels=False,       # nombres solo en hover, no encima del punto
    label_font_size=9,
    paper_quality=True,
)

print(f"\n[DONE] Resultados en: {OUTPUT_DIR}")
