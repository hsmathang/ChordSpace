"""
exp_estructuras_basicas_c3.py
==============================
Experimento aislando la RUGOSIDAD ESTRUCTURAL.
Compara las 12 díadas (hasta la octava) y las 4 tríadas básicas (May, Min, Dim, Aum)
en todas sus inversiones. 
CRÍTICO: Todos los acordes (incluso las inversiones) están "anclados" a C3 (MIDI 48) como nota más grave.
Esto elimina la variable de registro/tesitura, permitiendo comparar la estructura pura.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_utils import build_entries, run_experiment

OUTPUT_DIR = Path(__file__).parent / "outputs" / "estructuras_basicas_c3"

BASE_MIDI = 48 # C3

# ── 1. Las 12 Díadas (1 a 12 semitonos) ancladas a C3 ────────────────────────
diadas_raw = []
nombres_int = ["m2", "M2", "m3", "M3", "P4", "TT", "P5", "m6", "M6", "m7", "M7", "P8"]

for i in range(1, 13):
    notas = [BASE_MIDI, BASE_MIDI + i]
    nombre = f"Diada-{nombres_int[i-1]}"
    intervals = [i]
    freqs = [440.0 * 2**((m - 69) / 12.0) for m in notas]
    diadas_raw.append((nombre, intervals, freqs, notas))

# ── 2. Las 4 Tríadas Básicas y sus inversiones (Ancladas a C3) ──────────────
# Definimos la estructura de la tríada fundamental en semitonos desde la raíz
estructuras_triadas = {
    "Maj": [0, 4, 7],
    "Min": [0, 3, 7],
    "Dim": [0, 3, 6],
    "Aug": [0, 4, 8],
}

triadas_raw = []

for tipo, semitonos in estructuras_triadas.items():
    # Fundamental
    notas_fund = [BASE_MIDI + s for s in semitonos]
    
    # 1ra Inversión: subimos la raíz una octava, y luego transponemos TODO hacia abajo 
    # para que la nueva nota más grave vuelva a ser BASE_MIDI (C3)
    notas_inv1_tmp = [semitonos[1], semitonos[2], semitonos[0] + 12]
    dist_al_fondo_1 = notas_inv1_tmp[0]
    notas_inv1 = [BASE_MIDI + (n - dist_al_fondo_1) for n in notas_inv1_tmp]
    
    # 2da Inversión: subimos raíz y tercera una octava, transponemos al fondo
    notas_inv2_tmp = [semitonos[2], semitonos[0] + 12, semitonos[1] + 12]
    dist_al_fondo_2 = notas_inv2_tmp[0]
    notas_inv2 = [BASE_MIDI + (n - dist_al_fondo_2) for n in notas_inv2_tmp]
    
    variaciones = [
        (f"{tipo}-fund", notas_fund),
        (f"{tipo}-inv1", notas_inv1),
        (f"{tipo}-inv2", notas_inv2),
    ]
    
    for nombre, notas in variaciones:
        intervals = [notas[1] - notas[0], notas[2] - notas[1]]
        freqs = [440.0 * 2**((m - 69) / 12.0) for m in notas]
        triadas_raw.append((nombre, intervals, freqs, notas))

# ── 3. Unir todo ─────────────────────────────────────────────────────────────
chords_raw = diadas_raw + triadas_raw  # Reintegramos las díadas a petición
print(f"[INFO] Población: {len(chords_raw)} acordes (Díadas y Tríadas Básicas y sus inversiones)")

entries, X = build_entries(chords_raw, proposal="perclass_alpha0_75")

# ── 4. Ejecutar experimento ──────────────────────────────────────────────────
# Definir facetas (Categorías) y símbolos para el scatter plot
facets = []
symbols = []
for e in entries:
    nombre = e.identity_name
    # Extrae "Maj", "Min", "Dim", "Aug" de "Maj-fund"
    tipo = nombre.split("-")[0]
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
        facets.append(tipo)
        symbols.append("circle-open")

# 1. Versión General (Normal, sin facetas, sin etiquetas)
metrics_normal = run_experiment(
    entries=entries,
    X=X,
    metric="euclidean",
    output_dir=OUTPUT_DIR,
    experiment_name="Estructuras_Tríadas_General",
    scatter_title="Estructuras de Tríadas Básicas (Vista MDS Plana)",
    n_init=8,
    scatter_mode="normal",
    label_font_size=9,
    show_labels=False,
)

# 2. Versión Facetada (Vista General arriba + 2x2 abajo, sin etiquetas)
metrics_faceted = run_experiment(
    entries=entries,
    X=X,
    metric="euclidean",
    output_dir=OUTPUT_DIR,
    experiment_name="Estructuras_Tríadas_Facetas",
    scatter_title="Matriz de Estructuras Triádicas (Desglose por Cualidad)",
    n_init=8,
    scatter_mode="faceted",
    facet_labels=facets,
    facet_symbols=symbols,
    facet_layout="overview_top",
    label_font_size=9,
    show_labels=False,
)
# 3. Versión Solo Vista General pero conservando Leyenda y Símbolos
metrics_single = run_experiment(
    entries=entries,
    X=X,
    metric="euclidean",
    output_dir=OUTPUT_DIR,
    experiment_name="Estructuras_Tríadas_Simbolos_SinFondo",
    scatter_title="Vista General de Estructuras Triádicas Ancladas en C3",
    n_init=8,
    scatter_mode="faceted",
    facet_labels=facets,
    facet_symbols=symbols,
    facet_layout="single_overview",
    label_font_size=9,
    show_labels=False,
)

print(f"\n[DONE] → {OUTPUT_DIR}")
