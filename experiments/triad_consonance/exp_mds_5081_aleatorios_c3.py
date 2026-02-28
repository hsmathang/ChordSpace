"""
exp_mds_5081_aleatorios_c3.py
=============================
Experimento de escala masiva. Carga los 81 acordes originales y genera
5000 acordes VERDADERAMENTE aleatorios (cardinalidad 2–12, span 2 octavas)
anclados todos a MIDI 0 (C como raíz absoluta).

El orden de ploteo garantiza que los grupos más grandes se dibujen primero
(capas inferiores) y los más pequeños encima (capas superiores):
  1. Masa aleatoria (5000) → círculos a 50% opacidad → fondo
  2. Acordes extendidos originales (7as, 9as, etc.) → cuadrados abiertos
  3. Tríadas básicas (Maj/Min/Dim/Aug/sus) → triángulos abiertos → encima
  4. Díadas originales → rombos abiertos → encima de todo

Además la generación aleatoria es mucho más seria:
 - Cardinalidad varía aleatoriamente entre 2 y 12 de forma uniforme.
 - Notas internas escogidas con random.sample sobre 24 semitonos (2 octavas).
 - No hay ciclos forzados; la distribución es realmente uniforme sobre (card, estructura).
"""

from __future__ import annotations

import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_utils import build_entries, run_experiment
from experiments.triad_consonance.exp_umap_estructuras_extendidas_c3 import build_raw_from_jsonl

OUTPUT_DIR = Path(__file__).parent / "outputs" / "5081_aleatorios_c3_mds"
BASE_MIDI = 0   # Ancla absoluta en MIDI 0 (C más grave)

# Tríadas básicas identificadas por su par de intervalos internos exactos
TRIADAS_BASICAS = {(4, 3), (3, 4), (3, 3), (4, 4), (2, 5), (5, 2)}


def generate_random_chords(n_chords: int = 5000, max_semitones: int = 24) -> list:
    """
    Genera `n_chords` acordes aleatorios únicos anclados a BASE_MIDI.
    - Cardinalidad: uniforme entre 2 y 12 (sin ciclos forzados).
    - Span máximo: `max_semitones` semitonos sobre la raíz (2 octavas).
    - Cada acorde es una estructura única (conjunto de semitonos, no repetidas).
    """
    generated: set = set()
    chords_raw = []

    attempts = 0
    while len(generated) < n_chords:
        attempts += 1
        # Cardinalidad uniformemente distribuida entre 2 y 12
        card = random.randint(2, 12)
        if card - 1 > max_semitones:
            continue  # No hay suficientes semitonos disponibles para esta cardinalidad
        # Elegir (card-1) semitonos distintos en [1, max_semitones]
        others = random.sample(range(1, max_semitones + 1), card - 1)
        semitones = tuple(sorted([0] + others))

        if semitones not in generated:
            generated.add(semitones)
            nombre = f"Rnd_{len(generated)}"
            notas = [BASE_MIDI + s for s in semitones]
            intervals = [notas[i + 1] - notas[i] for i in range(len(notas) - 1)]
            freqs = [440.0 * 2 ** ((m - 69) / 12.0) for m in notas]
            chords_raw.append((nombre, intervals, freqs, notas))

        if attempts > n_chords * 100:
            print(f"[WARN] Límite de intentos alcanzado, generados {len(generated)} acordes únicos.")
            break

    return chords_raw


def classify_entry(e, identity_name: str) -> str:
    """Clasifica un ChordEntry en una de las 4 categorías para el plot."""
    if identity_name.startswith("Rnd_"):
        return "aleatorio"
    # Acordes originales: clasificar por estructura
    if e.n_notes == 2:
        return "diada"
    if e.n_notes == 3:
        intervals_tuple = tuple(e.acorde.intervals)
        if intervals_tuple in TRIADAS_BASICAS:
            return "triada"
        return "extendido"
    return "extendido"


def main():
    print("[INFO] Cargando los 81 acordes originales...")
    chords_raw_base = build_raw_from_jsonl()

    print(f"[INFO] Generando 5000 acordes aleatorios (card 2–12, span 2 oct, ancla MIDI={BASE_MIDI})...")
    chords_raw_random = generate_random_chords(n_chords=5000, max_semitones=24)

    # Combinar: ALEATORIOS primero (se plotearán abajo), ORIGINALES después
    chords_raw_total = chords_raw_random + chords_raw_base
    print(f"[INFO] Población total: {len(chords_raw_total)} acordes.")

    print("[INFO] Calculando vectores 12-D con 'perclass_alpha0_75'...")
    entries, X = build_entries(chords_raw_total, proposal="perclass_alpha0_75")

    # ─── Clasificación y asignación de facetas ───────────────────────────────
    # Orden deseado en el plot (de fondo a frente):
    #   1. Masa aleatoria
    #   2. Extendidos originales
    #   3. Tríadas básicas
    #   4. Díadas
    # Como Plotly renderiza en el orden en que se añaden los traces,
    # primero dejamos los aleatorios, luego el resto.
    # Las entries ya están en ese orden (randoms primero).

    # Conteos para leyenda
    counts = {"aleatorio": 0, "extendido": 0, "triada": 0, "diada": 0}
    categories = []
    for e in entries:
        cat = classify_entry(e, e.identity_name)
        categories.append(cat)
        counts[cat] += 1

    print(f"[INFO] Distribución → {counts}")

    # Construir listas de facetas/símbolos/opacidades en el MISMO orden que entries
    GRUPO_LABELS = {
        "aleatorio":  f"Masa aleatoria ({counts['aleatorio']})",
        "extendido":  f"Extendidos originales ({counts['extendido']})",
        "triada":     f"Tríadas básicas ({counts['triada']})",
        "diada":      f"Díadas ({counts['diada']})",
    }
    GRUPO_SYMBOLS = {
        "aleatorio": "circle",
        "extendido": "square-open",
        "triada":    "triangle-down-open",
        "diada":     "diamond-open",
    }
    GRUPO_OPACITY = {
        "aleatorio": 0.45,
        "extendido": 1.0,
        "triada":    1.0,
        "diada":     1.0,
    }

    facets    = [GRUPO_LABELS[c]   for c in categories]
    symbols   = [GRUPO_SYMBOLS[c]  for c in categories]
    opacities = [GRUPO_OPACITY[c]  for c in categories]

    print(f"[INFO] Ejecutando MDS sobre {len(chords_raw_total)} puntos (puede tardar)...")
    run_experiment(
        entries=entries,
        X=X,
        metric="euclidean",
        output_dir=OUTPUT_DIR,
        experiment_name="Acordes_Masivos_5081_MDS",
        scatter_title="5081 Acordes — Masa Aleatoria + Originales (2 octavas, card 2–12)",
        n_init=1,
        scatter_mode="faceted",
        facet_labels=facets,
        facet_symbols=symbols,
        facet_opacities=opacities,
        facet_layout="single_overview",
        label_font_size=9,
        show_labels=False,
        reducer="mds",
    )

    print(f"\n[DONE] Resultados en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
