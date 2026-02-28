"""
exp_estructuras_basicas_12tonos.py
====================================
Extensión de `exp_estructuras_basicas_c3.py`.

En lugar de anclar todo a C3, aquí se toma cada estructura básica
(12 díadas + 4 tríadas × 3 inversiones) y se transpone a los 12 tonos
del vocabulario cromático (raíz de 0 a 11), manteniendo TODAS las notas
dentro de un rango de dos octavas (MIDI 0–23).

Resultado:
  - 12 díadas × 12 transportes = 144 díadas
  - 4 × 3 inversiones × 12 transportes = 144 tríadas
  - Total: 288 acordes

Simbología en el plot (igual que en el experimento base):
  - Díadas → rombo sin relleno, agrupadas por tipo de intervalo
  - Tríadas → triángulo según cualidad (Maj/Min/Dim/Aug)
  - Coloreado por calidad, sin etiquetas de texto

Ordena las entries para que el plot dibuje primero las díadas (más)
y encima las tríadas (menos puntos → visibles).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_utils import build_entries, run_experiment

OUTPUT_DIR = Path(__file__).parent / "outputs" / "estructuras_basicas_12tonos"

# Rango de 2 octavas (MIDI 0 a 23), raíces de 0 a 11
RAICES = list(range(12))
OCTAVA = 12

# Tipos de díadas (1 a 12 semitonos)
NOMBRES_INT = ["m2", "M2", "m3", "M3", "P4", "TT", "P5", "m6", "M6", "m7", "M7", "P8"]

# Estructuras de tríadas básicas (semitonos desde la raíz)
ESTRUCTURAS_TRIADAS = {
    "Maj": [0, 4, 7],
    "Min": [0, 3, 7],
    "Dim": [0, 3, 6],
    "Aug": [0, 4, 8],
}


def to_freqs(notas_midi):
    return [440.0 * 2 ** ((m - 69) / 12.0) for m in notas_midi]


def build_all_chords():
    chords_raw = []

    # ── 1. Díadas: 12 intervalos × 12 raíces ─────────────────────────────────
    for raiz in RAICES:
        for i, nombre_int in enumerate(NOMBRES_INT, start=1):
            nota_baja = raiz
            nota_alta = raiz + i  # Siempre cabe en 2 octavas (max = 11+12 = 23)
            notas = [nota_baja, nota_alta]
            nombre = f"Diada-{nombre_int}-{raiz}"
            intervals = [i]
            chords_raw.append((nombre, intervals, to_freqs(notas), notas))

    # ── 2. Tríadas: 4 tipos × 3 inversiones × 12 raíces ──────────────────────
    for raiz in RAICES:
        for tipo, semitonos in ESTRUCTURAS_TRIADAS.items():
            # Fundamental: raiz + semitonos
            notas_fund = [raiz + s for s in semitonos]
            
            # 1ª Inversión: subimos la raíz una octava, reanclar al fondo
            inv1_tmp = [semitonos[1], semitonos[2], semitonos[0] + OCTAVA]
            offset1 = inv1_tmp[0]
            notas_inv1 = [raiz + (n - offset1) for n in inv1_tmp]
            
            # 2ª Inversión: subimos raíz y tercera una octava, reanclar
            inv2_tmp = [semitonos[2], semitonos[0] + OCTAVA, semitonos[1] + OCTAVA]
            offset2 = inv2_tmp[0]
            notas_inv2 = [raiz + (n - offset2) for n in inv2_tmp]
            
            variaciones = [
                (f"{tipo}-fund-{raiz}", notas_fund),
                (f"{tipo}-inv1-{raiz}", notas_inv1),
                (f"{tipo}-inv2-{raiz}", notas_inv2),
            ]
            
            for nombre, notas in variaciones:
                # Verificar que todas las notas caben en el rango MIDI 0-23
                if min(notas) < 0 or max(notas) > 23:
                    continue  # Descartar si sale del rango de 2 octavas
                intervals_ch = [notas[j + 1] - notas[j] for j in range(len(notas) - 1)]
                chords_raw.append((nombre, intervals_ch, to_freqs(notas), notas))

    return chords_raw


def main():
    chords_raw = build_all_chords()
    
    # Estadísticas de construcción
    n_diadas = sum(1 for c in chords_raw if c[0].startswith("Diada"))
    n_triadas = len(chords_raw) - n_diadas
    print(f"[INFO] Población: {len(chords_raw)} acordes "
          f"({n_diadas} díadas + {n_triadas} tríadas) — 12 raíces, 2 octavas")

    print("[INFO] Calculando vectores 12-D con 'perclass_alpha0_75'...")
    entries, X = build_entries(chords_raw, proposal="perclass_alpha0_75")

    # ─── Simbología y orden de renderizado ──────────────────────────────────
    # Se renderizan primero las díadas (más puntos) y encima las tríadas
    # (menos puntos → visibles aunque se superpongan).
    facets = []
    symbols = []

    for e in entries:
        nombre = e.identity_name
        partes = nombre.split("-")
        tipo = partes[0]  # "Diada", "Maj", "Min", "Dim", "Aug"
        
        if tipo == "Diada":
            int_name = partes[1]   # "m2", "P4", etc.
            # Agrupamos en intervalos de 3ª o menos / de 4ª a 7ª / octava
            intervalo = int(partes[2]) if len(partes) > 2 else 0  # raiz como int
            facets.append(f"Díadas")
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
            symbols.append("square-open")

    # Opacidades: díadas ligeramente más transparentes (más puntos), tríadas sólidas
    opacities = [0.6 if f.startswith("Díadas") else 1.0 for f in facets]

    print("[INFO] Ejecutando MDS...")
    run_experiment(
        entries=entries,
        X=X,
        metric="euclidean",
        output_dir=OUTPUT_DIR,
        experiment_name="EstructurasBasicas_12Tonos",
        scatter_title="Díadas y Tríadas Básicas — 12 Tonos, 2 Octavas",
        n_init=6,
        scatter_mode="faceted",
        facet_labels=facets,
        facet_symbols=symbols,
        facet_opacities=opacities,
        facet_layout="single_overview",
        label_font_size=8,
        show_labels=False,
    )

    print(f"\n[DONE] Resultados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
