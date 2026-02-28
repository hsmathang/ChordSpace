"""
experiments/k_selection_p_canon.py
------------------------------------
Experimento de selección de k para métricas de calidad del embedding
sobre la población P_CANON: todas las tríadas cromáticas en la octava 4
con span <= 12 semitonos (una octava cerrada).

Objetivo científico:
    Determinar empíricamente qué valor de k ∈ {3, 5, 10, 20} maximiza
    simultáneamente Trustworthiness T(k) y Continuity C(k) bajo MDS-SMACOF
    con métrica euclidiana sobre el espacio Φ ∈ ℝ¹².

Justificación (aclaraciones.md):
    El trabajo es Ciencia de Datos + Musicología computacional.
    La elección de k debe ser validada empíricamente, no impuesta
    desde la teoría armónica (como k=3 PLR). Este script genera
    el dato empírico que respalda—o refuta—la elección de k=3.

Población P_CANON:
    - Alfabeto A = {0, 1, ..., 11} (cromático completo)
    - Octava fija: octave_min = octave_max = 4
    - Cardinalidades: {3}  (tríadas)
    - Filtro: span <= 12 semitonos

Salida:
    - Tabla T(k), C(k), T+C promedio para cada k
    - Gráfica T(k) y C(k) vs k guardada en experiments/figures/
    - Recomendación del k óptimo impresa en consola
    - El resultado puede citarse directamente en §3.6 del capítulo de metodología
"""

import sys
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import trustworthiness as sk_trustworthiness

# --- Añadir raíz del proyecto al path ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.combinatorial_generator import generate_combinatorial_chords
from metrics import compute_continuity

# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN DEL EXPERIMENTO
# ─────────────────────────────────────────────
K_VALUES = [3, 5, 10, 20]  # vecindarios a evaluar
SPAN_MAX = 12               # semitonos – una octava
OCTAVE = 4                  # octava central (MIDI 60-71)
CARDINALITIES = [3]         # solo tríadas
ALPHABET = list(range(12))  # escala cromática completa
MDS_SEED = 17               # semilla determinista (SMACOF)
N_COMPONENTS = 2            # embedding 2D

FIGURES_DIR = ROOT / "experiments" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 2. GENERACIÓN DE P_CANON
# ─────────────────────────────────────────────
def build_p_canon() -> np.ndarray:
    """
    Genera P_CANON: tríadas cromáticas en octava 4, span <= SPAN_MAX.
    Retorna la matriz de vectores Φ ∈ ℝ¹² (chroma de roughness normalizado).
    """
    print(f"[k_selection] Generando P_CANON: alfa={ALPHABET}, oct={OCTAVE}, k={CARDINALITIES}")
    df = generate_combinatorial_chords(
        alphabet=ALPHABET,
        octave_min=OCTAVE,
        octave_max=OCTAVE,
        cardinalities=CARDINALITIES,
    )
    # Filtro span <= SPAN_MAX
    df = df[df["span_semitones"] <= SPAN_MAX].reset_index(drop=True)
    print(f"[k_selection] P_CANON: {len(df)} tríadas tras filtro span <= {SPAN_MAX} st")

    # Extraer vector Φ (chroma) — columna 'chroma' es lista de 12 floats
    phi = np.array([json.loads(row) if isinstance(row, str) else row
                    for row in df["chroma"]], dtype=float)
    print(f"[k_selection] Φ shape: {phi.shape}")
    return phi


# ─────────────────────────────────────────────
# 3. REDUCCIÓN MDS-SMACOF
# ─────────────────────────────────────────────
def run_mds(phi: np.ndarray) -> np.ndarray:
    """Aplica MDS métrico (SMACOF) al espacio Φ ∈ ℝ¹²."""
    print(f"[k_selection] Ejecutando MDS-SMACOF (seed={MDS_SEED})...")
    mds = MDS(
        n_components=N_COMPONENTS,
        dissimilarity="euclidean",
        random_state=MDS_SEED,
        n_init=4,
        max_iter=300,
    )
    Y = mds.fit_transform(phi)
    print(f"[k_selection] Stress final: {mds.stress_:.4f}")
    return Y


# ─────────────────────────────────────────────
# 4. CÁLCULO DE T(k) y C(k) PARA CADA k
# ─────────────────────────────────────────────
def evaluate_k_range(phi: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
    """Calcula T(k) y C(k) para cada k en K_VALUES."""
    results = []
    for k in K_VALUES:
        if k >= len(phi):
            print(f"[k_selection] k={k} >= N={len(phi)}, omitido.")
            continue
        t = sk_trustworthiness(phi, Y, n_neighbors=k)
        c = compute_continuity(phi, Y, n_neighbors=k)
        harmonic_mean = 2 * t * c / (t + c) if (t + c) > 0 else 0.0
        results.append({"k": k, "T(k)": round(t, 4), "C(k)": round(c, 4), "HM(T,C)": round(harmonic_mean, 4)})
        print(f"  k={k:2d}  T={t:.4f}  C={c:.4f}  HM={harmonic_mean:.4f}")
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 5. GRÁFICA T(k) y C(k)
# ─────────────────────────────────────────────
def plot_results(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["k"], df["T(k)"], "o-", label="Trustworthiness T(k)", color="#1f77b4")
    ax.plot(df["k"], df["C(k)"], "s-", label="Continuity C(k)", color="#ff7f0e")
    ax.plot(df["k"], df["HM(T,C)"], "^--", label="Media Armónica HM(T,C)", color="#2ca02c", alpha=0.7)
    ax.axvline(x=3, color="gray", linestyle=":", linewidth=1.2, label="k=3 (candidato)")
    ax.set_xlabel("k (vecindario)")
    ax.set_ylabel("Calidad del embedding")
    ax.set_title("Selección de k: T(k) y C(k) sobre P_CANON\n(Tríadas cromáticas, octava 4, span ≤ 12 st, MDS-SMACOF)")
    ax.legend()
    ax.set_xticks(df["k"])
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    out_path = FIGURES_DIR / "k_selection_p_canon.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[k_selection] Gráfica guardada en: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────
# 6. REPORTE FINAL
# ─────────────────────────────────────────────
def report(df: pd.DataFrame):
    best_row = df.loc[df["HM(T,C)"].idxmax()]
    print("\n═══════════════════════════════════════════════")
    print("RESULTADO DEL EXPERIMENTO K_SELECTION_P_CANON")
    print("═══════════════════════════════════════════════")
    print(df.to_string(index=False))
    print(f"\n→ k óptimo (máxima HM(T,C)): k = {int(best_row['k'])}")
    print(f"  T({int(best_row['k'])}) = {best_row['T(k)']:.4f}")
    print(f"  C({int(best_row['k'])}) = {best_row['C(k)']:.4f}")
    if int(best_row["k"]) == 3:
        print("\n✅ k=3 está empíricamente respaldado en P_CANON.")
        print("   Puede citarse en §3.6 con la siguiente fórmula:")
        print(f"   'k=3 maximizó la media armónica de T(k) y C(k) sobre P_CANON [T={"%.4f" % best_row["T(k)"]}, C={"%.4f" % best_row["C(k)"]}]'")
    else:
        print(f"\n⚠️  k={int(best_row['k'])} es el óptimo empírico, NO k=3.")
        print("   Actualizar la justificación en §3.6 con este resultado.")
    print("═══════════════════════════════════════════════\n")
    # Guardar tabla como CSV
    csv_path = FIGURES_DIR / "k_selection_p_canon_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[k_selection] Tabla guardada en: {csv_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    phi = build_p_canon()
    Y = run_mds(phi)
    print(f"\n[k_selection] Evaluando T(k) y C(k) para k ∈ {K_VALUES}...")
    df_results = evaluate_k_range(phi, Y)
    plot_results(df_results)
    report(df_results)
