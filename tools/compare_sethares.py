"""
Herramienta de comparación del modelo de rugosidad de Sethares.

Propósito
- Comparar, de manera controlada y reproducible, la salida de una implementación
  canónica (basada en literatura) vs. la implementación actual del repositorio
  (pre_process.ModeloSethares) para acordes simples (díadas) y para triadas.

Salida
- CSV con curvas de disonancia por intervalo (1..12) para cada implementación.
- PNG opcional con las curvas superpuestas.
- Resumen en consola con diferencias relativas por intervalo.

Uso rápido
  python -m tools.compare_sethares --plots --save-dir outputs/sethares --base-freq 500 --n-harmonics 10

Notas
- La implementación canónica usa todos los armónicos hasta N con decaimiento geométrico.
- La implementación del repo se toma de pre_process.ModeloSethares (la última clase
  definida en ese archivo). Si hay duplicados, Python conserva la segunda.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Importar desde el repo
from pre_process import Acorde, ModeloSethares as RepoSethares


# -----------------------------
# Sethares canónico (literatura)
# -----------------------------

def _disonancia_sethares_pair(f_min: float, f_max: float, a1: float, a2: float,
                              b1: float = 3.5, b2: float = 5.75, X_star: float = 0.24) -> float:
    """Disonancia entre dos parciales siguiendo Plomp-Levelt/Sethares estándar.

    s = X*/(0.0207*f_min + 18.96)
    D = a1 * a2 * (exp(-b1*s*df) - exp(-b2*s*df))
    """
    s = X_star / (0.0207 * f_min + 18.96)
    df = abs(f_max - f_min)
    return a1 * a2 * (np.exp(-b1 * s * df) - np.exp(-b2 * s * df))


@dataclass
class CanonicalConfig:
    base_freq: float = 440.0
    n_harmonics: int = 10
    decay: float = 0.8  # amplitud geométrica para el k-ésimo armónico: decay**(k-1)


def canonical_sethares_total_roughness(fundamentals: List[float], cfg: CanonicalConfig) -> float:
    """Suma la disonancia entre todos los pares de parciales de notas distintas.

    - Genera armónicos para cada nota (f_i * k, k=1..N) con amplitud decay**(k-1).
    - Suma disonancias entre parciales de notas distintas (no intra-nota).
    """
    # Construir lista de (frecuencia, amplitud, idx_nota)
    partials: List[Tuple[float, float, int]] = []
    for note_idx, f0 in enumerate(fundamentals):
        for k in range(1, cfg.n_harmonics + 1):
            f = f0 * k
            a = cfg.decay ** (k - 1)
            partials.append((f, a, note_idx))

    # Sumar disonancias solo entre parciales que provienen de notas distintas
    total = 0.0
    for i in range(len(partials) - 1):
        f1, a1, n1 = partials[i]
        for j in range(i + 1, len(partials)):
            f2, a2, n2 = partials[j]
            if n1 == n2:
                continue
            fmin, fmax = (f1, f2) if f1 < f2 else (f2, f1)
            total += _disonancia_sethares_pair(fmin, fmax, a1, a2)
    return float(total)


# --- Versión estilo notebook (C1*exp(A1*s*df)+C2*exp(A2*s*df), amplitud=product) ---

def _nb_pair(f1: float, f2: float, a1: float, a2: float,
             C1: float = 5.0, C2: float = -5.0, A1: float = -3.51, A2: float = -5.75,
             Dstar: float = 0.24, S1: float = 0.0207, S2: float = 18.96) -> float:
    fmin = min(f1, f2)
    s = Dstar / (S1 * fmin + S2)
    df = abs(f2 - f1)
    a = a1 * a2
    return a * (C1 * np.exp(A1 * s * df) + C2 * np.exp(A2 * s * df))


@dataclass
class NotebookConfig:
    base_freq: float = 440.0
    n_harmonics: int = 6
    decay: float = 0.88


def notebook_sethares_total_roughness(fundamentals: List[float], cfg: NotebookConfig) -> float:
    partials: List[Tuple[float, float, int]] = []
    for note_idx, f0 in enumerate(fundamentals):
        for k in range(1, cfg.n_harmonics + 1):
            f = f0 * k
            a = cfg.decay ** (k - 1)
            partials.append((f, a, note_idx))
    total = 0.0
    for i in range(len(partials) - 1):
        f1, a1, n1 = partials[i]
        for j in range(i + 1, len(partials)):
            f2, a2, n2 = partials[j]
            if n1 == n2:
                continue
            total += _nb_pair(f1, f2, a1, a2)
    return float(total)


# -----------------------------
# Ayudas para díadas y triadas
# -----------------------------

def dyad_fundamentals(base_freq: float, semitone_interval: int) -> List[float]:
    return [base_freq, base_freq * 2 ** (semitone_interval / 12.0)]


def triad_fundamentals(base_freq: float, intervals: List[int]) -> List[float]:
    semitones = np.cumsum([0] + intervals)
    return list(base_freq * 2 ** (semitones / 12.0))


# -----------------------------
# Comparación controlada
# -----------------------------

def compare_dyad_curve(base_freq: float, n_harmonics: int, decay: float,
                       save_dir: Path | None, make_plots: bool,
                       nb_n_harmonics: int = 6, nb_decay: float = 0.88) -> None:
    cfg = CanonicalConfig(base_freq=base_freq, n_harmonics=n_harmonics, decay=decay)
    repo_model = RepoSethares(config={"base_freq": base_freq})

    intervals = list(range(1, 13))  # 1..12
    canon_vals = []
    repo_vals = []
    nb_vals = []

    for k in intervals:
        fund = dyad_fundamentals(base_freq, k)
        canon = canonical_sethares_total_roughness(fund, cfg)
        canon_vals.append(canon)

        acorde = Acorde(name=f"dyad_{k}", intervals=[k], frequencies=fund)
        _, repo_total = repo_model.calcular(acorde)
        repo_vals.append(float(repo_total))

        nb_cfg = NotebookConfig(base_freq=base_freq, n_harmonics=nb_n_harmonics, decay=nb_decay)
        nb_total = notebook_sethares_total_roughness(fund, nb_cfg)
        nb_vals.append(nb_total)

    # Normalizaciones para comparación visual (es común normalizar por el máximo)
    cmax = max(canon_vals) or 1.0
    rmax = max(repo_vals) or 1.0
    nbmax = max(nb_vals) or 1.0
    canon_norm = [v / cmax for v in canon_vals]
    repo_norm = [v / rmax for v in repo_vals]
    nb_norm = [v / nbmax for v in nb_vals]

    # Exportar CSV
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / f"dyad_curve_b{int(base_freq)}_H{n_harmonics}_d{decay}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["interval", "canonical", "canonical_norm", "repo", "repo_norm", "nb", "nb_norm"])
            for k, c, cn, r, rn, nb, nbn in zip(intervals, canon_vals, canon_norm, repo_vals, repo_norm, nb_vals, nb_norm):
                w.writerow([k, c, cn, r, rn, nb, nbn])

    # Gráfica
    if make_plots:
        plt.figure(figsize=(8, 5))
        plt.plot(intervals, canon_norm, "-o", label="Canónico (norm)")
        plt.plot(intervals, repo_norm, "-o", label="Repo (norm)")
        plt.plot(intervals, nb_norm, "-o", label="Terico (norm)")
        plt.xticks(intervals)
        plt.xlabel("Intervalo (semitonos)")
        plt.ylabel("Rugosidad normalizada")
        plt.title(f"Curva díadas – f0={base_freq}Hz, N={n_harmonics}/{nb_n_harmonics}, decay={decay}/{nb_decay}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        if save_dir:
            plt.savefig(save_dir / f"dyad_curve_triple_b{int(base_freq)}_H{n_harmonics}_{nb_n_harmonics}_d{decay}_{nb_decay}.png", dpi=180, bbox_inches="tight")
        else:
            plt.show()

    # Reporte breve de diferencias relativas
    print("\nResumen de diferencias (norm.):")
    for k, cn, rn, nn in zip(intervals, canon_norm, repo_norm, nb_norm):
        print(f"  k={k:2d}: repo-canon={rn-cn:+.4f}, nb-canon={nn-cn:+.4f}, repo-nb={rn-nn:+.4f}")


def main():
    p = argparse.ArgumentParser(description="Comparación Sethares canónico vs repo")
    p.add_argument("--base-freq", type=float, default=500.0)
    p.add_argument("--n-harmonics", type=int, default=10)
    p.add_argument("--decay", type=float, default=0.8)
    p.add_argument("--save-dir", type=Path, default=None)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--nb-n-harmonics", type=int, default=6)
    p.add_argument("--nb-decay", type=float, default=0.88)
    args = p.parse_args()

    print("Comparando curvas de díadas…")
    compare_dyad_curve(
        base_freq=args.base_freq,
        n_harmonics=args.n_harmonics,
        decay=args.decay,
        save_dir=args.save_dir,
        make_plots=args.plots,
        nb_n_harmonics=args.nb_n_harmonics,
        nb_decay=args.nb_decay,
    )


if __name__ == "__main__":
    main()
