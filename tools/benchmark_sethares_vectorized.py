"""
Benchmark y equivalencia: ModeloSethares (bucles) vs versión vectorizada local

No modifica el repo. Compara salida (histograma 12D y total_roughness) y tiempos.

Uso rápido:
  python -m tools.benchmark_sethares_vectorized --repeat 50 --n-harmonics 6 --decay 0.88

Salidas:
  - Reporte en consola con difs máximas y speedup aproximado.

Notas:
  - La vectorización replica exactamente la fórmula usada en pre_process.ModeloSethares:
    suma de disonancias entre todos los pares de parciales de notas distintas, y binea
    por el intervalo (en semitonos) entre fundamentales (no entre parciales), usando
    interval_to_ui_bin(0->11, 1..11->0..10).
  - Las diferencias numéricas esperadas son solo de redondeo (orden de 1e-12..1e-9) por
    cambio del orden de suma en coma flotante.
"""

from __future__ import annotations

import time
import argparse
from typing import List, Tuple

import numpy as np

# Importar pre_process con fallback si no existe chordcodex
try:
    from pre_process import Acorde
    from pre_process import ModeloSethares as LoopSethares
except ModuleNotFoundError as e:
    if "chordcodex" in str(e):
        import sys
        import types
        model_module = types.ModuleType("chordcodex.model")
        class QueryExecutor:  # dummy para satisfacer import
            def __init__(self, *_, **__):
                pass
        model_module.QueryExecutor = QueryExecutor
        chordcodex_pkg = types.ModuleType("chordcodex")
        chordcodex_pkg.model = model_module
        sys.modules["chordcodex"] = chordcodex_pkg
        sys.modules["chordcodex.model"] = model_module
        from pre_process import Acorde
        from pre_process import ModeloSethares as LoopSethares
    else:
        raise
from config import (
    SETHARES_D_STAR,
    SETHARES_S1,
    SETHARES_S2,
    SETHARES_C1,
    SETHARES_C2,
    SETHARES_A1,
    SETHARES_A2,
    SETHARES_N_HARMONICS,
    SETHARES_DECAY,
    SETHARES_BASE_FREQ,
    CHORD_TYPE_INTERVALS,
)


def interval_to_ui_bin(intervalo: int) -> int:
    """Replica la convención del repo: 0->11, 1..11->0..10."""
    return (intervalo - 1) % 12


def sethares_pair_total_vectorized(f1: float, f2: float, n_h: int, decay: float) -> float:
    """Suma de disonancia entre todos los parciales de dos notas (vectorizado HxH).

    Fórmula: a = a1*a2; s = D*/(S1*fmin + S2); D = a*(C1*exp(A1*s*df) + C2*exp(A2*s*df))
    """
    # Armónicos y amplitudes
    K = np.arange(1, n_h + 1, dtype=float)
    A = decay ** (K - 1)

    # Parciales por nota (H,)
    P1 = f1 * K
    P2 = f2 * K

    # Mallas HxH por broadcasting
    P1g = P1[:, None]
    P2g = P2[None, :]
    Fmin = np.minimum(P1g, P2g)
    DF = np.abs(P2g - P1g)
    S = SETHARES_D_STAR / (SETHARES_S1 * Fmin + SETHARES_S2)
    Aprod = (A[:, None] * A[None, :])

    Dmat = Aprod * (
        SETHARES_C1 * np.exp(SETHARES_A1 * S * DF) +
        SETHARES_C2 * np.exp(SETHARES_A2 * S * DF)
    )
    return float(np.sum(Dmat))


def sethares_hist_vectorized(ac: Acorde, base_freq: float, n_h: int, decay: float) -> Tuple[np.ndarray, float]:
    """Replica el histograma 12D y total_roughness de ModeloSethares (vectorizando HxH).

    - Intervalo para bin: semitonos entre fundamentales (no parciales).
    - No aplica suavizado (igual que el retorno actual del modelo).
    """
    # 1) Semitonos relativos de las notas
    semitonos_rel = np.array(np.cumsum([0] + ac.intervals), dtype=float)
    n_notes = len(semitonos_rel)

    # 2) Fundamentales
    if ac.frequencies is not None:
        f0 = np.array(ac.frequencies, dtype=float)
        if f0.ndim > 1:
            f0 = f0[0]
    else:
        f0 = base_freq * (2.0 ** (semitonos_rel / 12.0))

    if len(f0) != n_notes:
        raise ValueError("Mismatch entre #frecuencias y #notas deducidas de 'intervals'.")

    # 3) Histograma y suma total
    hist = np.zeros(12, dtype=float)
    total = 0.0

    # Pares i<j
    for i in range(n_notes - 1):
        for j in range(i + 1, n_notes):
            intervalo = int((semitonos_rel[j] - semitonos_rel[i]) % 12)
            bin_idx = interval_to_ui_bin(intervalo)
            f1, f2 = float(f0[i]), float(f0[j])
            pair_total = sethares_pair_total_vectorized(f1, f2, n_h, decay)
            hist[bin_idx] += pair_total
            total += pair_total

    return hist, float(total)


def build_dataset_from_config_keys() -> List[Acorde]:
    """Usa las claves de CHORD_TYPE_INTERVALS como set base de intervalos típicos."""
    acordes: List[Acorde] = []
    # Para no inflar demasiado el benchmark, filtrar a tamaños 1..5 intervalos (2..6 notas)
    keys = [k for k in CHORD_TYPE_INTERVALS.keys() if 1 <= len(k) <= 5]
    # Crear un acorde por clave, con nombre de la etiqueta
    for k in keys:
        entry = CHORD_TYPE_INTERVALS.get(k)
        label = entry.get("name") if isinstance(entry, dict) else entry
        if not label:
            label = "?"
        acordes.append(Acorde(name=f"{label}_{k}", intervals=list(k)))
    return acordes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=50, help="Repeticiones del dataset para medir tiempo")
    ap.add_argument("--n-harmonics", type=int, default=SETHARES_N_HARMONICS)
    ap.add_argument("--decay", type=float, default=SETHARES_DECAY)
    ap.add_argument("--base-freq", type=float, default=SETHARES_BASE_FREQ)
    ap.add_argument("--atol", type=float, default=1e-8)
    args = ap.parse_args()

    dataset = build_dataset_from_config_keys()
    dataset = dataset * max(1, int(args.repeat))

    # 1) Referencia: implementación con bucles del repo
    loop_model = LoopSethares(config={
        "n_armonicos": int(args.n_harmonics),
        "decaimiento": float(args.decay),
        "base_freq": float(args.base_freq),
    })

    t0 = time.perf_counter()
    loop_out = [loop_model.calcular(ac) for ac in dataset]
    t1 = time.perf_counter()

    # 2) Vectorizado local
    t2 = time.perf_counter()
    vec_out = [sethares_hist_vectorized(ac, float(args.base_freq), int(args.n_harmonics), float(args.decay)) for ac in dataset]
    t3 = time.perf_counter()

    # 3) Comparación
    max_hist_diff = 0.0
    max_total_diff = 0.0
    worst_case = None
    for idx, (ref, vec) in enumerate(zip(loop_out, vec_out)):
        hist_ref, tot_ref = ref
        hist_vec, tot_vec = vec
        # Misma longitud
        if hist_ref.shape != hist_vec.shape:
            raise AssertionError(f"Hist shapes difieren en idx={idx}: {hist_ref.shape} vs {hist_vec.shape}")
        diff_hist = float(np.max(np.abs(hist_ref - hist_vec)))
        diff_total = float(abs(float(tot_ref) - float(tot_vec)))
        if diff_hist > max_hist_diff or diff_total > max_total_diff:
            max_hist_diff = max(max_hist_diff, diff_hist)
            max_total_diff = max(max_total_diff, diff_total)
            worst_case = (idx, dataset[idx].name, dataset[idx].intervals)

    loop_time = t1 - t0
    vec_time = t3 - t2
    speedup = loop_time / vec_time if vec_time > 0 else float("inf")

    print("=== Benchmark Sethares: bucles vs vectorizado ===")
    print(f"Dataset base (CHORD_TYPE_INTERVALS claves): {len(CHORD_TYPE_INTERVALS)} acordes únicos")
    print(f"Dataset total tras repeat={args.repeat}: {len(dataset)} acordes")
    print(f"Tiempo (loops): {loop_time:.4f}s | Tiempo (vector): {vec_time:.4f}s | speedup x{speedup:.2f}")
    print(f"Max |hist_ref - hist_vec| = {max_hist_diff:.3e}")
    print(f"Max |total_ref - total_vec| = {max_total_diff:.3e}")
    if worst_case is not None:
        idx, name, ivs = worst_case
        print(f"Peor caso en idx={idx}: {name} {ivs}")

    # Criterio de equivalencia (tolerancias)
    ok_hist = max_hist_diff <= args.atol
    ok_total = max_total_diff <= args.atol
    status = "OK" if ok_hist and ok_total else "MISMATCH"
    print(f"Equivalencia: {status} (atol={args.atol:g})")


if __name__ == "__main__":
    main()
