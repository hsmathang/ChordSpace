"""
Prueba de equivalencia: suma de díadas vs. rugosidad total del acorde

Objetivo
- Verificar empíricamente que la rugosidad total de un acorde (modelo Sethares)
  es igual a la suma de las rugosidades de todas las díadas que componen el acorde,
  cuando ambas usan el mismo esquema canónico (interacciones entre parciales con
  N armónicos y decaimiento geométrico) y no se aplican postprocesos no lineales.

Cómo funciona
- Para cada acorde, calculamos:
  (A) total_repo: total_roughness devuelto por pre_process.ModeloSethares.calcular(acorde)
  (B) total_sum_dyads: sum_{pares de notas} canonical_sethares_total_roughness(fundamentals_par, cfg)
- Reportamos diferencias absoluta y relativa, y exportamos un CSV.

Uso
  # dataset sintético
  python -m tools.test_total_roughness_equivalence --dataset synthetic --base-freq 440 --n-harmonics 10 --decay 0.8 --out outputs/equiv_synth

  # dataset desde DB
  python -m tools.test_total_roughness_equivalence --dataset db:QUERY_CHORDS_WITH_NAME --limit 100 --n-harmonics 10 --decay 0.8 --out outputs/equiv_db
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np

from pre_process import Acorde, ModeloSethares
from tools.compare_sethares import (
    CanonicalConfig, canonical_sethares_total_roughness,
    NotebookConfig, notebook_sethares_total_roughness,
)


def build_synthetic(base_freq: float) -> List[Acorde]:
    specs = [
        ("C_maj", [4, 3]), ("A_min", [3, 4]), ("B_dim", [3, 3]),
        ("G7", [4, 3, 3]), ("Dm7", [3, 4, 3]), ("Cmaj7", [4, 3, 4]),
        ("Csus2", [2, 5]), ("Csus4", [5, 2]),
    ]
    return [Acorde(name=n, intervals=iv) for n, iv in specs]


def build_from_db(const_name: str, limit: int | None) -> List[Acorde]:
    import config
    from chordcodex.model import QueryExecutor
    from pre_process import ChordAdapter

    query = getattr(config, const_name)
    qe = QueryExecutor(**config.config_db)
    df = qe.as_pandas(query)
    if limit is not None and len(df) > limit:
        df = df.sample(n=limit, random_state=42)
    acordes: List[Acorde] = []
    for _, row in df.iterrows():
        acordes.append(ChordAdapter.from_csv_row(row))
    return acordes


def fundamentals_for(ac: Acorde, base_freq: float) -> np.ndarray:
    if ac.frequencies is not None:
        freqs = np.array(ac.frequencies, dtype=float)
        if freqs.ndim > 1:
            freqs = freqs[0]
        return freqs
    semitones = np.cumsum([0] + ac.intervals)
    return base_freq * 2 ** (np.array(semitones) / 12.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="synthetic o db:<CONST_NAME> definida en config.py")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base-freq", type=float, default=440.0)
    ap.add_argument("--n-harmonics", type=int, default=10)
    ap.add_argument("--decay", type=float, default=0.8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--formula", choices=["notebook", "canonical"], default="notebook",
                    help="Fórmula para sumar díadas: 'notebook' (C1/C2/A1/A2, product) o 'canonical' (exp-diff, product)")
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    if args.dataset == "synthetic":
        acordes = build_synthetic(args.base_freq)
    elif args.dataset.startswith("db:"):
        acordes = build_from_db(args.dataset.split(":", 1)[1], args.limit)
    else:
        raise ValueError("--dataset debe ser 'synthetic' o 'db:<CONST_NAME>'")

    # Modelos/parámetros
    repo = ModeloSethares(config={
        "base_freq": args.base_freq,
        "n_armonicos": args.n_harmonics,
        "decaimiento": args.decay,
    })
    can_cfg = CanonicalConfig(base_freq=args.base_freq, n_harmonics=args.n_harmonics, decay=args.decay)
    nb_cfg = NotebookConfig(base_freq=args.base_freq, n_harmonics=args.n_harmonics, decay=args.decay)

    rows = [("name", "intervals", "total_repo", "total_sum_dyads", "abs_diff", "rel_diff")]
    abs_diffs = []
    rel_diffs = []

    for ac in acordes:
        # (A) Total del modelo del repo
        _, total_repo = repo.calcular(ac)

        # (B) Suma de díadas con la fórmula seleccionada
        f = fundamentals_for(ac, args.base_freq)
        total_sum = 0.0
        for i in range(len(f) - 1):
            for j in range(i + 1, len(f)):
                if args.formula == "notebook":
                    total_sum += notebook_sethares_total_roughness([f[i], f[j]], nb_cfg)
                else:
                    total_sum += canonical_sethares_total_roughness([f[i], f[j]], can_cfg)

        abs_diff = float(abs(total_repo - total_sum))
        rel_diff = float(abs_diff / max(1e-12, abs(total_sum)))
        abs_diffs.append(abs_diff)
        rel_diffs.append(rel_diff)
        rows.append((ac.name, str(ac.intervals), total_repo, total_sum, abs_diff, rel_diff))

    # Guardar CSV
    csv_path = out_dir / "equivalence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

    # Resumen
    summary = out_dir / "summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write(f"Base freq={args.base_freq}, N={args.n_harmonics}, decay={args.decay}\n")
        f.write(f"N acordes: {len(acordes)}\n")
        f.write(f"Abs diff: mean={np.mean(abs_diffs):.6e}, max={np.max(abs_diffs):.6e}\n")
        f.write(f"Rel diff: mean={np.mean(rel_diffs):.6e}, max={np.max(rel_diffs):.6e}\n")

    # Imprimir resumen en consola también
    print("Listo. Resultados en:", out_dir)
    print(f"Base freq={args.base_freq}, N={args.harmonics if hasattr(args,'harmonics') else args.n_harmonics}, decay={args.decay}, formula={args.formula}")
    print(f"Abs diff: mean={np.mean(abs_diffs):.6e}, max={np.max(abs_diffs):.6e}")
    print(f"Rel diff: mean={np.mean(rel_diffs):.6e}, max={np.max(rel_diffs):.6e}")


if __name__ == "__main__":
    main()
