"""
Comparación de embeddings entre dos modelos (Sethares repo vs Sethares canónico)
sin depender del notebook.

Características
- Carga dataset sintético o desde DB (usando config + chordcodex).
- Ejecuta el pipeline del laboratorio para cada modelo y guarda artefactos.
- Produce figuras HTML (scatter, heatmap, shepard) para cada modelo y un
  resumen con estadísticas cruzadas (correlación de distancias, etc.).

Uso
  # Dataset sintético
  python -m tools.compare_embeddings --dataset synthetic --metric cosine \
      --reduction UMAP --out outputs/compare_synth

  # Dataset desde DB (constante definida en config.py)
  python -m tools.compare_embeddings --dataset db:QUERY_CHORDS_WITH_NAME_AND_RANDOM_CHORDS_POBLATION \
      --limit 200 --metric cosine --reduction MDS --out outputs/compare_db

Parámetros de Sethares canónico
- --n-harmonics y --decay controlan número de armónicos y decaimiento geométrico.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from lab import LaboratorioAcordes
from pre_process import (
    Acorde, ModeloSethares as RepoSethares, safe_normalize, ChordAdapter
)
from visualization import (
    visualizar_scatter_density, visualizar_heatmap, graficar_shepard
)


# -----------------------------
# Sethares canónico como clase compatible
# -----------------------------

class CanonicalSethares:
    def __init__(self, base_freq: float = 440.0, n_harmonics: int = 10, decay: float = 0.8):
        self.config = {
            "base_freq": base_freq,
            "n_harmonics": n_harmonics,
            "decay": decay,
        }

    @staticmethod
    def _disonancia_pair(f_min: float, f_max: float, a1: float, a2: float,
                         b1: float = 3.5, b2: float = 5.75, X_star: float = 0.24) -> float:
        s = X_star / (0.0207 * f_min + 18.96)
        df = abs(f_max - f_min)
        return a1 * a2 * (np.exp(-b1 * s * df) - np.exp(-b2 * s * df))

    def calcular(self, acorde: Acorde) -> Tuple[np.ndarray, float]:
        # Obtener fundamentales
        if acorde.frequencies is not None:
            freqs = np.array(acorde.frequencies, dtype=float)
            if freqs.ndim > 1:
                freqs = freqs[0]
            fundamentals = freqs
        else:
            base_freq = self.config.get("base_freq", 440.0)
            semitones = np.cumsum([0] + acorde.intervals)
            fundamentals = base_freq * 2 ** (np.array(semitones) / 12.0)

        n = len(fundamentals)
        n_h = int(self.config.get("n_harmonics", 10))
        decay = float(self.config.get("decay", 0.8))

        # Histograma por intervalo en semitonos (entre fundamentales)
        hist = np.zeros(12, dtype=float)
        total = 0.0

        for i in range(n - 1):
            for j in range(i + 1, n):
                f1, f2 = fundamentals[i], fundamentals[j]
                # Sumar disonancia de todos los parciales entre estas dos notas
                pair_total = 0.0
                for k1 in range(1, n_h + 1):
                    for k2 in range(1, n_h + 1):
                        p1 = f1 * k1
                        p2 = f2 * k2
                        a1 = decay ** (k1 - 1)
                        a2 = decay ** (k2 - 1)
                        fmin, fmax = (p1, p2) if p1 < p2 else (p2, p1)
                        pair_total += self._disonancia_pair(fmin, fmax, a1, a2)

                # Intervalo en semitonos entre fundamentales (redondeado)
                ratio = f2 / f1 if f2 >= f1 else f1 / f2
                semi = int(np.round(12 * np.log2(ratio))) % 12
                if semi > 0:
                    hist[semi - 1] += pair_total
                total += pair_total

        return safe_normalize(hist), float(total)


# -----------------------------
# Sethares estilo notebook (C1*exp(A1*s*df)+C2*exp(A2*s*df), amplitud=product)
# -----------------------------

class NotebookSethares:
    def __init__(self, base_freq: float = 440.0, n_harmonics: int = 6, decay: float = 0.88,
                 C1: float = 5.0, C2: float = -5.0, A1: float = -3.51, A2: float = -5.75,
                 Dstar: float = 0.24, S1: float = 0.0207, S2: float = 18.96):
        self.config = {
            "base_freq": base_freq,
            "n_harmonics": n_harmonics,
            "decay": decay,
        }
        self.C1, self.C2, self.A1, self.A2 = C1, C2, A1, A2
        self.Dstar, self.S1, self.S2 = Dstar, S1, S2

    def _pair(self, f1: float, f2: float, a1: float, a2: float) -> float:
        fmin = min(f1, f2)
        s = self.Dstar / (self.S1 * fmin + self.S2)
        df = abs(f2 - f1)
        a = a1 * a2  # product
        return a * (self.C1 * np.exp(self.A1 * s * df) + self.C2 * np.exp(self.A2 * s * df))

    def calcular(self, acorde: Acorde) -> Tuple[np.ndarray, float]:
        if acorde.frequencies is not None:
            freqs = np.array(acorde.frequencies, dtype=float)
            if freqs.ndim > 1:
                freqs = freqs[0]
            fundamentals = freqs
        else:
            base_freq = self.config.get("base_freq", 440.0)
            semitones = np.cumsum([0] + acorde.intervals)
            fundamentals = base_freq * 2 ** (np.array(semitones) / 12.0)

        n = len(fundamentals)
        n_h = int(self.config.get("n_harmonics", 6))
        decay = float(self.config.get("decay", 0.88))

        hist = np.zeros(12, dtype=float)
        total = 0.0

        for i in range(n - 1):
            for j in range(i + 1, n):
                f1, f2 = fundamentals[i], fundamentals[j]
                pair_total = 0.0
                for k1 in range(1, n_h + 1):
                    for k2 in range(1, n_h + 1):
                        p1 = f1 * k1
                        p2 = f2 * k2
                        a1 = decay ** (k1 - 1)
                        a2 = decay ** (k2 - 1)
                        pair_total += self._pair(p1, p2, a1, a2)
                ratio = f2 / f1 if f2 >= f1 else f1 / f2
                semi = int(np.round(12 * np.log2(ratio))) % 12
                if semi > 0:
                    hist[semi - 1] += pair_total
                total += pair_total

        return safe_normalize(hist), float(total)


# -----------------------------
# Datasets
# -----------------------------

def dataset_synthetic() -> List[Acorde]:
    specs = [
        ("C_maj", [4, 3]), ("A_min", [3, 4]), ("B_dim", [3, 3]),
        ("G7", [4, 3, 3]), ("Dm7", [3, 4, 3]), ("Cmaj7", [4, 3, 4]),
        ("Csus2", [2, 5]), ("Csus4", [5, 2]),
    ]
    acordes = [Acorde(name=n, intervals=iv) for n, iv in specs]
    # Añadir variaciones simples
    for t in [2, 5, 7, 9]:
        for n, iv in [("T_maj", [4, 3]), ("T_min", [3, 4]), ("T_dim", [3, 3])]:
            acordes.append(Acorde(name=f"{n}_{t}", intervals=iv))
    return acordes


def dataset_db(query_name: str, limit: int | None = None) -> List[Acorde]:
    # Cargar constantes y conexión
    import config
    from chordcodex.model import QueryExecutor

    query = getattr(config, query_name)
    qe = QueryExecutor(**config.config_db)
    df = qe.as_pandas(query)
    if limit is not None and len(df) > limit:
        df = df.sample(n=limit, random_state=42)
    acordes: List[Acorde] = []
    for _, row in df.iterrows():
        acordes.append(ChordAdapter.from_csv_row(row))
    return acordes


# -----------------------------
# Ejecución y comparación
# -----------------------------

def run_pipeline(acordes: List[Acorde], model_name: str, model_obj, metric: str, reduction: str,
                 out_dir: Path, tag: str):
    lab = LaboratorioAcordes(acordes)
    res = lab.ejecutar_experimento(model_obj, None, metrica=metric, reduccion=reduction)

    # Artefactos por modelo
    np.save(out_dir / f"embeddings_{tag}.npy", res.embeddings)
    np.save(out_dir / f"distances_{tag}.npy", res.matriz_distancias)

    fig_sc = visualizar_scatter_density(res.embeddings, acordes, res.X_original)
    fig_hm = visualizar_heatmap(res.matriz_distancias, acordes)
    fig_sh = graficar_shepard(res.embeddings, res.matriz_distancias)
    fig_sc.write_html(out_dir / f"scatter_{tag}.html")
    fig_hm.write_html(out_dir / f"heatmap_{tag}.html")
    fig_sh.write_html(out_dir / f"shepard_{tag}.html")

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="synthetic o db:<CONST_NAME> definida en config.py (p.ej., db:QUERY_CHORDS_WITH_NAME)")
    ap.add_argument("--limit", type=int, default=None, help="limitar número de acordes si se carga de DB")
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--reduction", default="MDS")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-harmonics", type=int, default=10, help="Harmonics for 'canonical'")
    ap.add_argument("--decay", type=float, default=0.8, help="Decay for 'canonical'")
    ap.add_argument("--nb-n-harmonics", type=int, default=6, help="Harmonics for notebook-style model")
    ap.add_argument("--nb-decay", type=float, default=0.88, help="Decay for notebook-style model")
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    if args.dataset == "synthetic":
        acordes = dataset_synthetic()
    elif args.dataset.startswith("db:"):
        query_name = args.dataset.split(":", 1)[1]
        acordes = dataset_db(query_name, args.limit)
    else:
        raise ValueError("--dataset debe ser 'synthetic' o 'db:<CONST_NAME>'")

    # Modelos
    repo_model = RepoSethares(config={})
    canon_model = CanonicalSethares(n_harmonics=args.n_harmonics, decay=args.decay)
    nb_model = NotebookSethares(n_harmonics=args.nb_n_harmonics, decay=args.nb_decay)

    res_repo = run_pipeline(acordes, "sethares_repo", repo_model, args.metric, args.reduction, out_dir, "repo")
    res_canon = run_pipeline(acordes, "sethares_canon", canon_model, args.metric, args.reduction, out_dir, "canon")
    res_nb = run_pipeline(acordes, "sethares_nb", nb_model, args.metric, args.reduction, out_dir, "nb")

    # Comparativas cruzadas
    # Correlación de distancias originales
    d_repo = squareform(res_repo.matriz_distancias)
    d_canon = squareform(res_canon.matriz_distancias)
    rho_rc, _ = spearmanr(d_repo, d_canon)
    d_nb = squareform(res_nb.matriz_distancias)
    rho_rn, _ = spearmanr(d_repo, d_nb)
    rho_cn, _ = spearmanr(d_canon, d_nb)

    # Procrustes para alinear embeddings y medir similitud geométrica
    try:
        _, _, disp_rc = procrustes(res_repo.embeddings, res_canon.embeddings)
        _, _, disp_rn = procrustes(res_repo.embeddings, res_nb.embeddings)
        _, _, disp_cn = procrustes(res_canon.embeddings, res_nb.embeddings)
        procrustes_disparities = (float(disp_rc), float(disp_rn), float(disp_cn))
    except Exception:
        procrustes_disparities = None

    with (out_dir / "compare_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Metric: {args.metric}, Reduction: {args.reduction}\n")
        f.write(f"Canonical params: n_harmonics={args.n_harmonics}, decay={args.decay}\n")
        f.write(f"Spearman rho(repo, canon) = {rho_rc:.4f}\n")
        f.write(f"Spearman rho(repo, nb)    = {rho_rn:.4f}\n")
        f.write(f"Spearman rho(canon, nb)  = {rho_cn:.4f}\n")
        if procrustes_disparities is not None:
            drc, drn, dcn = procrustes_disparities
            f.write(f"Procrustes disparity repo-canon = {drc:.6f}\n")
            f.write(f"Procrustes disparity repo-nb    = {drn:.6f}\n")
            f.write(f"Procrustes disparity canon-nb   = {dcn:.6f}\n")

    print("Comparación lista. Artefactos en:", out_dir)


if __name__ == "__main__":
    main()
