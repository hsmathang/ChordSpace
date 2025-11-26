"""
Comparative runner for roughness-vector normalization proposals.

Loads reference dyads and triads, computes Sethares 12-D roughness vectors,
applies different normalisation strategies, evaluates multiple distance metrics,
reduces to 2D and generates a single HTML report with visualisations and metrics.
"""

from __future__ import annotations
import argparse
import json
import time
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform, jensenshannon

# Proveer shim para 'dotenv' si el entorno trae un paquete incompatible
try:  # pragma: no cover
    import dotenv as _dotenv  # type: ignore
    if not hasattr(_dotenv, "load_dotenv"):
        import types as _types, sys as _sys
        _shim = _types.ModuleType("dotenv")
        def _ld(*_a, **_k):
            return False
        _shim.load_dotenv = _ld  # type: ignore[attr-defined]
        _sys.modules["dotenv"] = _shim
except Exception:
    pass

from config import (
    QUERY_DYADS_REFERENCE,
    QUERY_TRIADS_CORE,
    config_db,
)
from pre_process import (
    ChordAdapter,
    ModeloSetharesVec,
    get_chord_type_from_intervals,
)
from tools.query_registry import resolve_query_sql
from tools.reporting import render_report_html
from tools.reporting.utils import (
    compute_rank,
    format_seed_list,
)
from tools.proposals_pipeline.population import ChordEntry, load_chords, stack_hist, l1_normalize
from tools.proposals_pipeline.metrics import (
    metric_distance,
    parallel_worker_setup,
    run_scenario_task,
    compute_embeddings as pipeline_compute_embeddings,
    evaluate_nn_hits as pipeline_evaluate_nn_hits,
    evaluate_mixture_error as pipeline_evaluate_mixture_error,
    summarise_embedding_metrics as pipeline_summarise_embedding_metrics,
    aggregate_seed_results,
    mean_std,
)
from tools.proposals_pipeline.figures import (
    ColorSettings,
    HighlightSettings,
    generate_figures,
)
from tools.proposals_pipeline.experiment import build_scenarios
from tools.proposals_pipeline.runner import run_experiment

try:  # Prefer packaged executor
    from chordcodex.model import QueryExecutor  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from synth_tools import QueryExecutor  # type: ignore


EPS = 1e-12

# ======================
# Color: parámetros y utilidades
# ======================
# Esta sección centraliza TODA la lógica y parámetros de normalización de color
# usada en el reporte (pestañas por modo de color). Si quieres experimentar con:
#   - usar P (pares) o (P-1),
#   - usar N (número de notas) o (N-1),
#   - aplicar log(1+·) o no,
# haz los cambios aquí y el resto del código los reflejará automáticamente.

# --- Configuración compartida por los modos de color ---------------------------
# Las pestañas del reporte derivan de combinaciones sobre la rugosidad total.
# Controlando estas constantes puedes ajustar la normalización sin tocar el resto.

#   Total bruto (raw):
#       c = TotalRug
#   Total/Pares      -> c = TotalRug / (P - COLOR_PER_PAIR_SUBTRACT)
#   Total/Notas      -> c = TotalRug / (N - COLOR_PER_NOTE_SUBTRACT)
#   Total ajustado   -> c = sum(vector ajustado)
#   Total ajustado/Tipos -> divide por el número de clases activas (PE - COLOR_PER_EXISTING_SUBTRACT)
#   log(1+·)         -> se aplica después de cada modo correspondiente.

# Para "per_pair" y derivados (evita dividir por cero en díadas)
COLOR_PER_PAIR_SUBTRACT: float = 0.0  # divide por P (sin restar).

# Para "per_note" y derivados
COLOR_PER_NOTE_SUBTRACT: float = 0.0  # usa 1.0 para N-1, etc.

# Para "per_existing": PE = nº de clases con contribución > COLOR_EXISTING_THRESHOLD
COLOR_PER_EXISTING_SUBTRACT: float = 0.0
COLOR_EXISTING_THRESHOLD: float = 1e-6

# Exponentes opcionales (γ). Mantener en 1.0 para comportamiento lineal.
COLOR_DEN_EXPONENT: float = 1.0      # aplica a todos los denominadores.
COLOR_OUTPUT_EXPONENT: float = 1.0   # potencia antes de aplicar logs.

# Lista de exponentes a explorar en las pestañas de color (aplicados al denominador).
# Exponentes de 0.00 a 1.00 en pasos de 0.05
COLOR_EXPONENTS: List[float] = [i/20.0 for i in range(0, 21)]

FAMILY_HIGHLIGHT_THRESHOLD: int = 2000
FAMILY_HIGHLIGHT_SIZE_SCALE: float = 1.35
FAMILY_HIGHLIGHT_SIZE_DELTA: float = 3.0
FAMILY_HIGHLIGHT_SELECTED_OPACITY: float = 0.95
FAMILY_HIGHLIGHT_UNSELECTED_OPACITY_FACTOR: float = 0.25

def _safe_denominator(raw: np.ndarray, subtract: float = 0.0) -> np.ndarray:
    """Construye un denominador seguro: max(raw - subtract, 1.0).

    - Evita divisiones por cero o negativas cuando raw <= subtract.
    - Está vectorizado para rendimiento.
    """
    den = np.asarray(raw, dtype=float)
    den = den - float(subtract)
    den[den < 1.0] = 1.0
    return den

# Símbolos dinámicos según cardinalidad (número de notas)
CARDINALITY_SYMBOLS: Dict[int, Tuple[str, int]] = {
    2: ("circle", 16),
    3: ("diamond", 18),
    4: ("square", 18),
    5: ("triangle-up", 18),
    6: ("triangle-down", 18),
    7: ("hexagon", 18),
    8: ("star", 18),
    9: ("x", 18),
    10: ("cross", 18),
}
DEFAULT_CARDINALITY_SYMBOL: Tuple[str, int] = ("circle-open", 16)
NAMED_BORDER_WIDTH = 0.6

# Default SQL for seventh chords (catalog of common 7th qualities; one per quality/root)
SEVENTHS_DEFAULT_SQL = """
WITH seventh_catalog(quality, intervals) AS (
    VALUES
        ('Maj7', ARRAY[4,3,4]::integer[]),
        ('7',    ARRAY[4,3,3]::integer[]),
        ('m7',   ARRAY[3,4,3]::integer[]),
        ('m7b5', ARRAY[3,3,4]::integer[]),
        ('Dim7', ARRAY[3,3,3]::integer[]),
        ('AugMaj7', ARRAY[4,4,3]::integer[])
),
ranked AS (
    SELECT
        c.*, seventh_catalog.quality,
        c.notes[1] AS root,
        ROW_NUMBER() OVER (
            PARTITION BY seventh_catalog.quality, c.notes[1]
            ORDER BY c.octave, c.id
        ) AS rn
    FROM chords c
    JOIN seventh_catalog ON c.interval = seventh_catalog.intervals
    WHERE c.n = 4
)
SELECT * FROM ranked WHERE rn = 1 ORDER BY quality, root;
"""



PROPOSAL_INFO = {
    "simplex": {
        "title": "Simplex (distribución)",
        "casual": "Reparte la rugosidad entre las 12 clases de intervalo para identificar qué mezcla de díadas caracteriza al acorde.",
        "technical": "Normaliza el histograma (H) sobre el simplex: (p_k = H_k / sum_j H_j). Las distancias se calculan sobre (p), lo que garantiza invariancia a cardinalidad.",
    },
    "simplex_sqrt": {
        "title": "Raíz + simplex",
        "casual": "Atenúa picos muy grandes antes de normalizar, dejando ver mejor las contribuciones secundarias.",
        "technical": "Aplica (sqrt{H}) previo al paso al simplex para comprimir amplitudes y estabilizar métricas angulares.",
    },
    "simplex_smooth": {
        "title": "Simplex suavizado",
        "casual": "Difumina ligeramente la distribución para tolerar intervalos vecinos en la rueda cromática.",
        "technical": "Convoluciona (p) con un kernel Gaussiano circular ((sigma = 0.75)) y renormaliza; evita discontinuidades mod 12.",
    },
    "perclass_alpha1": {
        "title": "Media por clase",
        "casual": "Promedia la rugosidad de cada tipo de díada sin importar cuántas veces se repita.",
        "technical": "Divide por la multiplicidad (m_k): (H'_k = H_k / m_k) y normaliza. Garantiza invariancia a duplicidades por clase.",
    },
    "perclass_alpha0_5": {
        "title": "Media por clase sublineal",
        "casual": "Reduce el peso de las repeticiones sin eliminarlas por completo.",
        "technical": "Usa (H'_k = H_k / m_k^{0.5}) como descuento sublineal para controlar redundancias fuertes.",
    },

    "perclass_alpha0_75": {
        "title": "Media por clase (α=0.75)",
        "casual": "Descuento sublineal moderado sobre repeticiones de díadas.",
        "technical": "Usa (H'_k = H_k / m_k^{0.75}) para atenuar la multiplicidad sin colapsarla como α=1.",
    },

    "perclass_alpha0_25": {
        "title": "Media por clase (α=0.25)",
        "casual": "Descuento leve, mantiene más la contribución de repeticiones.",
        "technical": "Usa (H'_k = H_k / m_k^{0.25}), apropiado cuando se desea penalización mínima por duplicidad.",
    },
    "global_pairs": {
        "title": "Media global por pares",
        "casual": "Escala el vector por el número total de díadas; conserva la forma pero reduce la magnitud.",
        "technical": "Normaliza por (P = n(n-1)/2): (bar{H} = H/P). Sirve como baseline que preserva la distribución relativa.",
    },
    "divide_mminus1": {
        "title": "División por (m-1)",
        "casual": "Heurística que intenta penalizar la repetición de díadas restando una unidad.",
        "technical": "Escala por (m_k - 1) cuando (m_k ge 2); se usa como control negativo frente a alternativas más formales.",
    },
    "identity": {
        "title": "Histograma original",
        "casual": "Usa el vector tal cual lo entrega el modelo de Sethares.",
        "technical": "Vector bruto (H); referencia para medir el efecto de cada normalización.",
    },
}


METRIC_INFO = {
    "cosine": {
        "title": "Cosine",
        "casual": "Mide el ángulo entre perfiles; importa la forma relativa más que la magnitud.",
        "technical": "(d(u,v) = 1 - frac{ucdot v}{|u|,|v|}). Adecuado para distribuciones en el simplex.",
    },
    "js": {
        "title": "Jensen–Shannon",
        "casual": "Compara distribuciones como diferencias de información simétrica.",
        "technical": "(d_{JS}(p,q) = sqrt{tfrac{1}{2} D_{KL}(p|m) + tfrac{1}{2} D_{KL}(q|m)}) con (m = (p+q)/2); métrica suave y finita.",
    },
    "hellinger": {
        "title": "Hellinger",
        "casual": "Distancia probabilística equilibrada, robusta a valores pequeños.",
        "technical": "(d_H(p,q) = tfrac{1}{sqrt{2}}|sqrt{p}-sqrt{q}|_2). Equivalente a la euclidiana en raíces.",
    },
    "euclidean": {
        "title": "Euclidiana",
        "casual": "Mide separaciones directas punto a punto.",
        "technical": "(d(u,v) = |u-v|_2). Con vectores normalizados refleja diferencias absolutas por clase.",
    },
    "l1": {
        "title": "Manhattan",
        "casual": "Suma diferencias absolutas por componente.",
        "technical": "(d(u,v) = |u-v|_1).",
    },
    "cityblock": {
        "title": "Manhattan",
        "casual": "Suma diferencias absolutas por componente.",
        "technical": "(d(u,v) = |u-v|_1).",
    },
    "manhattan": {
        "title": "Manhattan",
        "casual": "Suma diferencias absolutas por componente.",
        "technical": "(d(u,v) = |u-v|_1).",
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare roughness normalisation proposals on dyads/triads."
    )
    parser.add_argument(
        "--dyads-query",
        default="QUERY_DYADS_REFERENCE",
        help="Config constant or SQL for dyads (default: QUERY_DYADS_REFERENCE).",
    )
    parser.add_argument(
        "--triads-query",
        default="QUERY_TRIADS_CORE",
        help="Config constant or SQL for triads (default: QUERY_TRIADS_CORE).",
    )
    parser.add_argument(
        "--sevenths-query",
        default=SEVENTHS_DEFAULT_SQL,
        help="Config constant or SQL for seventh chords (default: built-in catalog).",
    )
    parser.add_argument(
        "--population-json",
        default=None,
        help="Ruta a un archivo JSON (registros) con la población ya preparada. Si se especifica, se ignoran las consultas individuales.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["deterministic", "parallel"],
        default="deterministic",
        help="Modo de ejecución: determinista (semillas fijas) o paralelo (sin semilla, usa múltiples núcleos).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Número de procesos para las reducciones (usa -1 para todos los núcleos). Por defecto: 1 en modo determinista, -1 en paralelo.",
    )
    parser.add_argument(
        "--mds-n-init",
        type=int,
        default=None,
        help="Número de inicializaciones para MDS (default: 4 en modo determinista, 1 en paralelo).",
    )
    parser.add_argument(
        "--proposals",
        default=(
            "simplex, simplexsqrt, simplexsmooth, "
            "perclass_alpha1, perclass_alpha0_75, perclass_alpha0_5, perclass_alpha0_25, "
            "global_pairs, divide_mminus1, baseline_identity"
        ),
        help="Comma separated list of proposal identifiers to run (default set cubre los casos clave).",
    )
    parser.add_argument(
        "--metrics",
        default="cosine,js,hellinger,euclidean",
        help="Comma separated metrics to evaluate for compatible proposals.",
    )
    # Reducciones: permitir múltiples métodos (p.ej. MDS,UMAP)
    parser.add_argument(
        "--reductions",
        default="MDS",
        help="Lista separada por comas de métodos de reducción (p.ej. MDS,UMAP).",
    )
    # Compatibilidad hacia atrás
    parser.add_argument(
        "--reduction",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: outputs/compare_proposals/<timestamp>).",
    )
    parser.add_argument(
        "--sections",
        default="all",
        help="Sections to include in report (comma): scatter,heatmap,shepard,table,metadata,all. Default: all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (MDS initialisation).",
    )
    parser.add_argument(
        "--seeds",
        default="11,23,37,53,71",
        help="Lista separada por comas de semillas para repetir el experimento.",
    )
    parser.add_argument(
        "--color-mode",
        choices=["total", "per_pair", "log_total", "log_per_pair"],
        default="log_per_pair",
        help=(
            "Modo de color para el scatter: total bruto, por par, log(total) o log(total/par). "
            "Por defecto: log_per_pair (recomendado para poblaciones mixtas)."
        ),
    )
    parser.add_argument(
        "--disable-baseline-identity",
        action="store_true",
        help="Omite el escenario baseline identity (control) para acelerar la corrida.",
    )
    parser.add_argument(
        "--run-metadata",
        default=None,
        help="Ruta a un JSON con metadatos de generación de población (proporcionado por la GUI).",
    )
    return parser.parse_args()


def parse_seed_list(seeds_arg: str) -> List[int]:
    if not seeds_arg:
        return []
    seeds: List[int] = []
    for part in seeds_arg.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            seeds.append(int(part))
        except ValueError:
            raise SystemExit(f"Semilla inválida: '{part}'")
    return seeds


def preprocess_simplex(hist: np.ndarray, **_) -> Tuple[np.ndarray, np.ndarray]:
    dist = l1_normalize(hist.copy())
    return dist, dist


def preprocess_simplex_sqrt(hist: np.ndarray, **_) -> Tuple[np.ndarray, np.ndarray]:
    sqrt_h = np.sqrt(np.clip(hist, 0.0, None))
    dist = l1_normalize(sqrt_h)
    return dist, dist


def preprocess_simplex_smooth(hist: np.ndarray, sigma: float = 0.75, **_) -> Tuple[np.ndarray, np.ndarray]:
    base = l1_normalize(hist.copy())
    smoothed = np.array(
        [gaussian_filter1d(row, sigma=sigma, mode="wrap") for row in base], dtype=float
    )
    dist = l1_normalize(smoothed)
    return dist, dist


def preprocess_per_class(hist: np.ndarray, counts: np.ndarray, alpha: float = 1.0, **_) -> Tuple[np.ndarray, np.ndarray]:
    """Divide H_k por m_k^alpha sin aplicar normalización L1.

    - X (salida 1): vector 'adjusted' en escala original (para Euclidiana/L1/Cosine).
    - salida 2 ('simplex') se deja igual a 'adjusted' para que el pipeline
      pueda decidir si normaliza al usar métricas de distribución.
    """
    adjusted = hist.copy()
    for i in range(adjusted.shape[0]):
        divisor = np.power(np.clip(counts[i], 1.0, None), alpha)
        adjusted[i] = adjusted[i] / divisor
    adjusted = np.clip(adjusted, 0.0, None)
    return adjusted, adjusted


def preprocess_global_pairs(hist: np.ndarray, pairs: np.ndarray, **_) -> Tuple[np.ndarray, np.ndarray]:
    adjusted = hist / pairs[:, None]
    dist = l1_normalize(np.clip(adjusted, 0.0, None))
    return adjusted, dist


def preprocess_divide_mminus1(hist: np.ndarray, counts: np.ndarray, **_) -> Tuple[np.ndarray, np.ndarray]:
    """Heurística 'divide por (m-1)' para penalizar duplicidades.

    Para cada fila (acorde) y cada bin k:
      - Si m_k >= 2, divide H_k por (m_k - 1).
      - Si m_k < 2, deja H_k sin cambios (evita divisor 0).

    Retorna:
      - X = 'adjusted' (vector en escala original, útil para métricas vectoriales como euclidiana/L1),
      - simplex = L1-normalización de 'adjusted' (para métricas de distribución como JS/Hellinger).
    """
    adjusted = hist.copy()
    for i in range(adjusted.shape[0]):
        divisor = np.where(counts[i] >= 2.0, counts[i] - 1.0, 1.0)
        adjusted[i] = adjusted[i] / divisor
    adjusted = np.clip(adjusted, 0.0, None)
    dist = l1_normalize(adjusted)
    return adjusted, dist


def preprocess_identity(hist: np.ndarray, **_) -> Tuple[np.ndarray, np.ndarray]:
    dist = l1_normalize(hist.copy())
    return hist, dist


PREPROCESSORS: Dict[str, Tuple[str, Callable[..., Tuple[np.ndarray, np.ndarray]], Dict[str, float]]] = {
    "simplex": ("Distribución simplex (H/sum)", preprocess_simplex, {}),
    "simplex_sqrt": ("Raíz + simplex (sqrt(H))", preprocess_simplex_sqrt, {}),
    "simplex_smooth": ("Suavizado Gaussiano (σ=0.75) + simplex", preprocess_simplex_smooth, {"sigma": 0.75}),
    "perclass_alpha1": ("Media por clase (H_k / m_k)", preprocess_per_class, {"alpha": 1.0}),
    "perclass_alpha0_5": ("Media por clase exponente 0.5", preprocess_per_class, {"alpha": 0.5}),
    "perclass_alpha0_75": ("Media por clase exponente 0.75", preprocess_per_class, {"alpha": 0.75}),
    "perclass_alpha0_25": ("Media por clase exponente 0.25", preprocess_per_class, {"alpha": 0.25}),
    "global_pairs": ("Media global por pares (H / P)", preprocess_global_pairs, {}),
    "divide_mminus1": ("División por (m-1)", preprocess_divide_mminus1, {}),
    "identity": ("Identidad (control)", preprocess_identity, {}),
}


def metric_distance(metric: str, X: np.ndarray, dist_simplex: np.ndarray) -> np.ndarray:
    metric = metric.lower()
    if metric == "cosine":
        return pdist(X, metric="cosine")
    if metric == "js":
        # Asegurar distribuciones válidas (normalizar por fila en el par)
        def _js(u, v):
            su = float(np.sum(u))
            sv = float(np.sum(v))
            uu = (u / su) if su > 0 else u
            vv = (v / sv) if sv > 0 else v
            return jensenshannon(uu, vv, base=2.0)
        return pdist(dist_simplex, _js)
    if metric == "hellinger":
        # Normalizar por fila al vuelo
        def _norm(u):
            s = float(np.sum(u))
            return (u / s) if s > 0 else u
        root = np.sqrt(np.apply_along_axis(_norm, 1, dist_simplex))
        return pdist(root, metric="euclidean") / np.sqrt(2.0)
    if metric in {"euclidean", "l2"}:
        return pdist(X, metric="euclidean")
    if metric in {"l1", "cityblock", "manhattan"}:
        return pdist(X, metric="cityblock")
    raise ValueError(f"Métrica no soportada: {metric}")


AVAILABLE_REDUCTIONS = ("MDS", "UMAP", "TSNE", "ISOMAP")

_PARALLEL_CONTEXT: Dict[str, Any] | None = None
BASE_VECTOR_METRICS = {"cosine", "euclidean", "l1", "l2", "cityblock", "manhattan"}


def _parallel_worker_setup(context: Dict[str, Any]) -> None:
    """Mantiene compatibilidad con scripts antiguos."""
    parallel_worker_setup(context)


def _run_scenario_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Mantiene compatibilidad con scripts antiguos."""
    return run_scenario_task(task)


def _generate_figures(
    payloads: Sequence[Dict[str, Any]],
    entries: List[ChordEntry],
    totals: np.ndarray,
    pairs: np.ndarray,
    preproc_cache: Dict[str, np.ndarray],
    dist_simplex_cache: Dict[str, np.ndarray],
    distance_cache: Dict[Tuple[str, str], np.ndarray],
) -> List[Tuple[str, go.Figure]]:
    """Delegación directa al módulo `tools.proposals_pipeline.figures`."""
    color_settings = ColorSettings(
        per_pair_subtract=COLOR_PER_PAIR_SUBTRACT,
        per_note_subtract=COLOR_PER_NOTE_SUBTRACT,
        per_existing_subtract=COLOR_PER_EXISTING_SUBTRACT,
        existing_threshold=COLOR_EXISTING_THRESHOLD,
        denominator_exponent=COLOR_DEN_EXPONENT,
        output_exponent=COLOR_OUTPUT_EXPONENT,
        exponents=COLOR_EXPONENTS,
    )
    highlight_settings = HighlightSettings(
        threshold=FAMILY_HIGHLIGHT_THRESHOLD,
        size_scale=FAMILY_HIGHLIGHT_SIZE_SCALE,
        size_delta=FAMILY_HIGHLIGHT_SIZE_DELTA,
        selected_opacity=FAMILY_HIGHLIGHT_SELECTED_OPACITY,
        fade_factor=FAMILY_HIGHLIGHT_UNSELECTED_OPACITY_FACTOR,
    )
    return generate_figures(
        payloads,
        entries,
        totals,
        pairs,
        preproc_cache,
        dist_simplex_cache,
        distance_cache,
        color_settings=color_settings,
        highlight_settings=highlight_settings,
    )


def compute_embeddings(
    dist_condensed: np.ndarray,
    reduction: str,
    seed: int,
    base_matrix: Optional[np.ndarray] = None,
    *,
    n_jobs: Optional[int] = None,
    deterministic: bool = True,
    mds_n_init: Optional[int] = None,
) -> np.ndarray:
    """Wrapper para mantener la API pública original."""
    return pipeline_compute_embeddings(
        dist_condensed,
        reduction,
        seed,
        base_matrix=base_matrix,
        n_jobs=n_jobs,
        deterministic=deterministic,
        mds_n_init=mds_n_init,
    )


def evaluate_nn_hits(
    dist_matrix: np.ndarray,
    entries: List[ChordEntry],
    simplex: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """Wrapper para compatibilidad."""
    return pipeline_evaluate_nn_hits(dist_matrix, entries, simplex)


def evaluate_mixture_error(
    simplex: np.ndarray,
    entries: List[ChordEntry],
) -> Tuple[Optional[float], Optional[float]]:
    """Wrapper para compatibilidad."""
    return pipeline_evaluate_mixture_error(simplex, entries)


def summarise_embedding_metrics(
    X_original: np.ndarray,
    embedding: np.ndarray,
    dist_matrix: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Wrapper para compatibilidad."""
    return pipeline_summarise_embedding_metrics(X_original, embedding, dist_matrix)


class TimingRecorder:
    """Acumula marcas de tiempo consecutivas para reportar duraciones amigables."""

    def __init__(self) -> None:
        self._marks: List[Tuple[str, float]] = [("start", time.perf_counter())]

    def mark(self, label: str) -> None:
        self._marks.append((label, time.perf_counter()))

    def summary(self) -> List[Tuple[str, float]]:
        """Devuelve pares (etapa, duración en segundos) excluyendo la marca inicial."""
        if len(self._marks) < 2:
            return []
        out: List[Tuple[str, float]] = []
        for idx in range(1, len(self._marks)):
            label, stamp = self._marks[idx]
            _, prev_stamp = self._marks[idx - 1]
            out.append((label, stamp - prev_stamp))
        return out

    def total(self) -> float:
        if len(self._marks) < 2:
            return 0.0
        return self._marks[-1][1] - self._marks[0][1]


def extract_stat(row: Dict[str, object], key: str) -> Tuple[Optional[float], Optional[float]]:
    return row.get(f"{key}_mean"), row.get(f"{key}_std")


def ensure_output_dir(path: Optional[str]) -> Path:
    if path:
        out_dir = Path(path).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / "compare_proposals" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def build_sections(ranked_df: pd.DataFrame) -> List[Dict[str, object]]:
    sections: List[Dict[str, object]] = []
    for metric, group in ranked_df.groupby("metric"):
        group_sorted = group.sort_values("rank")
        baseline_df = group_sorted[group_sorted["preproc_id"] == "identity"]
        baseline_row = baseline_df.iloc[0].to_dict() if not baseline_df.empty else None
        proposal_rows = [row._asdict() if hasattr(row, "_asdict") else row for row in group_sorted[group_sorted["preproc_id"] != "identity"].to_dict("records")]
        metric_info = METRIC_INFO.get(
            metric,
            {
                "title": metric.upper(),
                "casual": "",
                "technical": "",
            },
        )
        sections.append(
            {
                "metric": metric,
                "metric_info": metric_info,
                "baseline": baseline_row,
                "proposals": proposal_rows,
            }
        )
    return sections


def main() -> None:
    args = parse_args()
    include_identity = not getattr(args, "disable_baseline_identity", False)
    run_metadata: Optional[Dict[str, Any]] = None
    if getattr(args, "run_metadata", None):
        try:
            run_metadata = json.loads(Path(args.run_metadata).read_text(encoding="utf-8"))
            print(f"[input] Metadata adicional: {args.run_metadata}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[warn] No se pudo leer run-metadata ({args.run_metadata}): {exc}")
    df_override: Optional[pd.DataFrame] = None
    if getattr(args, "population_json", None):
        df_override = pd.read_json(args.population_json, orient="records", lines=True)
        print(f"[input] Población cargada desde JSON: {args.population_json} ({len(df_override)} filas)")
    timer = TimingRecorder()

    entries = load_chords(
        args.dyads_query,
        args.triads_query,
        args.sevenths_query,
        df_override=df_override,
    )
    timer.mark("load_chords")
    hist, totals, counts, pairs, notes = stack_hist(entries)
    timer.mark("stack_hist")

    cpu_count = os.cpu_count() or 1
    deterministic = args.execution_mode != "parallel"
    jobs = args.n_jobs if args.n_jobs is not None else (1 if deterministic else -1)
    if deterministic and args.n_jobs not in (None, 1):
        print("[aviso] Modo determinista requiere n_jobs=1 para reproducibilidad; se forzará a 1.")
        jobs = 1
    mds_n_init = args.mds_n_init if args.mds_n_init is not None else (4 if deterministic else 1)
    mode_label = "determinista (semilla fija)" if deterministic else "paralelo (multi-núcleo)"
    jobs_label = jobs if jobs is not None else ("auto" if deterministic else "-1")
    print(f"[recursos] Núcleos detectados: {cpu_count}")
    print(f"[recursos] Modo de ejecución: {mode_label} · n_jobs={jobs_label} · MDS n_init={mds_n_init}")

    proposals_requested = [p.strip().lower() for p in args.proposals.split(",") if p.strip()]
    metrics_requested = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]

    scenarios = build_scenarios(
        proposals_requested,
        metrics_requested,
        PREPROCESSORS,
        include_identity=include_identity,
    )
    # Reducciones solicitadas (compatibilidad: --reduction gana si se pasa)
    if args.reduction:
        reductions = [args.reduction]
    else:
        reductions = [r.strip() for r in (args.reductions or "MDS").split(',') if r.strip()]
    reductions = [r.upper() for r in reductions if r.strip()]
    reductions = [r for r in reductions if r in AVAILABLE_REDUCTIONS]
    if not reductions:
        reductions = ["MDS"]
    results: List[Dict[str, object]] = []
    figures: List[Tuple[str, go.Figure]] = []

    dist_simplex_cache: Dict[str, np.ndarray] = {}
    preproc_cache: Dict[str, np.ndarray] = {}

    seed_list = parse_seed_list(args.seeds)
    if not seed_list:
        seed_list = [args.seed]

    run_result = run_experiment(
        entries=entries,
        hist=hist,
        counts=counts,
        pairs=pairs,
        seed_list=seed_list,
        reductions=reductions,
        scenarios=scenarios,
        deterministic=deterministic,
        jobs=jobs,
        mds_n_init=mds_n_init,
        cpu_count=cpu_count,
    )
    results = run_result["results"]
    per_seed_records: List[Dict[str, object]] = run_result["per_seed_records"]
    figure_payloads = run_result["figure_payloads"]
    warnings = run_result["warnings"]
    scenario_time_details = run_result["scenario_time_details"]
    expected_order = run_result["expected_order"]
    preproc_cache = run_result["preproc_cache"]
    dist_simplex_cache = run_result["dist_simplex_cache"]
    distance_cache = run_result["distance_cache"]
    timer.mark("scenarios")

    for msg in warnings:
        print(msg)

    order_map = {name: idx for idx, name in enumerate(expected_order)}
    seed_rank = {seed: idx for idx, seed in enumerate(seed_list)}
    results.sort(key=lambda row: order_map.get(row["scenario"], len(order_map)))
    def _seed_rank(row: Dict[str, Any]) -> int:
        value = row.get("seed")
        if value is None:
            return len(seed_rank)
        try:
            return seed_rank.get(int(value), len(seed_rank))
        except (TypeError, ValueError):
            return len(seed_rank)

    per_seed_records.sort(
        key=lambda row: (
            order_map.get(row["scenario"], len(order_map)),
            _seed_rank(row),
        )
    )
    figure_payloads.sort(key=lambda payload: order_map.get(payload["scenario"], len(order_map)))
    scenario_time_details.sort(key=lambda item: order_map.get(item[0], len(order_map)))

    figures = _generate_figures(
        figure_payloads,
        entries,
        totals,
        pairs,
        preproc_cache,
        dist_simplex_cache,
        distance_cache,
    )
    # Filtrar figuras según secciones para report_builder
    sec_arg = (args.sections or "all").strip().lower()
    sec_enabled = {key: True for key in ("scatter", "heatmap", "shepard", "table", "metadata")}
    if sec_arg and sec_arg != "all":
        parts = [p.strip() for p in sec_arg.split(",") if p.strip()]
        if parts:
            sec_enabled = {key: False for key in sec_enabled}
            for p in parts:
                if p in sec_enabled:
                    sec_enabled[p] = True
    timer.mark("figures")

    if not results:
        raise SystemExit("No se generaron resultados. Revisa propuestas y métricas.")

    output_dir = ensure_output_dir(args.output)
    metrics_df = pd.DataFrame(results)
    metrics_df["rank"] = compute_rank(metrics_df)
    metrics_df.sort_values(by=["rank"], inplace=True)

    metrics_csv_df = metrics_df.copy()
    metrics_csv_df["seeds"] = metrics_csv_df["seeds"].apply(format_seed_list)
    metrics_path = output_dir / "metrics.csv"
    metrics_csv_df.to_csv(metrics_path, index=False, float_format="%.6f")

    json_path = output_dir / "metrics.json"
    json_path.write_text(metrics_df.to_json(orient="records", indent=2), encoding="utf-8")

    # Secciones habilitadas (permite ocultar partes del reporte)

    report_path = output_dir / "report.html"
    # New report layout (tabs + centralized methods)
    render_report_html(
        metrics_df,
        figures,
        report_path,
        seed_list,
        run_metadata=run_metadata,
        metric_info=METRIC_INFO,
        highlight_threshold=FAMILY_HIGHLIGHT_THRESHOLD,
        sections_enabled=sec_enabled,
    )
    timer.mark("report")

    if per_seed_records:
        per_seed_df = pd.DataFrame(per_seed_records)
        per_seed_df.to_csv(output_dir / "metrics_by_seed.csv", index=False, float_format="%.6f")
        timer.mark("metrics_by_seed")

    print(f"[ok] Reporte generado en: {report_path}")
    # Emitir resumen temporal para consumo en la GUI/logs.
    if args.seeds or args.seed:
        total_time = timer.total()
        print("[timing] resumen por etapas (s):")
        elapsed = 0.0
        for label, seconds in timer.summary():
            elapsed += seconds
            print(f"  - {label:<14}: {seconds:6.2f}")
        print(f"  - total          : {total_time:6.2f}")
        if scenario_time_details:
            print("[timing] escenarios detallados (s):")
            for key, seconds in scenario_time_details:
                # Evitar caracteres no ASCII para consolas cp1252/legacy
                friendly = key.replace(":", " -> ", 1)
                print(f"  - {friendly}: {seconds:7.2f}")


if __name__ == "__main__":
    main()
