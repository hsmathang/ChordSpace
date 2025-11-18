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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import os
def _format_exp(val: float) -> str:
    return f"{val:.2f}".rstrip("0").rstrip(".")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform, jensenshannon
from sklearn.manifold import MDS, TSNE, Isomap
try:
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None  # UMAP opcional

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
from lab import kruskal_stress_1
from metrics import (
    compute_continuity,
    compute_knn_recall,
    compute_rank_correlation,
    compute_trustworthiness,
)
from pre_process import (
    ChordAdapter,
    ModeloSetharesVec,
    get_chord_type_from_intervals,
)
from tools.query_registry import resolve_query_sql
from visualisations.proposals import build_scatter_payload

from tools.reporting import render_report_html
from tools.reporting.utils import (
    compute_rank,
    format_optional,
    format_rate,
    format_rate_with_std,
    format_seed_list,
    format_value_with_std,
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
    build_scatter_figure,
)
from tools.proposals_pipeline.population import ChordEntry, load_chords, stack_hist
from tools.proposals_pipeline.metrics import (
    metric_distance,
    parallel_worker_setup,
    run_scenario_task,
    aggregate_seed_results,
    mean_std,
)
from tools.proposals_pipeline.figures import (
    ColorSettings,
    HighlightSettings,
    generate_figures,
)

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
        "technical": "Normaliza el histograma \(H\) sobre el simplex: \(p_k = H_k / \sum_j H_j\). Las distancias se calculan sobre \(p\), lo que garantiza invariancia a cardinalidad.",
    },
    "simplex_sqrt": {
        "title": "Raíz + simplex",
        "casual": "Atenúa picos muy grandes antes de normalizar, dejando ver mejor las contribuciones secundarias.",
        "technical": "Aplica \(\sqrt{H}\) previo al paso al simplex para comprimir amplitudes y estabilizar métricas angulares.",
    },
    "simplex_smooth": {
        "title": "Simplex suavizado",
        "casual": "Difumina ligeramente la distribución para tolerar intervalos vecinos en la rueda cromática.",
        "technical": "Convoluciona \(p\) con un kernel Gaussiano circular (\(\sigma = 0.75\)) y renormaliza; evita discontinuidades mod 12.",
    },
    "perclass_alpha1": {
        "title": "Media por clase",
        "casual": "Promedia la rugosidad de cada tipo de díada sin importar cuántas veces se repita.",
        "technical": "Divide por la multiplicidad \(m_k\): \(H'_k = H_k / m_k\) y normaliza. Garantiza invariancia a duplicidades por clase.",
    },
    "perclass_alpha0_5": {
        "title": "Media por clase sublineal",
        "casual": "Reduce el peso de las repeticiones sin eliminarlas por completo.",
        "technical": "Usa \(H'_k = H_k / m_k^{0.5}\) como descuento sublineal para controlar redundancias fuertes.",
    },

    "perclass_alpha0_75": {
        "title": "Media por clase (α=0.75)",
        "casual": "Descuento sublineal moderado sobre repeticiones de díadas.",
        "technical": "Usa \(H'_k = H_k / m_k^{0.75}\) para atenuar la multiplicidad sin colapsarla como α=1.",
    },

    "perclass_alpha0_25": {
        "title": "Media por clase (α=0.25)",
        "casual": "Descuento leve, mantiene más la contribución de repeticiones.",
        "technical": "Usa \(H'_k = H_k / m_k^{0.25}\), apropiado cuando se desea penalización mínima por duplicidad.",
    },
    "global_pairs": {
        "title": "Media global por pares",
        "casual": "Escala el vector por el número total de díadas; conserva la forma pero reduce la magnitud.",
        "technical": "Normaliza por \(P = n(n-1)/2\): \(\bar{H} = H/P\). Sirve como baseline que preserva la distribución relativa.",
    },
    "divide_mminus1": {
        "title": "División por \(m-1\)",
        "casual": "Heurística que intenta penalizar la repetición de díadas restando una unidad.",
        "technical": "Escala por \(m_k - 1\) cuando \(m_k \ge 2\); se usa como control negativo frente a alternativas más formales.",
    },
    "identity": {
        "title": "Histograma original",
        "casual": "Usa el vector tal cual lo entrega el modelo de Sethares.",
        "technical": "Vector bruto \(H\); referencia para medir el efecto de cada normalización.",
    },
}


METRIC_INFO = {
    "cosine": {
        "title": "Cosine",
        "casual": "Mide el ángulo entre perfiles; importa la forma relativa más que la magnitud.",
        "technical": "\(d(u,v) = 1 - \frac{u\cdot v}{\|u\|\,\|v\|}\). Adecuado para distribuciones en el simplex.",
    },
    "js": {
        "title": "Jensen–Shannon",
        "casual": "Compara distribuciones como diferencias de información simétrica.",
        "technical": "\(d_{JS}(p,q) = \sqrt{\tfrac{1}{2} D_{KL}(p\|m) + \tfrac{1}{2} D_{KL}(q\|m)}\) con \(m = (p+q)/2\); métrica suave y finita.",
    },
    "hellinger": {
        "title": "Hellinger",
        "casual": "Distancia probabilística equilibrada, robusta a valores pequeños.",
        "technical": "\(d_H(p,q) = \tfrac{1}{\sqrt{2}}\|\sqrt{p}-\sqrt{q}\|_2\). Equivalente a la euclidiana en raíces.",
    },
    "euclidean": {
        "title": "Euclidiana",
        "casual": "Mide separaciones directas punto a punto.",
        "technical": "\(d(u,v) = \|u-v\|_2\). Con vectores normalizados refleja diferencias absolutas por clase.",
    },
    "l1": {
        "title": "Manhattan",
        "casual": "Suma diferencias absolutas por componente.",
        "technical": "\(d(u,v) = \|u-v\|_1\).",
    },
    "cityblock": {
        "title": "Manhattan",
        "casual": "Suma diferencias absolutas por componente.",
        "technical": "\(d(u,v) = \|u-v\|_1\).",
    },
    "manhattan": {
        "title": "Manhattan",
        "casual": "Suma diferencias absolutas por componente.",
        "technical": "\(d(u,v) = \|u-v\|_1\).",
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


def top_bins(dist_vector: np.ndarray, top_k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    if not np.any(dist_vector > 0):
        return np.array([], dtype=int), np.array([], dtype=float)
    idx_sorted = np.argsort(dist_vector)[::-1]
    idx_sorted = idx_sorted[:top_k]
    weights = dist_vector[idx_sorted]
    positive_mask = weights > 0
    idx_sorted = idx_sorted[positive_mask]
    weights = weights[positive_mask]
    return idx_sorted, weights


def evaluate_nn_hits(
    dist_matrix: np.ndarray,
    entries: List[ChordEntry],
    simplex: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    if not any(e.n_notes == 3 for e in entries):
        return None, None
    hits_top1: List[int] = []
    hits_top2: List[int] = []
    for idx, entry in enumerate(entries):
        if entry.n_notes != 3:
            continue
        row = dist_matrix[idx].copy()
        row[idx] = np.inf
        neighbor = int(np.argmin(row))
        if entries[neighbor].n_notes != 2:
            hits_top1.append(0)
            hits_top2.append(0)
            continue
        bins, weights = top_bins(simplex[idx], top_k=2)
        if bins.size == 0:
            hits_top1.append(0)
            hits_top2.append(0)
            continue
        target_bins = set(int(b) for b in bins)
        neighbor_bin = entries[neighbor].dyad_bin
        hit1 = 1 if neighbor_bin is not None and neighbor_bin == int(bins[0]) else 0
        hit_any = 1 if neighbor_bin is not None and neighbor_bin in target_bins else 0
        hits_top1.append(hit1)
        hits_top2.append(hit_any)
    if hits_top1:
        top1_rate = float(np.mean(hits_top1))
        top2_rate = float(np.mean(hits_top2))
    else:
        top1_rate = None
        top2_rate = None
    return top1_rate, top2_rate


def evaluate_mixture_error(simplex: np.ndarray, entries: List[ChordEntry]) -> Tuple[Optional[float], Optional[float]]:
    errors: List[float] = []
    for idx, entry in enumerate(entries):
        if entry.n_notes != 3:
            continue
        bins, weights = top_bins(simplex[idx], top_k=2)
        if bins.size == 0:
            continue
        weights = weights / weights.sum()
        mixture = np.zeros(12, dtype=float)
        for bin_idx, weight in zip(bins, weights):
            mixture[int(bin_idx)] = weight
        error = float(np.linalg.norm(simplex[idx] - mixture, ord=1))
        errors.append(error)
    if not errors:
        return None, None
    return float(np.mean(errors)), float(np.max(errors))


def summarise_embedding_metrics(
    X_original: np.ndarray,
    embedding: np.ndarray,
    dist_matrix: np.ndarray,
) -> Dict[str, Optional[float]]:
    try:
        trust = float(compute_trustworthiness(X_original, embedding))
    except Exception:
        trust = None
    try:
        cont = float(compute_continuity(X_original, embedding))
    except Exception:
        cont = None
    try:
        knn = float(compute_knn_recall(X_original, embedding))
    except Exception:
        knn = None
    try:
        rank_corr = float(compute_rank_correlation(X_original, embedding))
    except Exception:
        rank_corr = None
    try:
        stress = float(
            kruskal_stress_1(dist_matrix, squareform(pdist(embedding, metric="euclidean")))
        )
    except Exception:
        stress = None
    return {
        "trustworthiness": trust,
        "continuity": cont,
        "knn_recall": knn,
        "rank_corr": rank_corr,
        "stress": stress,
    }


def marker_style_for_cardinality(n_notes: int) -> Tuple[str, int]:
    return CARDINALITY_SYMBOLS.get(n_notes, DEFAULT_CARDINALITY_SYMBOL)


def group_entries_by_cardinality(entries: List[ChordEntry]) -> List[Tuple[int, List[int]]]:
    buckets: Dict[int, List[int]] = {}
    for idx, entry in enumerate(entries):
        buckets.setdefault(entry.n_notes, []).append(idx)
    return sorted(buckets.items(), key=lambda pair: pair[0])


def build_scatter_figure(
    embedding: np.ndarray,
    entries: List[ChordEntry],
    color_values: np.ndarray,
    pair_counts: np.ndarray,
    type_counts: np.ndarray,
    vectors: np.ndarray,
    adjusted_vectors: np.ndarray,
    title: str,
    *,
    is_proposal: bool = False,
    color_title: str = "Color",
    meta: Optional[Dict[str, Any]] = None,
    substitution_options: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    highlight_cfg = {
        "threshold": FAMILY_HIGHLIGHT_THRESHOLD,
        "size_scale": FAMILY_HIGHLIGHT_SIZE_SCALE,
        "size_delta": FAMILY_HIGHLIGHT_SIZE_DELTA,
        "selected_opacity": FAMILY_HIGHLIGHT_SELECTED_OPACITY,
        "fade_factor": FAMILY_HIGHLIGHT_UNSELECTED_OPACITY_FACTOR,
    }
    payload = build_scatter_payload(
        embedding=embedding,
        entries=entries,
        color_values=color_values,
        pair_counts=pair_counts,
        type_counts=type_counts,
        vectors=vectors,
        adjusted_vectors=adjusted_vectors,
        title=title,
        color_title=color_title,
        is_proposal=is_proposal,
        highlight=highlight_cfg,
        meta=meta,
        substitution_options=substitution_options,
    )
    return go.Figure(data=payload["data"], layout=payload["layout"])

def _format_vec(vec: np.ndarray, *, precision: int = 2, max_len: int = 12) -> str:
    slice_vec = vec[:max_len]
    values = ", ".join(f"{float(v):.{precision}f}" for v in slice_vec)
    if len(vec) > max_len:
        values += ", ..."
    return f"[{values}]"


def build_hover(
    entry: ChordEntry,
    vector_used: np.ndarray,
    vector_adjusted: np.ndarray,
    color_value: float,
    color_title: str,
    pair_count: int,
    type_count: int,
    *,
    is_proposal: bool,
    family_size: Optional[int] = None,
) -> str:
    """Hover rich text.

    Incluye la rugosidad normalizada (según la pestaña de color activa) y,
    para propuestas, también el total ajustado.
    """
    acorde = entry.acorde
    intervals = getattr(acorde, "intervals", [])
    tipo = getattr(acorde, "name", "Unknown")
    total = entry.total
    n = entry.n_notes
    identity_label = entry.identity_name if entry.is_named else "Desconocido"
    alias_line = ""
    if entry.identity_aliases:
        alias_line = f"Alias: {', '.join(entry.identity_aliases)}<br>"
    color_line = f"{color_title}: {float(color_value):.4f}<br>"
    pair_line = f"Pares totales (P): {pair_count}<br>"
    type_line = f"Tipos activos (PE): {type_count}<br>"
    family_line = ""
    has_family_id = entry.family_id is not None
    if has_family_id or entry.is_inversion:
        family_label = str(entry.family_id) if has_family_id else "—"
        role = "Inversión" if entry.is_inversion else "Acorde base"
        details: List[str] = []
        if family_size is not None and family_size > 0:
            details.append(f"miembros: {family_size}")
        if entry.is_inversion and entry.inversion_rotation is not None:
            details.append(f"rotación: {entry.inversion_rotation}")
        details_text = f" ({role}{', ' + ', '.join(details) if details else ''})" if role or details else ""
        family_line = f"Familia: {family_label}{details_text}<br>"
    if is_proposal:
        total_adj = float(np.sum(vector_adjusted))
        return (
            f"Acorde: {tipo}<br>"
            f"Notas: {n}<br>"
            f"Intervalos: {intervals}<br>"
            f"Identidad: {identity_label}<br>"
            f"{alias_line}"
            f"{family_line}"
            f"TotalRug (bruto): {total:.4f}<br>"
            f"TotalRug (ajustado): {total_adj:.4f}<br>"
            f"H bruto: {_format_vec(entry.hist)}<br>"
            f"H ajustado: {_format_vec(vector_adjusted)}<br>"
            f"{color_line}"
            f"{pair_line}"
            f"{type_line}"
        )
    else:
        return (
            f"Acorde: {tipo}<br>"
            f"Notas: {n}<br>"
            f"Intervalos: {intervals}<br>"
            f"Identidad: {identity_label}<br>"
            f"{alias_line}"
            f"{family_line}"
            f"TotalRug: {total:.4f}<br>"
            f"{color_line}"
            f"{pair_line}"
            f"{type_line}"
            f"H bruto: {_format_vec(entry.hist)}<br>"
        )


def build_hover_summary(
    entry: ChordEntry,
    family_size: Optional[int],
    color_value: float,
    color_title: str,
) -> str:
    acorde = entry.acorde
    name = getattr(acorde, "name", None)
    if not name or name == "Unknown":
        name = entry.identity_name or "Acorde"
    intervals = getattr(acorde, "intervals", [])
    interval_label = ""
    try:
        if intervals:
            interval_label = " " + "[" + ",".join(str(int(i)) for i in intervals) + "]"
    except Exception:
        interval_label = ""
    fam_label = family_size if family_size and family_size > 0 else 1
    return (
        f"{name}{interval_label} · {color_title}: {float(color_value):.2f} · "
        f"Familia: {fam_label}"
    )

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

    per_seed_records: List[Dict[str, object]] = []

    scenario_tasks: List[Dict[str, Any]] = []
    expected_order: List[str] = []
    distance_cache: Dict[Tuple[str, str], np.ndarray] = {}

    for scenario in scenarios:
        preproc_id = scenario["preproc_id"]
        if preproc_id not in dist_simplex_cache:
            preproc_func = scenario["preproc_func"]
            kwargs = scenario["preproc_kwargs"]
            X, simplex = preproc_func(hist, counts=counts, pairs=pairs, **kwargs)
            preproc_cache[preproc_id] = X
            dist_simplex_cache[preproc_id] = simplex
        key = (preproc_id, scenario["metric"])
        if key not in distance_cache:
            X = preproc_cache[preproc_id]
            simplex = dist_simplex_cache[preproc_id]
            try:
                dist_condensed = metric_distance(scenario["metric"], X, simplex)
            except ValueError as exc:
                print(f"[skip] {scenario['name']}: {exc}")
                continue
            distance_cache[key] = dist_condensed
        dist_condensed = distance_cache[key]

        for reduction in reductions:
            expected_order.append(f"{reduction}:{scenario['name']}")
        scenario_tasks.append(
            {
                "scenario": scenario,
                "reductions": list(reductions),
                "seed_list": list(seed_list),
                "deterministic": deterministic,
                "jobs": jobs,
                "mds_n_init": mds_n_init,
            }
        )

    figure_payloads: List[Dict[str, Any]] = []
    warnings: List[str] = []
    scenario_time_details: List[Tuple[str, float]] = []

    if scenario_tasks:
        context = {
            "entries": entries,
            "preproc_cache": preproc_cache,
            "dist_simplex_cache": dist_simplex_cache,
            "distance_cache": distance_cache,
        }
        use_parallel = len(scenario_tasks) > 1 and cpu_count > 1
        if use_parallel:
            max_workers = min(len(scenario_tasks), cpu_count)
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=parallel_worker_setup,
                initargs=(context,),
            ) as executor:
                futures = [executor.submit(run_scenario_task, task) for task in scenario_tasks]
                for fut in as_completed(futures):
                    res = fut.result()
                    warnings.extend(res["warnings"])
                    results.extend(res["results"])
                    per_seed_records.extend(res["per_seed_records"])
                    figure_payloads.extend(res["figure_payloads"])
                    scenario_time_details.extend(res.get("timings", []))
        else:
            parallel_worker_setup(context)
            for task in scenario_tasks:
                res = run_scenario_task(task)
                warnings.extend(res["warnings"])
                results.extend(res["results"])
                per_seed_records.extend(res["per_seed_records"])
                figure_payloads.extend(res["figure_payloads"])
                scenario_time_details.extend(res.get("timings", []))
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
                friendly = key.replace(":", " ▸ ", 1)
                print(f"  · {friendly}: {seconds:7.2f}")


def build_scenarios(
    proposals: Iterable[str],
    metrics: Iterable[str],
    *,
    include_identity: bool = True,
) -> List[Dict[str, object]]:
    scenarios: List[Dict[str, object]] = []
    metrics = list(metrics)
    for proposal in proposals:
        proposal = proposal.strip().lower()
        if proposal in {"simplex", "simplex_cosine"}:
            preproc_id = "simplex"
            preproc_func = PREPROCESSORS["simplex"][1]
            kwargs = PREPROCESSORS["simplex"][2]
            description = PREPROCESSORS["simplex"][0]
        elif proposal in {"simplexsqrt", "simplex_sqrt"}:
            preproc_id = "simplex_sqrt"
            preproc_func = PREPROCESSORS["simplex_sqrt"][1]
            kwargs = PREPROCESSORS["simplex_sqrt"][2]
            description = PREPROCESSORS["simplex_sqrt"][0]
        elif proposal in {"simplexsmooth", "simplex_smooth"}:
            preproc_id = "simplex_smooth"
            preproc_func = PREPROCESSORS["simplex_smooth"][1]
            kwargs = PREPROCESSORS["simplex_smooth"][2]
            description = PREPROCESSORS["simplex_smooth"][0]
        elif proposal == "perclass_alpha1":
            preproc_id = "perclass_alpha1"
            preproc_func = PREPROCESSORS["perclass_alpha1"][1]
            kwargs = PREPROCESSORS["perclass_alpha1"][2]
            description = PREPROCESSORS["perclass_alpha1"][0]
        elif proposal in {"perclass_alpha0_5", "perclass_alpha05"}:
            preproc_id = "perclass_alpha0_5"
            preproc_func = PREPROCESSORS["perclass_alpha0_5"][1]
            kwargs = PREPROCESSORS["perclass_alpha0_5"][2]
            description = PREPROCESSORS["perclass_alpha0_5"][0]
        elif proposal in {"perclass_alpha0_75", "perclass_alpha075", "perclass_alpha75"}:
            preproc_id = "perclass_alpha0_75"
            preproc_func = PREPROCESSORS["perclass_alpha0_75"][1]
            kwargs = PREPROCESSORS["perclass_alpha0_75"][2]
            description = PREPROCESSORS["perclass_alpha0_75"][0]
        elif proposal in {"perclass_alpha0_25", "perclass_alpha025", "perclass_alpha25"}:
            preproc_id = "perclass_alpha0_25"
            preproc_func = PREPROCESSORS["perclass_alpha0_25"][1]
            kwargs = PREPROCESSORS["perclass_alpha0_25"][2]
            description = PREPROCESSORS["perclass_alpha0_25"][0]
        elif proposal == "global_pairs":
            preproc_id = "global_pairs"
            preproc_func = PREPROCESSORS["global_pairs"][1]
            kwargs = PREPROCESSORS["global_pairs"][2]
            description = PREPROCESSORS["global_pairs"][0]
        elif proposal in {"divide_mminus1", "divide_m-1"}:
            preproc_id = "divide_mminus1"
            preproc_func = PREPROCESSORS["divide_mminus1"][1]
            kwargs = PREPROCESSORS["divide_mminus1"][2]
            description = PREPROCESSORS["divide_mminus1"][0]
        elif proposal in {"baseline_identity", "identity"}:
            preproc_id = "identity"
            preproc_func = PREPROCESSORS["identity"][1]
            kwargs = PREPROCESSORS["identity"][2]
            description = "Histograma original (control)"
        else:
            print(f"[warn] Propuesta desconocida: {proposal}. Se ignora.")
            continue

        for metric in metrics:
            metric = metric.strip().lower()
            scenarios.append(
                {
                    "name": f"{proposal} | {metric}",
                    "description": description,
                    "preproc_id": preproc_id,
                    "preproc_func": preproc_func,
                    "preproc_kwargs": kwargs,
                    "metric": metric,
                }
            )
    if include_identity:
        for metric in metrics:
            if not any(s["preproc_id"] == "identity" and s["metric"] == metric for s in scenarios):
                scenarios.append(
                    {
                        "name": f"identity | {metric}",
                        "description": "Histograma original (control)",
                        "preproc_id": "identity",
                        "preproc_func": PREPROCESSORS["identity"][1],
                        "preproc_kwargs": PREPROCESSORS["identity"][2],
                        "metric": metric,
                    }
                )

    return scenarios


if __name__ == "__main__":
    main()
