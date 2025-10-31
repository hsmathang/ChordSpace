"""
Comparative runner for roughness-vector normalization proposals.

Loads reference dyads and triads, computes Sethares 12-D roughness vectors,
applies different normalisation strategies, evaluates multiple distance metrics,
reduces to 2D and generates a single HTML report with visualisations and metrics.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import os
def _format_exp(val: float) -> str:
    return f"{val:.2f}".rstrip("0").rstrip(".")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
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


@dataclass
class ChordEntry:
    acorde: object  # pre_process.Acorde
    hist: np.ndarray
    total: float
    counts: np.ndarray
    total_pairs: float
    n_notes: int
    dyad_bin: Optional[int]
    identity_name: str
    identity_aliases: Tuple[str, ...]
    is_named: bool
    is_inversion: bool = False
    family_id: Optional[object] = None
    inversion_rotation: Optional[int] = None



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


def load_chords(
    dyads_query: str,
    triads_query: str,
    sevenths_query: Optional[str] = None,
    df_override: Optional[pd.DataFrame] = None,
) -> List[ChordEntry]:
    if df_override is not None:
        df_all = df_override.copy()
    else:
        executor = QueryExecutor(**config_db)
        frames: List[pd.DataFrame] = []
        for query in (dyads_query, triads_query, sevenths_query):
            if not query:
                continue
            sql = resolve_query_sql(query) if query.upper().startswith("QUERY_") else query
            frames.append(executor.as_pandas(sql))
        if not frames:
            raise SystemExit("No se proporcionaron consultas válidas ni población precombinada.")
        df_all = pd.concat(frames, ignore_index=True)

    has_family = "__family_id" in df_all.columns
    has_family_size = "__family_size" in df_all.columns
    has_inv_flag = "__inv_flag" in df_all.columns
    has_inv_source = "__inv_source_id" in df_all.columns
    has_inv_rotation = "__inv_rotation" in df_all.columns

    modelo = ModeloSetharesVec(config={})
    entries: List[ChordEntry] = []

    for _, row in df_all.iterrows():
        acorde = ChordAdapter.from_csv_row(row)
        identity_obj = get_chord_type_from_intervals(acorde.intervals, with_alias=True)
        identity_name = getattr(identity_obj, "name", str(identity_obj))
        identity_aliases = tuple(getattr(identity_obj, "aliases", ()))
        is_named = bool(identity_name and identity_name != "Unknown")
        hist, total = modelo.calcular(acorde)
        hist = np.asarray(hist, dtype=float)
        counts = compute_interval_counts(acorde.intervals)
        total_pairs = float(np.sum(counts))
        n_notes = len(acorde.intervals) + 1
        dyad_bin = determine_dyad_bin(acorde.intervals) if n_notes == 2 else None
        inv_flag = False
        family_id: Optional[object] = None
        inv_rotation: Optional[int] = None

        if has_family:
            raw_family = row.get("__family_id")
            if pd.notna(raw_family):
                try:
                    family_id = int(raw_family)
                except (TypeError, ValueError):
                    family_id = str(raw_family)

        if has_inv_flag:
            raw_flag = row.get("__inv_flag")
            inv_flag = bool(raw_flag) if pd.notna(raw_flag) else False

        if has_inv_source and family_id is None:
            raw_family = row.get("__inv_source_id")
            if pd.notna(raw_family):
                try:
                    family_id = int(raw_family)
                except (TypeError, ValueError):
                    family_id = str(raw_family)

        if family_id is None:
            raw_id = row.get("id")
            if pd.notna(raw_id):
                try:
                    family_id = int(raw_id)
                except (TypeError, ValueError):
                    family_id = str(raw_id)

        if has_inv_rotation:
            raw_rot = row.get("__inv_rotation")
            if pd.notna(raw_rot):
                try:
                    inv_rotation = int(raw_rot)
                except (TypeError, ValueError):
                    inv_rotation = None

        entries.append(
            ChordEntry(
                acorde=acorde,
                hist=hist,
                total=float(total),
                counts=counts,
                total_pairs=total_pairs if total_pairs > 0 else 1.0,
                n_notes=n_notes,
                dyad_bin=dyad_bin,
                identity_name=identity_name,
                identity_aliases=identity_aliases,
                is_named=is_named,
                is_inversion=inv_flag,
                family_id=family_id,
                inversion_rotation=inv_rotation,
            )
        )
    return entries


def compute_interval_counts(intervals: Sequence[int]) -> np.ndarray:
    """Count number of pairs per interval class using UI bin order."""
    semitonos = [0]
    for step in intervals:
        semitonos.append((semitonos[-1] + int(step)) % 12)
    counts = np.zeros(12, dtype=float)
    for i in range(len(semitonos) - 1):
        for j in range(i + 1, len(semitonos)):
            intervalo = (semitonos[j] - semitonos[i]) % 12
            bin_idx = (intervalo - 1) % 12
            counts[bin_idx] += 1.0
    return counts


def determine_dyad_bin(intervals: Sequence[int]) -> Optional[int]:
    if not intervals:
        return None
    intervalo = int(intervals[0]) % 12
    return (intervalo - 1) % 12


def stack_hist(entries: List[ChordEntry]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hist = np.stack([e.hist for e in entries], axis=0)
    totals = np.array([e.total for e in entries], dtype=float)
    counts = np.stack([e.counts for e in entries], axis=0)
    pairs = np.array([e.total_pairs for e in entries], dtype=float)
    notes = np.array([float(e.n_notes) for e in entries], dtype=float)
    return hist, totals, counts, pairs, notes


def l1_normalize(matrix: np.ndarray) -> np.ndarray:
    sums = np.sum(matrix, axis=1, keepdims=True)
    sums[np.isclose(sums, 0.0)] = 1.0
    return matrix / sums


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


def _parallel_worker_setup(context: Dict[str, Any]) -> None:
    """Initialise shared read-only context inside worker processes."""
    global _PARALLEL_CONTEXT
    _PARALLEL_CONTEXT = context


def _run_scenario_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a scenario (preproc + metric) across all requested reductions."""
    if _PARALLEL_CONTEXT is None:
        raise RuntimeError("Parallel context not initialised.")

    entries = _PARALLEL_CONTEXT["entries"]
    preproc_cache: Dict[str, np.ndarray] = _PARALLEL_CONTEXT["preproc_cache"]
    dist_simplex_cache: Dict[str, np.ndarray] = _PARALLEL_CONTEXT["dist_simplex_cache"]

    scenario = task["scenario"]
    reductions: Sequence[str] = task["reductions"]
    seed_list: Sequence[int] = task["seed_list"]
    deterministic: bool = task["deterministic"]
    jobs = task["jobs"]
    mds_n_init = task["mds_n_init"]

    scenario_name_base: str = scenario["name"]
    metric: str = scenario["metric"]
    preproc_id: str = scenario["preproc_id"]
    description: str = scenario["description"]

    X = np.asarray(preproc_cache[preproc_id], dtype=float)
    simplex = np.asarray(dist_simplex_cache[preproc_id], dtype=float)

    warnings: List[str] = []
    results: List[Dict[str, Any]] = []
    per_seed_records: List[Dict[str, Any]] = []
    figure_payloads: List[Dict[str, Any]] = []

    base_vector_metrics = {"cosine", "euclidean", "l1", "l2", "cityblock", "manhattan"}

    for reduction in reductions:
        try:
            dist_condensed = metric_distance(metric, X, simplex)
        except ValueError as exc:
            warnings.append(f"[skip] {scenario_name_base} · {reduction}: {exc}")
            continue

        dist_matrix = squareform(dist_condensed)
        base_matrix = X if metric in base_vector_metrics else simplex

        nn_top1, nn_top2 = evaluate_nn_hits(dist_matrix, entries, simplex)
        mix_mean, mix_max = evaluate_mixture_error(simplex, entries)

        seed_rows: List[Dict[str, Optional[float]]] = []
        figure_embedding: Optional[np.ndarray] = None
        figure_seed: Optional[int] = None

        for seed in seed_list:
            embedding = compute_embeddings(
                dist_condensed,
                reduction,
                seed,
                base_matrix=base_matrix,
                n_jobs=jobs,
                deterministic=deterministic,
                mds_n_init=mds_n_init,
            )
            metrics_summary = summarise_embedding_metrics(base_matrix, embedding, dist_matrix)
            row: Dict[str, Optional[float]] = {
                "scenario": f"{reduction}:{scenario_name_base}",
                "description": description,
                "metric": metric,
                "preproc_id": preproc_id,
                "seed": seed,
                "reduction": reduction,
                "nn_hit_top1": nn_top1,
                "nn_hit_top2": nn_top2,
                "mixture_l1_mean": mix_mean,
                "mixture_l1_max": mix_max,
                **metrics_summary,
            }
            seed_rows.append(row)
            if figure_embedding is None:
                figure_embedding = embedding
                figure_seed = seed

        if not seed_rows:
            continue

        summary = aggregate_seed_results(seed_rows, seed_list)
        summary.update(
            {
                "scenario": f"{reduction}:{scenario_name_base}",
                "description": description,
                "metric": metric,
                "preproc_id": preproc_id,
                "figure_seed": figure_seed,
                "reduction": reduction,
            }
        )
        results.append(summary)
        per_seed_records.extend(seed_rows)
        if figure_embedding is not None:
            figure_payloads.append(
                {
                    "scenario": f"{reduction}:{scenario_name_base}",
                    "preproc_id": preproc_id,
                    "metric": metric,
                    "description": description,
                    "reduction": reduction,
                    "figure_seed": figure_seed,
                    "embedding": figure_embedding,
                }
            )

    return {
        "warnings": warnings,
        "results": results,
        "per_seed_records": per_seed_records,
        "figure_payloads": figure_payloads,
    }


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
    reduction = (reduction or "MDS").upper()
    jobs = n_jobs if n_jobs is not None else (1 if deterministic else -1)
    rng = seed if deterministic else None
    if reduction == "MDS":
        dist_matrix = squareform(dist_condensed)
        reducer = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=rng,
            normalized_stress=False,
            n_jobs=jobs,
            n_init=mds_n_init if mds_n_init is not None else (4 if deterministic else 1),
        )
        embedding = reducer.fit_transform(dist_matrix)
        return embedding
    if reduction == "UMAP":
        if umap is None:
            raise ValueError("UMAP no está instalado. Añade 'umap-learn' al entorno.")
        # UMAP sobre matriz de distancias (precomputed) requiere transformar a afinidad
        # Estrategia: usar 1/(1+d) para similitud aproximada.
        dmat = squareform(dist_condensed)
        sim = 1.0 / (1.0 + np.asarray(dmat, dtype=float))
        reducer = umap.UMAP(
            n_components=2,
            metric="precomputed",
            random_state=rng,
            n_jobs=jobs if jobs is not None else 1,
        )
        # Para 'precomputed', umap espera distancias, no similitudes. Pasamos dmat directamente.
        embedding = reducer.fit_transform(dmat)
        return embedding
    if reduction == "TSNE":
        dist_matrix = squareform(dist_condensed)
        n = dist_matrix.shape[0]
        perplexity = max(5, min(30, (n - 1)))
        reducer = TSNE(
            n_components=2,
            metric="precomputed",
            perplexity=perplexity,
            random_state=rng,
            init="random",
            max_iter=1000,
            n_jobs=jobs if jobs is not None else None,
        )
        embedding = reducer.fit_transform(dist_matrix)
        return embedding
    if reduction == "ISOMAP":
        if base_matrix is None:
            raise ValueError("ISOMAP requiere la matriz base de características.")
        X = np.asarray(base_matrix, dtype=float)
        n_neighbors = min(10, max(3, X.shape[0] - 1))
        reducer = Isomap(
            n_neighbors=n_neighbors,
            n_components=2,
            n_jobs=jobs if jobs is not None else None,
        )
        embedding = reducer.fit_transform(X)
        return embedding
    raise ValueError(f"Reducción no soportada: {reduction}")


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
) -> go.Figure:
    x = embedding[:, 0]
    y = embedding[:, 1]
    color_values = np.asarray(color_values, dtype=float)
    cmin = float(np.min(color_values))
    cmax = float(np.max(color_values))

    fig = go.Figure()
    total_points = len(entries)

    family_tags: List[str] = []
    family_counts: Dict[str, int] = {}
    highlight_summary: Dict[str, object] = {
        "enabled": False,
        "threshold": FAMILY_HIGHLIGHT_THRESHOLD,
        "total_points": total_points,
        "families": 0,
        "size_scale": FAMILY_HIGHLIGHT_SIZE_SCALE,
        "size_delta": FAMILY_HIGHLIGHT_SIZE_DELTA,
        "selected_opacity": FAMILY_HIGHLIGHT_SELECTED_OPACITY,
        "fade_factor": FAMILY_HIGHLIGHT_UNSELECTED_OPACITY_FACTOR,
        "has_inversions": any(e.is_inversion for e in entries),
    }
    customdata_all: List[List[object]] = []

    if total_points:
        def _normalize_family(raw_value: Optional[object], idx: int) -> str:
            if raw_value is None:
                return f"__solo_{idx}"
            if isinstance(raw_value, float) and np.isnan(raw_value):
                return f"__solo_{idx}"
            return str(raw_value)

        for idx, entry in enumerate(entries):
            tag = _normalize_family(entry.family_id, idx)
            family_tags.append(tag)
            family_counts[tag] = family_counts.get(tag, 0) + 1

        families_with_links = sum(1 for count in family_counts.values() if count > 1)
        highlight_enabled = (
            total_points <= FAMILY_HIGHLIGHT_THRESHOLD and families_with_links > 0
        )
        highlight_summary.update(
            {
                "enabled": highlight_enabled,
                "families": int(families_with_links),
            }
        )
    else:
        highlight_enabled = False

    detail_texts: List[str] = []
    summary_texts: List[str] = []
    for idx in range(total_points):
        fam_size = family_counts.get(family_tags[idx], 1)
        detail_texts.append(
            build_hover(
                entries[idx],
                vectors[idx],
                adjusted_vectors[idx],
                color_values[idx],
                color_title,
                int(round(pair_counts[idx])),
                int(round(type_counts[idx])),
                is_proposal=is_proposal,
                family_size=fam_size,
            )
        )
        summary_texts.append(
            build_hover_summary(
                entries[idx],
                fam_size,
                color_values[idx],
                color_title,
            )
        )

    customdata_all = [
        [
            family_tags[i],
            1 if entries[i].is_inversion else 0,
            family_counts.get(family_tags[i], 1),
            summary_texts[i],
            detail_texts[i],
        ]
        for i in range(total_points)
    ]

    def _base_marker_params(count: int) -> Tuple[float, float]:
        # Tamaño máximo para datasets pequeños y mínimo para muy grandes.
        if count <= 40:
            return 20.0, 0.5
        if count >= 1000:
            return 6.0, 0.2
        # Interpolación lineal entre 40 y 1000
        frac = (count - 40) / (1000 - 40)
        size = 20.0 - frac * (20.0 - 6.0)
        opacity = 0.5 - frac * (0.5 - 0.2)
        return max(size, 4.0), max(min(opacity, 0.5), 0.2)

    def _highlight_markers(base_size: float, base_opacity: float) -> Tuple[Dict[str, object], Dict[str, object]]:
        selected_size = max(base_size * FAMILY_HIGHLIGHT_SIZE_SCALE, base_size + FAMILY_HIGHLIGHT_SIZE_DELTA)
        selected_opacity = min(1.0, max(FAMILY_HIGHLIGHT_SELECTED_OPACITY, base_opacity + 0.35))
        unselected_opacity = max(0.05, base_opacity * FAMILY_HIGHLIGHT_UNSELECTED_OPACITY_FACTOR)
        selected_marker: Dict[str, object] = {
            "size": selected_size,
            "opacity": selected_opacity,
        }
        unselected_marker: Dict[str, object] = {
            "opacity": unselected_opacity,
        }
        return selected_marker, unselected_marker

    def _symbol_for_cardinality(n: int) -> str:
        if n == 3:
            return "triangle-up"
        if n == 4:
            return "square"
        if n == 5:
            return "star"
        return "circle"

    def _size_for_cardinality(_n: int, base: float) -> float:
        return base

    base_size, base_opacity = _base_marker_params(total_points)

    named_groups = {
        "Diadas": [],
        "Triadas": [],
        "Séptimas": [],
        "Extensiones": [],
    }
    unnamed_groups: Dict[str, List[int]] = {
        "3 notas": [],
        "4 notas": [],
        "5 notas": [],
        "Más de 5 notas": [],
    }

    def _classify_named(entry: ChordEntry) -> Optional[str]:
        if not entry.is_named:
            return None
        n = entry.n_notes
        name = (entry.identity_name or "").lower()
        if n == 2:
            return "Diadas"
        if n == 3:
            return "Triadas"
        if n == 4 and "7" in name:
            return "Séptimas"
        return "Extensiones"

    for idx, entry in enumerate(entries):
        category = _classify_named(entry)
        if category is not None:
            named_groups[category].append(idx)
            continue
        if entry.n_notes == 3:
            unnamed_groups["3 notas"].append(idx)
        elif entry.n_notes == 4:
            unnamed_groups["4 notas"].append(idx)
        elif entry.n_notes == 5:
            unnamed_groups["5 notas"].append(idx)
        else:
            unnamed_groups["Más de 5 notas"].append(idx)

    named_symbol_map = {
        "Diadas": "diamond",
        "Triadas": "triangle-down",
        "Séptimas": "square",
        "Extensiones": "cross",
    }

    named_size = max(base_size + 2.5, base_size * 1.2, 8.0)
    named_opacity = min(0.9, base_opacity + 0.25)

    for label in ["Diadas", "Triadas", "Séptimas", "Extensiones"]:
        idxs = named_groups[label]
        if not idxs:
            continue
        base_marker = dict(
            symbol=named_symbol_map[label],
            size=named_size,
            color=color_values[idxs],
            colorscale="Turbo",
            cmin=cmin,
            cmax=cmax,
            coloraxis="coloraxis",
            opacity=named_opacity,
            line=dict(width=0),
        )
        trace_kwargs: Dict[str, object] = {
            "x": x[idxs],
            "y": y[idxs],
            "mode": "markers",
            "name": f"{label} ({len(idxs)})",
            "marker": base_marker,
            "text": [summary_texts[i] for i in idxs],
            "customdata": [customdata_all[i] for i in idxs],
            "hovertemplate": "%{text}<extra></extra>",
        }
        if highlight_enabled:
            selected_marker, unselected_marker = _highlight_markers(named_size, named_opacity)
            trace_kwargs["selected"] = {"marker": selected_marker}
            trace_kwargs["unselected"] = {"marker": unselected_marker}
        fig.add_trace(go.Scatter(**trace_kwargs))

    unnamed_symbol_map = {
        "3 notas": "triangle-up",
        "4 notas": "x",
        "5 notas": "star",
        "Más de 5 notas": "circle",
    }

    for label in ["3 notas", "4 notas", "5 notas", "Más de 5 notas"]:
        idxs = unnamed_groups[label]
        if not idxs:
            continue
        unnamed_size = _size_for_cardinality(0, base_size)
        base_marker = dict(
            symbol=unnamed_symbol_map[label],
            size=unnamed_size,
            color=color_values[idxs],
            colorscale="Turbo",
            cmin=cmin,
            cmax=cmax,
            coloraxis="coloraxis",
            opacity=base_opacity,
            line=dict(width=0),
        )
        trace_kwargs = {
            "x": x[idxs],
            "y": y[idxs],
            "mode": "markers",
            "name": f"{label} ({len(idxs)})",
            "marker": base_marker,
            "text": [summary_texts[i] for i in idxs],
            "customdata": [customdata_all[i] for i in idxs],
            "hovertemplate": "%{text}<extra></extra>",
        }
        if highlight_enabled:
            selected_marker, unselected_marker = _highlight_markers(unnamed_size, base_opacity)
            trace_kwargs["selected"] = {"marker": selected_marker}
            trace_kwargs["unselected"] = {"marker": unselected_marker}
        fig.add_trace(go.Scatter(**trace_kwargs))

    fig.update_layout(
        title=title,
        width=640,
        height=420,
        plot_bgcolor="white",
        margin=dict(l=40, r=200, t=64, b=42),
        legend=dict(
            orientation="v",
            x=1.22,  # más a la derecha del colorbar
            y=0.5,
            xanchor="left",
            yanchor="middle",
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="#ccc",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        coloraxis=dict(
            colorscale="Turbo",
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(title=color_title, thickness=14, len=0.75, x=1.08),
        ),
        meta={"familyHighlight": highlight_summary},
    )
    return fig


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

def format_rate(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def format_optional(value: Optional[float]) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value:.4f}"


def format_rate_with_std(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None or np.isnan(mean):
        return "n/a"
    text = f"{100.0 * mean:.1f}%"
    if std is not None and not np.isnan(std) and std > 0:
        text += f" ± {100.0 * std:.1f}%"
    return text


def format_value_with_std(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None or np.isnan(mean):
        return "n/a"
    text = f"{mean:.4f}"
    if std is not None and not np.isnan(std) and std > 0:
        text += f" ± {std:.4f}"
    return text


def format_seed_list(seeds: Optional[Sequence[int]]) -> str:
    if not seeds:
        return "-"
    return ", ".join(str(s) for s in seeds)


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

def build_report_html_v2(
    metrics_df: pd.DataFrame,
    figures: List[Tuple[str, go.Figure]],
    output_path: Path,
    seeds: Sequence[int],
) -> None:
    ranked_df = metrics_df.copy()
    ranked_df["rank"] = compute_rank(ranked_df)
    ranked_df.sort_values(by="rank", inplace=True)

    display_df = ranked_df[
        [
            "rank",
            "scenario",
            "metric",
            "stress_mean",
            "stress_std",
            "trustworthiness_mean",
            "trustworthiness_std",
            "mixture_l1_mean_mean",
            "mixture_l1_mean_std",
            "seeds",
        ]
    ].copy()
    display_df["Stress"] = display_df.apply(
        lambda row: format_value_with_std(row["stress_mean"], row["stress_std"]), axis=1
    )
    display_df["Trustworthiness"] = display_df.apply(
        lambda row: format_value_with_std(row["trustworthiness_mean"], row["trustworthiness_std"]), axis=1
    )
    display_df["Mixture L1"] = display_df.apply(
        lambda row: format_value_with_std(row["mixture_l1_mean_mean"], row["mixture_l1_mean_std"]), axis=1
    )
    display_df["Semillas"] = display_df["seeds"].apply(format_seed_list)
    display_df = display_df.rename(
        columns={"rank": "Ranking", "scenario": "Escenario", "metric": "Métrica"}
    )
    display_df = display_df[["Ranking", "Escenario", "Métrica", "Stress", "Trustworthiness", "Mixture L1", "Semillas"]]
    table_html = display_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")

    figure_map = {title: fig for title, fig in figures}

    # Order metrics in a friendly order
    preferred_order = ["euclidean", "cosine", "js", "hellinger", "l1", "cityblock", "manhattan"]
    def _metric_key(m: str) -> Tuple[int, str]:
        return (preferred_order.index(m) if m in preferred_order else len(preferred_order), m)

    # Top-level tabs by reduction method
    reductions_present = list(dict.fromkeys(ranked_df.get("reduction", pd.Series(["MDS"])) .tolist()))

    include_js = True
    subtab_counter = 0

    def render_card(row: Dict[str, object], *, is_baseline: bool) -> str:
        nonlocal include_js, subtab_counter
        if row is None:
            return ""
        scenario = row.get("scenario", "")
        scenario_prefix = f"{scenario}||"

        card_highlight_info: Optional[Dict[str, object]] = None
        panel_entries: List[Tuple[Tuple[int, float], str, Optional[float], str, str]] = []
        for key, fig in figure_map.items():
            if not key.startswith(scenario_prefix):
                continue
            suffix = key[len(scenario_prefix):]
            if suffix == "raw_total":
                mode = "raw_total"
                exponent = None
                order = (0, 0.0)
            elif suffix.startswith("pair_exp_"):
                try:
                    exponent = int(suffix.rsplit("_", 1)[-1]) / 100.0
                except Exception:
                    continue
                mode = "pair_exp"
                order = (1, float(exponent))
            elif suffix.startswith("types_exp_"):
                try:
                    exponent = int(suffix.rsplit("_", 1)[-1]) / 100.0
                except Exception:
                    continue
                mode = "types_exp"
                order = (2, float(exponent))
            else:
                continue

            if fig is not None:
                if card_highlight_info is None or not bool(card_highlight_info.get("enabled")):
                    meta_obj = getattr(fig, "layout", None)
                    meta_dict = getattr(meta_obj, "meta", None) if meta_obj is not None else None
                    if isinstance(meta_dict, dict):
                        fh = meta_dict.get("familyHighlight")
                        if isinstance(fh, dict):
                            candidate_info = dict(fh)
                            if card_highlight_info is None or candidate_info.get("enabled"):
                                card_highlight_info = candidate_info
                inner_html = to_html(fig, include_plotlyjs="cdn" if include_js else False, full_html=False)
                include_js = False
            else:
                inner_html = "<p>Figura no disponible.</p>"
            panel_entries.append((order, mode, exponent, suffix, inner_html))

        if not panel_entries:
            return ""

        panel_entries.sort(key=lambda item: item[0])

        subtab_counter += 1
        sid = f"sub{subtab_counter}"

        mode_defaults: Dict[str, float] = {}
        for _, mode, exponent, _, _ in panel_entries:
            if exponent is None:
                mode_defaults.setdefault(mode, 0.0)
            else:
                if mode not in mode_defaults or abs(exponent - 1.0) < 1e-9:
                    mode_defaults[mode] = exponent

        if any(mode == "raw_total" for _, mode, _, _, _ in panel_entries):
            default_mode = "raw_total"
            default_exponent: Optional[float] = None
        else:
            default_mode = "pair_exp"
            default_exponent = 1.0
            found = False
            for _, mode, exponent, _, _ in panel_entries:
                if mode == "pair_exp" and exponent is not None and abs(exponent - 1.0) < 1e-9:
                    default_exponent = exponent
                    found = True
                    break
            if not found:
                _, default_mode, default_exponent, _, _ = panel_entries[0]

        MODE_LABELS = {
            "raw_total": "Rugosidad bruta",
            "pair_exp": "Normalización por pares",
            "types_exp": "Normalización por tipos",
        }

        mode_order: List[str] = []
        for _, mode, _, _, _ in panel_entries:
            if mode not in mode_order:
                mode_order.append(mode)

        options_html: List[str] = []
        for mode in mode_order:
            label = MODE_LABELS.get(mode, mode)
            selected_attr = " selected" if mode == default_mode else ""
            default_exp_val = mode_defaults.get(mode, 0.0)
            options_html.append(
                f"<option value='{mode}' data-default-exp='{default_exp_val:.2f}'{selected_attr}>{label}</option>"
            )

        panels_html: List[str] = []
        for _, mode, exponent, suffix, inner_html in panel_entries:
            target_id = f"{sid}-{suffix}"
            is_active = (mode == default_mode and (
                (exponent is None and default_exponent is None) or
                (exponent is not None and default_exponent is not None and abs(exponent - default_exponent) < 1e-9)
            ))
            exp_str = f"{exponent:.2f}" if exponent is not None else ""
            style_attr = " style='display:block;'" if is_active else " style='display:none;'"
            panels_html.append(
                f"<div id='{target_id}' class='subtab-panel{' active' if is_active else ''}' data-mode='{mode}' data-exp='{exp_str}'{style_attr}>"
                f"{inner_html}"
                "</div>"
            )

        stress = format_value_with_std(row.get("stress_mean"), row.get("stress_std"))
        trust = format_value_with_std(row.get("trustworthiness_mean"), row.get("trustworthiness_std"))
        mixture = format_value_with_std(row.get("mixture_l1_mean_mean"), row.get("mixture_l1_mean_std"))
        seeds_text = format_seed_list(row.get("seeds"))
        badge = "<span class='badge'>Base</span>" if is_baseline else ""
        header = f"<div class='card-header'><strong>{scenario}</strong> {badge}</div>"
        metrics_line = f"<div class='metrics-line'>Stress: {stress} · Trust: {trust} · Mixture L1: {mixture} · Semillas: {seeds_text}</div>"

        default_exp_str = f"{default_exponent:.2f}" if default_exponent is not None else "0.00"
        default_exp_display = f"{default_exponent:.2f}" if default_exponent is not None else "--"
        slider_disabled_attr = "" if default_mode in {"pair_exp", "types_exp"} else " disabled"

        controls = (
            f"<div class='color-controls' data-default-mode='{default_mode}' data-default-exp='{default_exp_str}'>"
            f"  <label>Color por:"
            f"    <select id='{sid}-mode'>"
            f"      {''.join(options_html)}"
            f"    </select>"
            f"  </label>"
            f"  <label> Exponente:"
            f"    <input id='{sid}-exp' type='range' min='0' max='1' step='0.05' value='{default_exp_str}'{slider_disabled_attr}/>"
            f"    <span id='{sid}-exp-val'>{default_exp_display}</span>"
            f"  </label>"
            f"</div>"
        )
        highlight_note_html = ""
        highlight_enabled_flag = bool(card_highlight_info and card_highlight_info.get("enabled"))
        card_attrs = [f"data-sid='{sid}'", f"data-family-highlight='{'1' if highlight_enabled_flag else '0'}'"]
        if card_highlight_info:
            families_detected = int(card_highlight_info.get("families", 0) or 0)
            threshold_limit = int(card_highlight_info.get("threshold", FAMILY_HIGHLIGHT_THRESHOLD) or FAMILY_HIGHLIGHT_THRESHOLD)
            card_attrs.append(f"data-highlight-threshold='{threshold_limit}'")
            card_attrs.append(f"data-highlight-families='{families_detected}'")
            if highlight_enabled_flag:
                if families_detected > 0:
                    highlight_note_html = (
                        f"<div class='highlight-note'>Resaltado de familias activo · {families_detected} familias detectadas (≤{threshold_limit} acordes).</div>"
                    )
                else:
                    highlight_note_html = (
                        f"<div class='highlight-note'>Resaltado de familias activo (≤{threshold_limit} acordes).</div>"
                    )
            elif families_detected > 0:
                highlight_note_html = (
                    f"<div class='highlight-note muted'>Se detectaron {families_detected} familias, pero el resaltado se desactiva para poblaciones mayores a {threshold_limit} acordes.</div>"
                )
        else:
            card_attrs.append(f"data-highlight-threshold='{FAMILY_HIGHLIGHT_THRESHOLD}'")
            card_attrs.append("data-highlight-families='0'")
        panels = f"<div class='subtab-panels'>{''.join(panels_html)}</div>"
        detail_panel = (
            "<div class='detail-panel' data-default-msg='Haz clic en un punto para ver el detalle completo.'>"
            "Haz clic en un punto para ver el detalle completo."
            "</div>"
        )
        card_attrs_str = " ".join(card_attrs)
        return (
            f"<div class='plot-card' {card_attrs_str}>{header}{metrics_line}{controls}"
            f"{highlight_note_html}{detail_panel}{panels}</div>"
        )

    # Build nested tabs: first by reduction, then by metric
    outer_headers: List[str] = []
    outer_bodies: List[str] = []
    for ridx, reduction in enumerate(reductions_present):
        outer_headers.append(f"<li class='tab{' active' if ridx==0 else ''}' data-target='tab-red-{ridx}'>{reduction}</li>")
        red_group = ranked_df[ranked_df.get("reduction").fillna("MDS") == reduction]
        # metric-level tabs inside
        metrics_present = list(dict.fromkeys(red_group["metric"].tolist()))
        metrics_sorted = sorted(metrics_present, key=_metric_key)
        inner_headers: List[str] = []
        inner_bodies: List[str] = []
        for midx, metric in enumerate(metrics_sorted):
            metric_info = METRIC_INFO.get(metric, {"title": metric.upper(), "casual": "", "technical": ""})
            inner_headers.append(f"<li class='subtab{' active' if midx==0 else ''}' data-target='tab-{ridx}-{metric}'>" + metric_info['title'] + "</li>")
            group = red_group[red_group["metric"] == metric].sort_values("rank")
            baseline_df = group[group["preproc_id"] == "identity"]
            baseline_row = baseline_df.iloc[0].to_dict() if not baseline_df.empty else None
            proposal_rows = [row._asdict() if hasattr(row, "_asdict") else row for row in group[group["preproc_id"] != "identity"].to_dict("records")]
            body_parts: List[str] = [
                f"<div id='tab-{ridx}-{metric}' class='tab-panel{' active' if midx==0 else ''}'>",
                f"<div class='metric-intro'><p>{metric_info['casual']}</p><p><em>Detalle técnico:</em> {metric_info['technical']}</p></div>",
            ]
            if baseline_row is None and not proposal_rows:
                body_parts.append("<p>No se registraron resultados para esta métrica.</p>")
            elif not proposal_rows:
                base_card = render_card(baseline_row, is_baseline=True)
                body_parts.append(f"<div class='plot-grid'>{base_card}</div>")
            else:
                for pr in proposal_rows:
                    cards = []
                    base_card = render_card(baseline_row, is_baseline=True)
                    if base_card:
                        cards.append(base_card)
                    cards.append(render_card(pr, is_baseline=False))
                    body_parts.append(f"<div class='plot-grid'>{''.join(cards)}</div>")
            body_parts.append("</div>")
            inner_bodies.append("".join(body_parts))
        # assemble inner tabs
        inner_tabs = (
            f"<div class='subtabs'>"
            f"  <ul class='subtab-headers'>{''.join(inner_headers)}</ul>"
            f"  <div class='tab-panels'>{''.join(inner_bodies)}</div>"
            f"</div>"
        )
        outer_bodies.append(
            f"<div id='tab-red-{ridx}' class='tab-panel{' active' if ridx==0 else ''}'>" + inner_tabs + "</div>"
        )

    tabs_html = f"""
    <div class='tabs'>
      <ul class='tab-headers'>
        {''.join(outer_headers)}
      </ul>
      <div class='tab-panels'>
        {''.join(outer_bodies)}
      </div>
    </div>
    """

    script_js = """
  <script>
    (function() {
      const outerHeaders = document.querySelectorAll('.tab-headers li');
      const outerPanels = document.querySelectorAll('.tabs > .tab-panels > .tab-panel');

      // Inner tab switches
      document.querySelectorAll('.subtabs').forEach(block => {
        const sheaders = block.querySelectorAll('.subtab-headers .subtab');
        const spans = block.querySelectorAll('.tab-panels > .tab-panel');
        function sactivate(targetId) {
          sheaders.forEach(h => h.classList.toggle('active', h.dataset.target === targetId));
          spans.forEach(p => {
            const isMatch = p.id === targetId;
            p.classList.toggle('active', isMatch);
            p.style.display = isMatch ? 'block' : 'none';
          });
        }
        sheaders.forEach(h => h.addEventListener('click', () => sactivate(h.dataset.target)));
        if (sheaders.length) sactivate(sheaders[0].dataset.target);
      });

      function activateOuter(idx) {
        outerHeaders.forEach((h,i)=>h.classList.toggle('active', i===idx));
        outerPanels.forEach((p,i)=> {
          const isActive = i===idx;
          p.classList.toggle('active', isActive);
          p.style.display = isActive ? 'block' : 'none';
          if (isActive) {
            const firstSub = p.querySelector('.subtab-headers .subtab');
            if (firstSub) firstSub.click();
          }
        });
      }

      outerHeaders.forEach((h,i)=>h.addEventListener('click', ()=>activateOuter(i)));
      if (outerHeaders.length) {
        activateOuter(0);
      } else {
        outerPanels.forEach(p => {
          p.classList.add('active');
          p.style.display = 'block';
        });
      }

      // Color control por tarjeta
      document.querySelectorAll('.plot-card').forEach(card => {
        const sid = card.dataset.sid || '';
        const controls = card.querySelector('.color-controls');
        const panels = card.querySelectorAll('.subtab-panels .subtab-panel');
        const modeSel = card.querySelector(`#${sid}-mode`);
        const expSlider = card.querySelector(`#${sid}-exp`);
        const expVal = card.querySelector(`#${sid}-exp-val`);
        const defaultMode = controls ? controls.dataset.defaultMode || 'pair_exp' : 'pair_exp';
        const defaultExp = controls ? parseFloat(controls.dataset.defaultExp || '1') : 1;
        if (modeSel && !modeSel.value) modeSel.value = defaultMode;
        if (expSlider && !expSlider.value) expSlider.value = defaultExp.toFixed(2);

        function showPanel() {
          const mode = modeSel ? modeSel.value : defaultMode;
          let optionDefaultExp = defaultExp;
          if (modeSel) {
            const opt = modeSel.options[modeSel.selectedIndex];
            if (opt && opt.dataset.defaultExp) {
              const candidate = parseFloat(opt.dataset.defaultExp);
              if (Number.isFinite(candidate)) optionDefaultExp = candidate;
            }
          }
          const needsExp = (mode === 'pair_exp' || mode === 'types_exp');
          let exp = needsExp ? (expSlider ? parseFloat(expSlider.value) : optionDefaultExp) : optionDefaultExp;
          if (!Number.isFinite(exp)) exp = optionDefaultExp;
          const code = String(Math.round(exp * 100)).padStart(3, '0');
          const target = needsExp ? `${sid}-${mode}_${code}` : `${sid}-${mode}`;
          panels.forEach(p => {
            const isTarget = p.id === target;
            p.classList.toggle('active', isTarget);
            p.style.display = isTarget ? 'block' : 'none';
          });
          if (expSlider) {
            if (needsExp) {
              expSlider.removeAttribute('disabled');
              expSlider.value = exp.toFixed(2);
              if (expVal) expVal.textContent = exp.toFixed(2);
            } else {
              expSlider.setAttribute('disabled', 'disabled');
              expSlider.value = optionDefaultExp.toFixed(2);
              if (expVal) expVal.textContent = '--';
            }
          } else if (expVal) {
            expVal.textContent = needsExp ? exp.toFixed(2) : '--';
          }
        }

        if (modeSel) modeSel.addEventListener('change', () => showPanel());
        if (expSlider) {
          expSlider.addEventListener('input', () => showPanel());
          expSlider.addEventListener('change', () => showPanel());
        }
        showPanel();
      });

      function setupFamilyHighlight(gd) {
        if (!gd || gd.__familyHighlightBound) return;
        const info = gd.layout && gd.layout.meta && gd.layout.meta.familyHighlight;
        if (!info || !info.enabled) return;
        gd.__familyHighlightBound = true;
        let activeTag = null;

        function applySelection(tag) {
          if (tag === activeTag) return;
          activeTag = tag;
          gd.data.forEach((trace, traceIndex) => {
            const custom = trace.customdata || [];
            if (!custom.length) {
              Plotly.restyle(gd, {selectedpoints: [null]}, [traceIndex]);
              return;
            }
            const matches = [];
            if (tag) {
              const tagStr = String(tag);
              for (let i = 0; i < custom.length; i++) {
                const row = custom[i];
                if (!row) continue;
                if (String(row[0]) === tagStr) {
                  matches.push(i);
                }
              }
            }
            Plotly.restyle(gd, {selectedpoints: [matches.length ? matches : null]}, [traceIndex]);
          });
        }

        gd.on('plotly_hover', ev => {
          const pt = ev.points && ev.points[0];
          if (!pt || !pt.customdata) {
            applySelection(null);
            return;
          }
          const familySize = parseInt(pt.customdata[2], 10) || 0;
          if (familySize < 2) {
            applySelection(null);
            return;
          }
          applySelection(String(pt.customdata[0]));
        });

        gd.on('plotly_unhover', () => applySelection(null));
        gd.on('plotly_click', () => applySelection(null));
        applySelection(null);
      }

      function registerCardHighlight(card) {
        if (!card || card.dataset.familyHighlight !== '1') return;
        const figures = card.querySelectorAll('.js-plotly-plot');
        figures.forEach(gd => {
          const attach = () => {
            const info = gd.layout && gd.layout.meta && gd.layout.meta.familyHighlight;
            if (!info || !info.enabled) return;
            setupFamilyHighlight(gd);
          };
          if (gd.layout && gd.layout.meta) {
            attach();
          } else {
            const handler = () => {
              gd.removeListener('plotly_afterplot', handler);
              attach();
            };
            gd.on('plotly_afterplot', handler);
          }
        });
      }

      function registerCardDetail(card) {
        const detailPanel = card.querySelector('.detail-panel');
        if (!detailPanel) return;
        const defaultMsg = detailPanel.dataset.defaultMsg || 'Haz clic en un punto para ver el detalle completo.';
        detailPanel.innerHTML = defaultMsg;
        const figures = card.querySelectorAll('.js-plotly-plot');
        figures.forEach(gd => {
          const updatePanel = content => {
            detailPanel.innerHTML = content || defaultMsg;
          };
          gd.on('plotly_click', ev => {
            const pt = ev.points && ev.points[0];
            if (!pt || !pt.customdata || pt.customdata.length < 5) {
              updatePanel(defaultMsg);
              return;
            }
            updatePanel(pt.customdata[4]);
          });
          gd.on('plotly_doubleclick', () => updatePanel(defaultMsg));
        });
      }

      document.querySelectorAll('.plot-card').forEach(card => {
        registerCardHighlight(card);
        registerCardDetail(card);
      });
    })();
  </script>
"""

    html_content = f"""
<!DOCTYPE html>
<html lang='es'>
<head>
  <meta charset='utf-8'/>
  <title>Comparación de Propuestas de Rugosidad</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; }}
    h1 {{ margin: 0 0 8px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0 20px 0; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: center; }}
    .tabs {{ margin-top: 14px; }}
    .tab-headers {{ list-style: none; padding: 0; display: flex; gap: 10px; border-bottom: 1px solid #ddd; margin: 0 0 10px 0; }}
    .tab-headers li {{ padding: 8px 12px; cursor: pointer; border-radius: 8px 8px 0 0; background: #f6f7fb; }}
    .tab-headers li.active {{ background: #fff; border: 1px solid #ddd; border-bottom: none; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .metric-intro {{ background: #f7f9ff; padding: 10px 12px; border-radius: 8px; margin-bottom: 12px; }}
    .plot-grid {{ display: grid; grid-template-columns: repeat(2, minmax(360px, 1fr)); gap: 22px; }}
    .plot-card {{ background: #f8f9fb; padding: 12px; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
    .card-header {{ margin: 0 0 6px 0; font-size: 1.02rem; }}
    .metrics-line {{ font-size: 0.9rem; color: #444; margin: 0 0 8px 0; }}
    .badge {{ display: inline-block; background: #4458ff; color: white; font-size: 0.72rem; padding: 2px 6px; border-radius: 6px; margin-left: 6px; vertical-align: middle; }}
    /* Subtabs styling */
    .subtabs {{ margin-top: 6px; }}
    .subtab-headers {{ list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 8px 0; }}
    .subtab-headers .subtab {{ padding: 4px 8px; cursor: pointer; border-radius: 6px; background: #eef1ff; font-size: 0.86rem; }}
    .subtab-headers .subtab.active {{ background: #fff; border: 1px solid #ddd; }}
    .subtab-panels .subtab-panel {{ display: none; }}
    .subtab-panels .subtab-panel.active {{ display: block; }}
    .color-controls {{ margin: 8px 0 10px 0; display: flex; gap: 16px; align-items: center; }}
    .color-controls label {{ font-size: 0.9rem; color: #333; }}
    .color-controls input[disabled] {{ opacity: 0.5; cursor: not-allowed; }}
    .highlight-note {{ font-size: 0.82rem; color: #1f4c5c; margin: 6px 0 10px 0; }}
    .highlight-note.muted {{ color: #6a6f7a; font-style: italic; }}
    .detail-panel {{ margin: 8px 0 10px 0; padding: 10px 12px; background: #fffbe6; border: 1px solid #f0d98c; border-radius: 8px; font-size: 0.88rem; min-height: 56px; line-height: 1.35; }}
  </style>
</head>
<body>
  <h1>Comparación de Propuestas de Normalización de Rugosidad</h1>
  <h3>Resumen global</h3>
  {table_html}
  {tabs_html}
  {script_js}
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")

def build_report_html(
    metrics_df: pd.DataFrame,
    figures: List[Tuple[str, go.Figure]],
    output_path: Path,
    seeds: Sequence[int],
) -> None:
    ranked_df = metrics_df.copy()
    ranked_df["rank"] = compute_rank(ranked_df)
    ranked_df.sort_values(by="rank", inplace=True)
    display_df = ranked_df[
        [
            "rank",
            "scenario",
            "metric",
            "stress_mean",
            "stress_std",
            "trustworthiness_mean",
            "trustworthiness_std",
            "mixture_l1_mean_mean",
            "mixture_l1_mean_std",
            "seeds",
        ]
    ].copy()
    display_df["Stress"] = display_df.apply(
        lambda row: format_value_with_std(row["stress_mean"], row["stress_std"]), axis=1
    )
    display_df["Trustworthiness"] = display_df.apply(
        lambda row: format_value_with_std(row["trustworthiness_mean"], row["trustworthiness_std"]), axis=1
    )
    display_df["Mixture L1"] = display_df.apply(
        lambda row: format_value_with_std(row["mixture_l1_mean_mean"], row["mixture_l1_mean_std"]), axis=1
    )
    display_df["Semillas"] = display_df["seeds"].apply(format_seed_list)
    display_df = display_df.rename(
        columns={
            "rank": "Ranking",
            "scenario": "Escenario",
            "metric": "Métrica",
        }
    )
    display_df = display_df[
        ["Ranking", "Escenario", "Métrica", "Stress", "Trustworthiness", "Mixture L1", "Semillas"]
    ]
    table_html = display_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")

    figure_map = {title: fig for title, fig in figures}
    sections = build_sections(ranked_df)
    sections_html: List[str] = []
    include_js = True

    def render_card(row: Dict[str, object], *, is_baseline: bool) -> str:
        nonlocal include_js
        if row is None:
            return ""
        scenario = row.get("scenario", "")
        preproc_id = row.get("preproc_id", "identity")
        prop_info = PROPOSAL_INFO.get(
            preproc_id,
            {"title": preproc_id, "casual": "", "technical": ""},
        )
        fig = figure_map.get(scenario)
        if fig is not None:
            fig_html = to_html(fig, include_plotlyjs="cdn" if include_js else False, full_html=False)
            include_js = False
        else:
            fig_html = "<p>Figura no disponible.</p>"

        stress = format_value_with_std(row.get("stress_mean"), row.get("stress_std"))
        trust = format_value_with_std(row.get("trustworthiness_mean"), row.get("trustworthiness_std"))
        mixture = format_value_with_std(row.get("mixture_l1_mean_mean"), row.get("mixture_l1_mean_std"))
        seeds_text = format_seed_list(row.get("seeds"))
        badge = "<span class='badge'>Base</span>" if is_baseline else ""
        description_html = (
            f"<p><strong>{prop_info['title']}</strong> {badge}</p>"
            f"<p>{prop_info['casual']}</p>"
            f"<p><em>Detalle técnico:</em> {prop_info['technical']}</p>"
        )
        metrics_line = (
            f"<p class='metrics-line'>Stress: {stress} · Trust: {trust} · Mixture L1: {mixture} · Semillas: {seeds_text}</p>"
        )
        card_html = f"<div class='plot-card'>{description_html}{metrics_line}{fig_html}</div>"
        return card_html

    for section in sections:
        metric = section["metric"]
        metric_info = section["metric_info"]
        baseline_row = section["baseline"]
        proposals = section["proposals"] or ([] if baseline_row else [])

        header_html = (
            f"<section><h2>{metric_info['title']} ({metric.upper()})</h2>"
            f"<div class='metric-desc'><p>{metric_info['casual']}</p>"
            f"<p><em>Detalle técnico:</em> {metric_info['technical']}</p></div>"
        )
        section_body: List[str] = [header_html]

        if baseline_row is None and not proposals:
            section_body.append("<p>No se registraron resultados para esta métrica.</p>")
        elif not proposals:
            baseline_card = render_card(baseline_row, is_baseline=True)
            if baseline_card:
                section_body.append(f"<div class='plot-grid'>{baseline_card}</div>")
        else:
            for proposal_row in proposals:
                row_cards: List[str] = []
                baseline_card = render_card(baseline_row, is_baseline=True)
                if baseline_card:
                    row_cards.append(baseline_card)
                proposal_card = render_card(proposal_row, is_baseline=False)
                row_cards.append(proposal_card)
                section_body.append(f"<div class='plot-grid'>{''.join(row_cards)}</div>")

        section_body.append("</section>")
        sections_html.append("".join(section_body))

    seeds_text = format_seed_list(seeds)
    html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8"/>
    <title>Comparación de Propuestas de Rugosidad</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; margin-bottom: 30px; width: 100%; }}
        th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: center; }}
        h1 {{ margin-bottom: 10px; }}
        h2 {{ margin-top: 28px; }}
        section {{ margin-bottom: 40px; }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(360px, 1fr));
            gap: 28px;
        }}
        .plot-card {{
            background: #f8f9fb;
            padding: 16px;
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }}
        .plot-card p {{ margin: 0 0 8px 0; }}
        .metrics-line {{ font-size: 0.92rem; color: #444; }}
        .notation {{ background: #eef3ff; border-radius: 10px; padding: 14px 18px; margin-bottom: 24px; }}
        .notation ul {{ margin: 8px 0 0 20px; }}
        .metric-desc {{ background: #f0f4ff; padding: 12px 16px; border-radius: 10px; margin-bottom: 20px; }}
        .badge {{ display: inline-block; background: #4458ff; color: white; font-size: 0.7rem; padding: 2px 6px; border-radius: 6px; margin-left: 6px; vertical-align: middle; }}
    </style>
</head>
<body>
    <h1>Comparación de Propuestas de Normalización de Rugosidad</h1>
    <div class="notation">
        <p><strong>Notación</strong></p>
        <ul>
            <li>\(H_k\): contribución de rugosidad del acorde en la clase de intervalo \(k\) (mod 12).</li>
            <li>\(m_k\): número de díadas del acorde que caen en la clase \(k\).</li>
            <li>\(p_k = H_k / \sum_j H_j\): distribución simplex de las contribuciones.</li>
            <li>TotalRug: \(\sum_k H_k\), usado exclusivamente para colorear los puntos.</li>
        </ul>
        <p><strong>Métricas</strong>: el <em>Stress</em> mide la distorsión media del embebido MDS frente a las distancias originales; <em>Trustworthiness</em> comprueba si los vecinos preservan la geometría local. La métrica de mezcla L1 cuantifica cuán lejos está \(p\) de la combinación convexa de sus díadas prototipo.</p>
        <p><strong>Semillas evaluadas</strong>: {seeds_text}</p>
    </div>
    <h3>Métricas globales</h3>
    {table_html}
    {''.join(sections_html)}
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")


def compute_rank(df: pd.DataFrame) -> List[int]:
    tmp = df.copy()
    tmp["_stress"] = tmp["stress_mean"].fillna(np.inf)
    tmp["_trust"] = tmp["trustworthiness_mean"].fillna(-np.inf)
    ordered = tmp.sort_values(by=["_stress", "_trust"], ascending=[True, False])
    rank_map = {idx: rank for rank, idx in enumerate(ordered.index, start=1)}
    return [rank_map[idx] for idx in df.index]


def mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    clean = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return None, None
    arr = np.asarray(clean, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate_seed_results(seed_rows: List[Dict[str, Optional[float]]], seeds: Sequence[int]) -> Dict[str, object]:
    metrics_keys = [
        "nn_hit_top1",
        "nn_hit_top2",
        "mixture_l1_mean",
        "mixture_l1_max",
        "trustworthiness",
        "continuity",
        "knn_recall",
        "rank_corr",
        "stress",
    ]
    summary: Dict[str, object] = {"seeds": list(seeds)}
    for key in metrics_keys:
        values = [row.get(key) for row in seed_rows]
        mean_val, std_val = mean_std(values)
        summary[f"{key}_mean"] = mean_val
        summary[f"{key}_std"] = std_val
    return summary


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
    df_override: Optional[pd.DataFrame] = None
    if getattr(args, "population_json", None):
        df_override = pd.read_json(args.population_json, orient="records", lines=True)
        print(f"[input] Población cargada desde JSON: {args.population_json} ({len(df_override)} filas)")
    entries = load_chords(
        args.dyads_query,
        args.triads_query,
        args.sevenths_query,
        df_override=df_override,
    )
    hist, totals, counts, pairs, notes = stack_hist(entries)

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

    scenarios = build_scenarios(proposals_requested, metrics_requested)
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

    for scenario in scenarios:
        preproc_id = scenario["preproc_id"]
        if preproc_id not in dist_simplex_cache:
            preproc_func = scenario["preproc_func"]
            kwargs = scenario["preproc_kwargs"]
            X, simplex = preproc_func(hist, counts=counts, pairs=pairs, **kwargs)
            preproc_cache[preproc_id] = X
            dist_simplex_cache[preproc_id] = simplex
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

    if scenario_tasks:
        context = {
            "entries": entries,
            "preproc_cache": preproc_cache,
            "dist_simplex_cache": dist_simplex_cache,
        }
        use_parallel = len(scenario_tasks) > 1 and cpu_count > 1
        if use_parallel:
            max_workers = min(len(scenario_tasks), cpu_count)
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_parallel_worker_setup,
                initargs=(context,),
            ) as executor:
                futures = [executor.submit(_run_scenario_task, task) for task in scenario_tasks]
                for fut in as_completed(futures):
                    res = fut.result()
                    warnings.extend(res["warnings"])
                    results.extend(res["results"])
                    per_seed_records.extend(res["per_seed_records"])
                    figure_payloads.extend(res["figure_payloads"])
        else:
            _parallel_worker_setup(context)
            for task in scenario_tasks:
                res = _run_scenario_task(task)
                warnings.extend(res["warnings"])
                results.extend(res["results"])
                per_seed_records.extend(res["per_seed_records"])
                figure_payloads.extend(res["figure_payloads"])

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

    figures = []
    base_vector_metrics = {"cosine", "euclidean", "l1", "l2", "cityblock", "manhattan"}

    def _format_exp(val: float) -> str:
        return f"{val:.2f}".rstrip("0").rstrip(".")

    def _apply_color_mode(
        mode: str,
        exponent: Optional[float],
        totals_raw: np.ndarray,
        totals_adjusted: np.ndarray,
        pairs_arr: np.ndarray,
        types_arr: np.ndarray,
    ) -> Tuple[np.ndarray, str]:
        """Devuelve (valores normalizados, título de la barra)."""
        mode_lower = mode.lower()
        if mode_lower == "pair_exp":
            exp = exponent if exponent is not None else 1.0
            denom = _safe_denominator(pairs_arr, subtract=COLOR_PER_PAIR_SUBTRACT)
            denom = np.power(denom, exp)
            if not np.isclose(COLOR_DEN_EXPONENT, 1.0):
                denom = np.power(denom, COLOR_DEN_EXPONENT)
            vals = totals_raw / denom
            title = f"Total/Pares^{_format_exp(exp)}"
        elif mode_lower == "types_exp":
            exp = exponent if exponent is not None else 1.0
            denom = _safe_denominator(types_arr, subtract=COLOR_PER_EXISTING_SUBTRACT)
            denom = np.power(denom, exp)
            if not np.isclose(COLOR_DEN_EXPONENT, 1.0):
                denom = np.power(denom, COLOR_DEN_EXPONENT)
            vals = totals_adjusted / denom
            title = f"Total ajustado/Tipos^{_format_exp(exp)}"
        elif mode_lower == "raw_total":
            vals = totals_raw.copy()
            title = "Total bruto"
        else:
            raise ValueError(f"Modo de color no soportado: {mode}")

        if not np.isclose(COLOR_OUTPUT_EXPONENT, 1.0):
            vals = np.power(np.clip(vals, 0.0, None), COLOR_OUTPUT_EXPONENT)
        return vals, title

    for payload in figure_payloads:
        scenario_name = payload["scenario"]
        preproc_id = payload["preproc_id"]
        metric = payload["metric"]
        reduction = payload["reduction"]
        figure_seed = payload["figure_seed"]
        embedding = np.asarray(payload["embedding"], dtype=float)
        base_matrix = (
            np.asarray(preproc_cache[preproc_id], dtype=float)
            if metric in base_vector_metrics
            else np.asarray(dist_simplex_cache[preproc_id], dtype=float)
        )
        vectors_adjusted = np.asarray(preproc_cache[preproc_id], dtype=float)
        totals_adj = np.sum(vectors_adjusted, axis=1)
        existing_counts = np.sum(vectors_adjusted > COLOR_EXISTING_THRESHOLD, axis=1).astype(float)

        color_modes: List[Tuple[str, Optional[float]]] = []
        if preproc_id == "identity":
            color_modes.append(("raw_total", None))
        for exp in COLOR_EXPONENTS:
            color_modes.append(("pair_exp", exp))
            color_modes.append(("types_exp", exp))

        fig_title = f"{scenario_name} (seed {figure_seed})"
        for mode, exponent in color_modes:
            vals, ctitle = _apply_color_mode(
                mode,
                exponent,
                totals,
                totals_adj,
                pairs,
                existing_counts,
            )
            key = mode if exponent is None else f"{mode}_{int(round(exponent * 100)):03d}"
            fig = build_scatter_figure(
                embedding=embedding,
                entries=entries,
                color_values=vals,
                pair_counts=pairs,
                type_counts=existing_counts,
                vectors=base_matrix,
                adjusted_vectors=vectors_adjusted,
                title=fig_title,
                is_proposal=(preproc_id != "identity"),
                color_title=ctitle,
            )
            figures.append((f"{scenario_name}||{key}", fig))

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
    build_report_html_v2(metrics_df, figures, report_path, seed_list)

    if per_seed_records:
        per_seed_df = pd.DataFrame(per_seed_records)
        per_seed_df.to_csv(output_dir / "metrics_by_seed.csv", index=False, float_format="%.6f")

    print(f"[ok] Reporte generado en: {report_path}")


def build_scenarios(proposals: Iterable[str], metrics: Iterable[str]) -> List[Dict[str, object]]:
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
