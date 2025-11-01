"""Immutable registries describing proposal components."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Tuple

import numpy as np

from .normalisation import (
    preprocess_divide_mminus1,
    preprocess_global_pairs,
    preprocess_identity,
    preprocess_per_class,
    preprocess_simplex,
    preprocess_simplex_smooth,
    preprocess_simplex_sqrt,
)


@dataclass(frozen=True)
class PreprocessorSpec:
    """Descriptor for a normalisation strategy."""

    label: str
    function: Callable[..., Tuple[np.ndarray, np.ndarray]]
    params: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


def _freeze_nested(data: Mapping[str, Mapping[str, str]]) -> Mapping[str, Mapping[str, str]]:
    return MappingProxyType({key: MappingProxyType(dict(value)) for key, value in data.items()})


PROPOSAL_INFO: Mapping[str, Mapping[str, str]] = _freeze_nested(
    {
        "simplex": {
            "title": "Simplex (distribución)",
            "casual": "Reparte la rugosidad entre las 12 clases de intervalo para identificar qué mezcla de díadas caracteriza al acorde.",
            "technical": r"Normaliza el histograma \(H\) sobre el simplex: \(p_k = H_k / \sum_j H_j\). Las distancias se calculan sobre \(p\), lo que garantiza invariancia a cardinalidad.",
        },
        "simplex_sqrt": {
            "title": "Raíz + simplex",
            "casual": "Atenúa picos muy grandes antes de normalizar, dejando ver mejor las contribuciones secundarias.",
            "technical": r"Aplica \(\sqrt{H}\) previo al paso al simplex para comprimir amplitudes y estabilizar métricas angulares.",
        },
        "simplex_smooth": {
            "title": "Simplex suavizado",
            "casual": "Difumina ligeramente la distribución para tolerar intervalos vecinos en la rueda cromática.",
            "technical": r"Convoluciona \(p\) con un kernel Gaussiano circular (\(\sigma = 0.75\)) y renormaliza; evita discontinuidades mod 12.",
        },
        "perclass_alpha1": {
            "title": "Media por clase",
            "casual": "Promedia la rugosidad de cada tipo de díada sin importar cuántas veces se repita.",
            "technical": r"Divide por la multiplicidad \(m_k\): \(H'_k = H_k / m_k\) y normaliza. Garantiza invariancia a duplicidades por clase.",
        },
        "perclass_alpha0_5": {
            "title": "Media por clase sublineal",
            "casual": "Reduce el peso de las repeticiones sin eliminarlas por completo.",
            "technical": r"Usa \(H'_k = H_k / m_k^{0.5}\) como descuento sublineal para controlar redundancias fuertes.",
        },
        "perclass_alpha0_75": {
            "title": "Media por clase (α=0.75)",
            "casual": "Descuento sublineal moderado sobre repeticiones de díadas.",
            "technical": r"Usa \(H'_k = H_k / m_k^{0.75}\) para atenuar la multiplicidad sin colapsarla como α=1.",
        },
        "perclass_alpha0_25": {
            "title": "Media por clase (α=0.25)",
            "casual": "Descuento leve, mantiene más la contribución de repeticiones.",
            "technical": r"Usa \(H'_k = H_k / m_k^{0.25}\), apropiado cuando se desea penalización mínima por duplicidad.",
        },
        "global_pairs": {
            "title": "Media global por pares",
            "casual": "Escala el vector por el número total de díadas; conserva la forma pero reduce la magnitud.",
            "technical": r"Normaliza por \(P = n(n-1)/2\): \(\bar{H} = H/P\). Sirve como baseline que preserva la distribución relativa.",
        },
        "divide_mminus1": {
            "title": r"División por \(m-1\)",
            "casual": "Heurística que intenta penalizar la repetición de díadas restando una unidad.",
            "technical": r"Escala por \(m_k - 1\) cuando \(m_k \ge 2\); se usa como control negativo frente a alternativas más formales.",
        },
        "identity": {
            "title": "Histograma original",
            "casual": "Usa el vector tal cual lo entrega el modelo de Sethares.",
            "technical": r"Vector bruto \(H\); referencia para medir el efecto de cada normalización.",
        },
    }
)


METRIC_INFO: Mapping[str, Mapping[str, str]] = _freeze_nested(
    {
        "cosine": {
            "title": "Cosine",
            "casual": "Mide el ángulo entre perfiles; importa la forma relativa más que la magnitud.",
            "technical": r"\(d(u,v) = 1 - \frac{u\cdot v}{\|u\|\,\|v\|}\). Adecuado para distribuciones en el simplex.",
        },
        "js": {
            "title": "Jensen–Shannon",
            "casual": "Compara distribuciones como diferencias de información simétrica.",
            "technical": r"\(d_{JS}(p,q) = \sqrt{\tfrac{1}{2} D_{KL}(p\|m) + \tfrac{1}{2} D_{KL}(q\|m)}\) con \(m = (p+q)/2\); métrica suave y finita.",
        },
        "hellinger": {
            "title": "Hellinger",
            "casual": "Distancia probabilística equilibrada, robusta a valores pequeños.",
            "technical": r"\(d_H(p,q) = \tfrac{1}{\sqrt{2}}\|\sqrt{p}-\sqrt{q}\|_2\). Equivalente a la euclidiana en raíces.",
        },
        "euclidean": {
            "title": "Euclidiana",
            "casual": "Mide separaciones directas punto a punto.",
            "technical": r"\(d(u,v) = \|u-v\|_2\). Con vectores normalizados refleja diferencias absolutas por clase.",
        },
        "l1": {
            "title": "Manhattan",
            "casual": "Suma diferencias absolutas por componente.",
            "technical": r"\(d(u,v) = \|u-v\|_1\).",
        },
        "cityblock": {
            "title": "Manhattan",
            "casual": "Suma diferencias absolutas por componente.",
            "technical": r"\(d(u,v) = \|u-v\|_1\).",
        },
        "manhattan": {
            "title": "Manhattan",
            "casual": "Suma diferencias absolutas por componente.",
            "technical": r"\(d(u,v) = \|u-v\|_1\).",
        },
    }
)


_AVAILABLE_REDUCTIONS: Tuple[str, ...] = ("MDS", "UMAP", "TSNE", "ISOMAP")
AVAILABLE_REDUCTIONS: Tuple[str, ...] = tuple(_AVAILABLE_REDUCTIONS)


_PREPROCESSORS = {
    "simplex": PreprocessorSpec("Distribución simplex (H/sum)", preprocess_simplex, {}),
    "simplex_sqrt": PreprocessorSpec("Raíz + simplex (sqrt(H))", preprocess_simplex_sqrt, {}),
    "simplex_smooth": PreprocessorSpec(
        "Suavizado Gaussiano (σ=0.75) + simplex",
        preprocess_simplex_smooth,
        {"sigma": 0.75},
    ),
    "perclass_alpha1": PreprocessorSpec("Media por clase (H_k / m_k)", preprocess_per_class, {"alpha": 1.0}),
    "perclass_alpha0_5": PreprocessorSpec(
        "Media por clase exponente 0.5",
        preprocess_per_class,
        {"alpha": 0.5},
    ),
    "perclass_alpha0_75": PreprocessorSpec(
        "Media por clase exponente 0.75",
        preprocess_per_class,
        {"alpha": 0.75},
    ),
    "perclass_alpha0_25": PreprocessorSpec(
        "Media por clase exponente 0.25",
        preprocess_per_class,
        {"alpha": 0.25},
    ),
    "global_pairs": PreprocessorSpec("Media global por pares (H / P)", preprocess_global_pairs, {}),
    "divide_mminus1": PreprocessorSpec("División por (m-1)", preprocess_divide_mminus1, {}),
    "identity": PreprocessorSpec("Identidad (control)", preprocess_identity, {}),
}

PREPROCESSORS: Mapping[str, PreprocessorSpec] = MappingProxyType(dict(_PREPROCESSORS))


__all__ = [
    "AVAILABLE_REDUCTIONS",
    "METRIC_INFO",
    "PREPROCESSORS",
    "PROPOSAL_INFO",
    "PreprocessorSpec",
]
