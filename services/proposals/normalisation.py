"""Normalisation strategies for roughness histograms."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


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


def preprocess_per_class(
    hist: np.ndarray, counts: np.ndarray, alpha: float = 1.0, **_
) -> Tuple[np.ndarray, np.ndarray]:
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


def preprocess_divide_mminus1(
    hist: np.ndarray, counts: np.ndarray, **_
) -> Tuple[np.ndarray, np.ndarray]:
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


__all__ = [
    "l1_normalize",
    "preprocess_simplex",
    "preprocess_simplex_sqrt",
    "preprocess_simplex_smooth",
    "preprocess_per_class",
    "preprocess_global_pairs",
    "preprocess_divide_mminus1",
    "preprocess_identity",
]
