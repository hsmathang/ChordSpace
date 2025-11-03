import math
import time
from itertools import combinations

import numpy as np

from pre_process import ModeloSetharesVec


def _time_execution(fn, repeats=3):
    best = math.inf
    total_value = None
    for _ in range(repeats):
        duration, value = fn()
        best = min(best, duration)
        total_value = value
    return best, total_value


def test_vector_model_reuses_harmonic_arrays_for_large_sets():
    """Simple profiling check ensuring cached harmonic arrays speed up large runs."""

    config = {"n_armonicos": 64, "decaimiento": 0.88}
    model = ModeloSetharesVec(config=config)

    fundamentals = np.linspace(110.0, 880.0, 24)
    pairs = list(combinations(fundamentals, 2))

    K_cached, A_cached = model._get_harmonic_arrays(config["n_armonicos"], config["decaimiento"])

    def run_cached():
        start = time.perf_counter()
        total = 0.0
        for f1, f2 in pairs:
            total += model._pair_total(f1, f2, K_cached, A_cached)
        return time.perf_counter() - start, total

    def run_uncached():
        start = time.perf_counter()
        total = 0.0
        for f1, f2 in pairs:
            K_legacy = np.arange(1, config["n_armonicos"] + 1, dtype=float)
            A_legacy = config["decaimiento"] ** (K_legacy - 1)
            total += model._pair_total(f1, f2, K_legacy, A_legacy)
        return time.perf_counter() - start, total

    cached_duration, cached_total = _time_execution(run_cached)
    uncached_duration, uncached_total = _time_execution(run_uncached)

    assert math.isclose(cached_total, uncached_total, rel_tol=1e-12)
    assert cached_duration < uncached_duration
