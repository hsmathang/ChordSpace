"""
Compares all dimensionality reduction techniques on a single dataset.
"""

from __future__ import annotations
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import (
    QUERY_DYADS_REFERENCE,
    QUERY_TRIADS_CORE,
)
from tools.proposals_pipeline.population import load_chords, stack_hist
from tools.proposals_pipeline.metrics import (
    parallel_worker_setup,
    summarise_embedding_metrics,
    mean_std,
)
import tools.compare_proposals as cp  # Reuse preprocessors and logic

from sklearn.manifold import MDS, TSNE, Isomap
try:
    import umap
except Exception:
    umap = None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dimensionality reduction techniques."
    )
    parser.add_argument("--dyads-query", default=QUERY_DYADS_REFERENCE)
    parser.add_argument("--triads-query", default=QUERY_TRIADS_CORE)
    parser.add_argument("--seeds", default="42,123,999")
    parser.add_argument("--output", default=None)
    parser.add_argument("--proposal", default="simplex_sqrt", help="Single proposal to use for base distance.")
    parser.add_argument("--metric", default="hellinger", help="Metric to use.")
    return parser.parse_args()

def run_reductions(
    dist_condensed: np.ndarray,
    dist_matrix: np.ndarray,
    X_original: np.ndarray,
    seeds: List[int]
) -> List[Dict[str, Any]]:

    methods = ["MDS", "TSNE", "ISOMAP"]
    if umap:
        methods.append("UMAP")

    results = []

    # We need a dummy 'labels' array for compatibility with the new summarise_embedding_metrics signature
    # Since compare_reductions.py doesn't seem to pass entries deeply, we might not have labels handy.
    # However, 'load_chords' was called in main. Ideally we pass entries here.
    # For now, we will pass zeros and ignore cluster metrics in this specific script output if necessary,
    # or better, refactor to pass entries.
    # Let's assume N samples.
    n_samples = dist_matrix.shape[0]
    dummy_labels = np.zeros(n_samples)

    for method in methods:
        print(f"Running {method}...")

        row_stats = {
            "reduction": method,
            "seeds": seeds,
        }

        trusts = []
        conts = []
        stresses = []

        for seed in seeds:
            start = time.perf_counter()
            # cp.compute_embeddings uses pipeline_compute_embeddings which returns (emb, params)
            # cp.compute_embeddings is a wrapper that returns just emb.
            # So we can use cp.compute_embeddings safely.
            embedding = cp.compute_embeddings(
                dist_condensed,
                method,
                seed,
                base_matrix=X_original,
                n_jobs=1,
                deterministic=True
            )
            elapsed = time.perf_counter() - start

            # Use pipeline metrics, but we need to match signature
            # summarise_embedding_metrics(X_original, embedding, dist_matrix, dist_condensed, labels, seed)
            metrics = summarise_embedding_metrics(
                X_original,
                embedding,
                dist_matrix,
                dist_condensed,
                dummy_labels,
                seed
            )

            trusts.append(metrics["trustworthiness"])
            conts.append(metrics["continuity"])
            stresses.append(metrics["stress"])

        row_stats["trust_mean"], row_stats["trust_std"] = mean_std(trusts)
        row_stats["cont_mean"], row_stats["cont_std"] = mean_std(conts)
        row_stats["stress_mean"], row_stats["stress_std"] = mean_std(stresses)

        results.append(row_stats)

    return results

def main() -> None:
    args = parse_args()

    print("Loading population...")
    entries = load_chords(args.dyads_query, args.triads_query, "")
    hist, totals, counts, pairs, notes = stack_hist(entries)

    print(f"Preprocessing with {args.proposal}...")
    preprocessor = cp.PREPROCESSORS[args.proposal][1]
    kwargs = cp.PREPROCESSORS[args.proposal][2]
    X_original, dist_simplex = preprocessor(hist, counts=counts, pairs=pairs, **kwargs)

    print(f"Computing distances ({args.metric})...")
    dist_condensed = cp.metric_distance(args.metric, X_original, dist_simplex)
    dist_matrix = squareform(dist_condensed)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    results = run_reductions(dist_condensed, dist_matrix, X_original, seeds)

    df = pd.DataFrame(results)
    print("\nResults:")
    print(df.to_string())

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "reduction_comparison.csv", index=False)
        print(f"Saved to {out}")

if __name__ == "__main__":
    main()
