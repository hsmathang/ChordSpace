"""
CLI tool for running ChordSpace experiments using the new architecture.
Replaces the old compare_proposals.py logic with a thin wrapper around services.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from services.domain import (
    ExperimentConfig,
    PopulationConfig,
    RoughnessConfig,
    ReductionConfig,
    ExecutionConfig,
    VisualizationConfig
)
from services.space_experiments import ExperimentService
from services.space_visualization import VisualizationService

# Import metric info to make available for report (or we can inject it in VisualizationService)
from tools.compare_proposals import METRIC_INFO

def parse_args():
    parser = argparse.ArgumentParser(description="Run ChordSpace experiment (New Arch)")

    # Population Args
    parser.add_argument("--population-json", help="Path to population JSON file")
    parser.add_argument("--dyads-query", help="Query for dyads")
    parser.add_argument("--triads-query", help="Query for triads")
    parser.add_argument("--sevenths-query", help="Query for sevenths")

    # Roughness/Metrics Args
    parser.add_argument("--proposals", default="simplex,identity", help="Roughness proposals")
    parser.add_argument("--metrics", default="cosine,euclidean", help="Metrics to calculate")
    parser.add_argument("--disable-baseline-identity", action="store_true")

    # Reduction Args
    parser.add_argument("--reductions", default="MDS", help="Reduction methods")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--mds-n-init", type=int, default=4)

    # Execution Args
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", help="List of seeds")
    parser.add_argument("--execution-mode", choices=["deterministic", "parallel"], default="deterministic")
    parser.add_argument("--output", default="outputs/experiment_new", help="Output directory")

    # Visualization Args
    parser.add_argument("--sections", default="all")
    parser.add_argument("--color-mode", default="log_per_pair")

    # Metadata
    parser.add_argument("--run-metadata", help="Path to metadata JSON")

    return parser.parse_args()

def main():
    args = parse_args()

    # Build Configuration Objects

    # 1. Population
    meta = {}
    if args.run_metadata:
        try:
            with open(args.run_metadata, 'r') as f:
                meta = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")

    pop_config = PopulationConfig(
        source_type="file" if args.population_json else "db",
        file_path=args.population_json,
        dyads_query=args.dyads_query,
        triads_query=args.triads_query,
        sevenths_query=args.sevenths_query,
        metadata=meta
    )

    # 2. Roughness
    rough_config = RoughnessConfig(
        proposals=[p.strip() for p in args.proposals.split(",") if p.strip()],
        metrics=[m.strip() for m in args.metrics.split(",") if m.strip()],
        disable_baseline=args.disable_baseline_identity
    )

    # 3. Reduction
    red_config = ReductionConfig(
        methods=[r.strip() for r in args.reductions.split(",") if r.strip()],
        n_init=args.mds_n_init,
        n_jobs=args.n_jobs if args.n_jobs is not None else 1
    )

    # 4. Execution
    seeds = [args.seed]
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    exec_config = ExecutionConfig(
        seeds=seeds,
        deterministic=(args.execution_mode == "deterministic"),
        output_dir=args.output
    )

    # 5. Visualization
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    vis_config = VisualizationConfig(
        sections=sections,
        color_mode=args.color_mode
    )

    experiment_config = ExperimentConfig(
        population=pop_config,
        roughness=rough_config,
        reduction=red_config,
        execution=exec_config,
        name="cli_experiment"
    )

    # Run Service
    print("Running Experiment Service...")
    service = ExperimentService()
    result = service.run_experiment(experiment_config)

    # Visualization
    print("Generating Report...")
    vis_service = VisualizationService()
    vis_service.generate_report_full(result, vis_config)

    print(f"Done. Output at {result.output_path}")

if __name__ == "__main__":
    main()
