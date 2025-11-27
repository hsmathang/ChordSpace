# Reference Experiments

This document describes the reference experiments used to verify the correctness of the system during refactoring.

## Golden Master: Diadas Estructurales (Small)

A small, deterministic experiment to verify the entire pipeline (generation -> roughness -> reduction -> report).

### Configuration

*   **Population**: Structural Dyads (Octave 3, C3-B3).
    *   Query: `QUERY_DYADS_REFERENCE` (but filtered or small subset for speed if possible, here using standard reference).
*   **Roughness**: Simplex, Identity.
*   **Reduction**: MDS (Deterministic, Seed 42).
*   **Metrics**: Cosine, Euclidean.

### CLI Command (Legacy)

```bash
python3 -m tools.compare_proposals \
  --dyads-query "QUERY_DYADS_REFERENCE" \
  --proposals "simplex,identity" \
  --metrics "cosine,euclidean" \
  --reductions "MDS" \
  --seed 42 \
  --execution-mode deterministic \
  --n-jobs 1 \
  --output outputs/reference_experiment
```

### CLI Command (New)

```bash
python3 -m tools.compare_proposals \
  --config configs/reference_experiment.yaml
```
*(Once config file support is added to CLI)*

### Expected Metrics (Reference Values)

After running the legacy command, we record key metrics here.

*   **Simplex (Cosine) Stress**: [To be filled]
*   **Identity (Euclidean) Stress**: [To be filled]

## Verification Protocol

1.  Run Legacy Command.
2.  Save `metrics.csv` and `report.html`.
3.  Run New Command (via `run_experiment` wrapper).
4.  Compare `metrics.csv` (allow float tolerance `1e-6`).
5.  Check `report.html` contains expected plots.
