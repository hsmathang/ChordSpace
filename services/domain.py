"""
Domain definitions for ChordSpace.
Contains configuration dataclasses, result structures, and core domain types.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

@dataclass
class PopulationConfig:
    """Configuration for loading or generating a chord population."""
    source_type: str = "file" # 'file', 'db', 'preset'
    # For file source
    file_path: Optional[str] = None
    # For DB source
    dyads_query: Optional[str] = None
    triads_query: Optional[str] = None
    sevenths_query: Optional[str] = None
    # For preset
    preset_name: Optional[str] = None

    # Metadata for reporting
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RoughnessConfig:
    """Configuration for roughness models and normalization proposals."""
    proposals: List[str] = field(default_factory=lambda: [
        "simplex", "simplex_sqrt", "simplex_smooth",
        "perclass_alpha1", "perclass_alpha0_75", "perclass_alpha0_5", "perclass_alpha0_25",
        "global_pairs", "divide_mminus1", "identity"
    ])
    metrics: List[str] = field(default_factory=lambda: ["cosine", "js", "hellinger", "euclidean"])
    disable_baseline: bool = False
    metric_params: Dict[str, Dict[str, float]] = field(default_factory=dict)

@dataclass
class ReductionConfig:
    """Configuration for dimensionality reduction algorithms."""
    methods: List[str] = field(default_factory=lambda: ["MDS"]) # MDS, UMAP, TSNE, ISOMAP
    n_init: int = 4
    n_jobs: int = 1

@dataclass
class ExecutionConfig:
    """Configuration for experiment execution environment."""
    seeds: List[int] = field(default_factory=lambda: [42])
    deterministic: bool = True
    output_dir: Optional[str] = None

@dataclass
class ExperimentConfig:
    """Complete configuration for an experiment run."""
    population: PopulationConfig
    roughness: RoughnessConfig
    reduction: ReductionConfig
    execution: ExecutionConfig
    name: str = "experiment"

@dataclass
class AudioConfig:
    """Configuration for audio playback in reports."""
    enabled: bool = False

@dataclass
class VisualizationConfig:
    """Configuration for report generation."""
    sections: List[str] = field(default_factory=lambda: ["scatter", "heatmap", "shepard", "table", "metadata"])
    color_mode: str = "log_per_pair"
    highlight_threshold: int = 2000
    audio: AudioConfig = field(default_factory=AudioConfig)
    instructional_mode: bool = False

@dataclass
class ExperimentResult:
    """Result of an experiment run."""
    config: ExperimentConfig
    # Data
    population_df: pd.DataFrame
    # Computed results
    metrics_df: pd.DataFrame
    embeddings: Dict[str, np.ndarray] # keys like "{reduction}:{scenario}"
    # Raw artifacts for debugging/caching
    dist_matrices: Dict[str, np.ndarray] = field(default_factory=dict)
    preproc_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    dist_simplex_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    # Population caches for idempotency
    entries: List[Any] = field(default_factory=list)
    totals: Any = None
    pairs: Any = None
    # Opaque payloads for visualization reconstruction (internal use)
    visualization_payloads: List[Dict[str, Any]] = field(default_factory=list)
    # Metadata
    warnings: List[str] = field(default_factory=list)
    timing: List[Tuple[str, float]] = field(default_factory=list)
    output_path: Optional[Path] = None
