"""
generate_umap_results.py
---------------------------
Adapts the modern `ExperimentService` to run the extended 81-chord C3 anchored 
population via the "perclass_alpha0_75" roughness proposal, but substituting
MDS for UMAP to see the nonlinear manifold structures. Generates `report.html`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.domain import (
    ExperimentConfig, PopulationConfig, RoughnessConfig,
    ReductionConfig, ExecutionConfig, VisualizationConfig, AudioConfig,
)
from services.space_experiments import ExperimentService
from services.space_visualization import VisualizationService

POP_FILE   = ROOT / "experiments" / "triad_consonance" / "poblacion_extendida_c3.jsonl"
OUTPUT_DIR = ROOT / "experiments" / "triad_consonance" / "outputs" / "reporte_extendida_umap"


def main() -> None:
    if not POP_FILE.exists():
        print(f"[ERROR] Fichero de población no encontrado: {POP_FILE}")
        sys.exit(1)

    print(f"[1] Población -> {POP_FILE}")

    pop_config = PopulationConfig(
        source_type="file",
        file_path=str(POP_FILE),
        metadata={
            "descripcion": "81 estructuras extendidas (Díadas, Tríadas, Tétradas, 9nas) ancladas a C3",
            "octave_range": "3",
            "escala": "Cromática Extendida",
        },
    )

    rough_config = RoughnessConfig(
        proposals=["perclass_alpha0_75"],  
        metrics=["euclidean"],  # UMAP works differently, keeping it clean with euclidean
        disable_baseline=True,          
    )

    red_config = ReductionConfig(
        methods=["UMAP"], # CAMBIO DE MDS A UMAP
        n_init=1,  # n_init is mostly an MDS concept in this codebase setting
        n_jobs=1,
    )

    exec_config = ExecutionConfig(
        seeds=[42],
        deterministic=True,
        output_dir=str(OUTPUT_DIR),
    )

    vis_config = VisualizationConfig(
        sections=["scatter", "heatmap", "table", "metadata"], # Shepard is MDS specific
        color_mode="log_per_pair",
        audio=AudioConfig(enabled=False),
        skip_html_report=False,
    )

    exp_config = ExperimentConfig(
        population=pop_config,
        roughness=rough_config,
        reduction=red_config,
        execution=exec_config,
        name="extendidas_c3_umap",
    )

    print("[2] Corriendo ExperimentService con UMAP…")
    svc = ExperimentService()
    result = svc.run_experiment(exp_config, logger=print)

    print("[3] Generando reporte HTML…")
    vis = VisualizationService()
    vis.generate_report_full(result, vis_config, logger=print)

    print(f"\n[OK] Resultados UMAP en: {result.output_path.resolve()}")


if __name__ == "__main__":
    main()
