"""
generate_thesis_results.py
---------------------------
Corre el experimento de tesis cargando la población pre-generada
(outputs/tesis_resultados_finales/population.json) que contiene:
  - 12 díadas básicas (m2, M2, m3, M3, P4, Tritono, P5, m6, M6, m7, M7, P8)
  - 21 tríadas diatónicas de C mayor con inversiones (C, Dm, Em, F, G, Am, Bdim)

Métricas:  Euclidiana y Coseno
Reducciones: MDS y UMAP

Uso (desde raíz del proyecto con el venv activo):
    python -m generate_thesis_results
"""
from __future__ import annotations

import sys
from pathlib import Path

from services.domain import (
    ExperimentConfig, PopulationConfig, RoughnessConfig,
    ReductionConfig, ExecutionConfig, VisualizationConfig, AudioConfig,
)
from services.space_experiments import ExperimentService
from services.space_visualization import VisualizationService

POP_FILE   = "outputs/tesis_resultados_finales/population.json"
OUTPUT_DIR = "outputs/tesis_resultados_finales"


def main() -> None:
    pop_path = Path(POP_FILE)
    if not pop_path.exists():
        print(f"[ERROR] Fichero de población no encontrado: {pop_path}")
        print("  -> Crea primero el fichero ejecutando:  python gen_thesis_pop.py")
        sys.exit(1)

    print(f"[1] Población -> {pop_path}  ({sum(1 for _ in open(pop_path))} acordes)")

    pop_config = PopulationConfig(
        source_type="file",
        file_path=str(pop_path),
        metadata={
            "descripcion": "21 tríadas diatónicas c/ inversiones + 12 díadas básicas",
            "octave_range": "3-4",
            "escala": "C mayor diatónica",
        },
    )

    rough_config = RoughnessConfig(
        proposals=["simplex"],          # propuesta central de rugosidad normalizada
        metrics=["euclidean", "cosine"],
        disable_baseline=True,          # no identity baseline → sólo simplex
    )

    red_config = ReductionConfig(
        methods=["MDS", "UMAP"],
        n_init=4,
        n_jobs=1,
    )

    exec_config = ExecutionConfig(
        seeds=[42],
        deterministic=True,
        output_dir=OUTPUT_DIR,
    )

    vis_config = VisualizationConfig(
        sections=["scatter", "heatmap", "table", "metadata"],
        color_mode="log_per_pair",
        audio=AudioConfig(enabled=False),
        skip_html_report=False,
    )

    exp_config = ExperimentConfig(
        population=pop_config,
        roughness=rough_config,
        reduction=red_config,
        execution=exec_config,
        name="tesis_triadas_diadas",
    )

    print("[2] Corriendo ExperimentService…")
    svc = ExperimentService()
    result = svc.run_experiment(exp_config, logger=print)

    print("[3] Generando reporte HTML…")
    vis = VisualizationService()
    vis.generate_report_full(result, vis_config, logger=print)

    print(f"\n[OK] Resultados en: {result.output_path.resolve()}")


if __name__ == "__main__":
    main()
