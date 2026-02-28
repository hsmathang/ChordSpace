from services.domain import (
    ExperimentConfig,
    PopulationConfig,
    RoughnessConfig,
    ReductionConfig,
    ExecutionConfig
)
from services.space_experiments import ExperimentService
import sys

def main():
    print("Test isolated experiment execution...")
    
    pop_config = PopulationConfig(
        source_type="file", 
        file_path="outputs/tesis_resultados_finales/population.json"
    )
    
    rough_config = RoughnessConfig(
        proposals=["simplex"],
        metrics=["euclidean"], 
        disable_baseline=True
    )
    
    red_config = ReductionConfig(methods=["MDS"], n_init=1)
    
    exec_config = ExecutionConfig(
        seeds=[42],
        deterministic=True,
        output_dir="outputs/tesis_resultados_finales"
    )
    
    experiment_config = ExperimentConfig(
        population=pop_config,
        roughness=rough_config,
        reduction=red_config,
        execution=exec_config,
        name="test"
    )
    
    print("Running Experiment Service...", flush=True)
    service = ExperimentService()
    try:
        result = service.run_experiment(experiment_config)
        print("Experiment completed!", flush=True)
        print(f"Metrics computed: {result.metrics_df.shape}", flush=True)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
