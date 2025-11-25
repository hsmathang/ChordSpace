import numpy as np
from tools.proposals_pipeline.runner import run_experiment
from tools.compare_proposals import PREPROCESSORS


def test_run_experiment_smoke():
    # Población mínima de 2 acordes (histograma 3 bins)
    hist = np.array([[0.5, 0.5, 0.0], [0.4, 0.6, 0.0]])
    counts = np.ones_like(hist)
    pairs = np.array([1.0, 1.0])
    scenarios = [
        {
            "name": "simplex | cosine",
            "description": "",
            "preproc_id": "simplex",
            "preproc_func": PREPROCESSORS["simplex"][1],
            "preproc_kwargs": PREPROCESSORS["simplex"][2],
            "metric": "cosine",
        }
    ]
    res = run_experiment(
        entries=[],  # no se usan en metric_distance
        hist=hist,
        counts=counts,
        pairs=pairs,
        seed_list=[0],
        reductions=["MDS"],
        scenarios=scenarios,
        deterministic=True,
        jobs=1,
        mds_n_init=1,
        cpu_count=1,
    )
    assert res["results"]
    assert res["figure_payloads"]
    assert res["expected_order"] == ["MDS:simplex | cosine"]
