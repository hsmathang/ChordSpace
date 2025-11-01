import numpy as np
import pandas as pd
import pytest

from services.proposals.data import PopulationLoader
from services.proposals.service import ProposalsComparisonService


@pytest.fixture
def sample_population_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "interval": [4],
                "code": "DyadMaj",
                "chroma": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "notes": [60, 64],
                "__family_id": 1,
            },
            {
                "interval": [4, 3],
                "code": "TriadMaj",
                "chroma": [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                "notes": [60, 64, 67],
                "__family_id": 2,
            },
        ]
    )


def test_prepare_population_returns_serialisable_payload(sample_population_df):
    service = ProposalsComparisonService()
    payload = service.prepare_population(
        dyads_query="", triads_query="", dataframe=sample_population_df
    )

    assert len(payload["entries"]) == len(sample_population_df)
    assert len(payload["totals"]) == len(sample_population_df)
    first_entry = payload["entries"][0]
    assert isinstance(first_entry["hist"], list)
    assert len(first_entry["hist"]) == 12


def test_compute_metrics_with_simplex(sample_population_df):
    service = ProposalsComparisonService()
    service.prepare_population("", "", dataframe=sample_population_df)

    metrics = service.compute_metrics("simplex", "euclidean")

    assert metrics["preprocessor"] == "simplex"
    assert metrics["preprocessor_params"] == {}
    assert metrics["metric"] == "euclidean"
    assert isinstance(metrics["distances"], list)
    assert len(metrics["distances"]) == 1
    # La métrica debe registrar resultados top-N al haber una tríada y una díada
    assert metrics["nn_hit_top1"] is not None
    assert metrics["nn_hit_top2"] is not None


def test_render_visualisations_allows_custom_embedding(sample_population_df):
    service = ProposalsComparisonService()
    service.prepare_population("", "", dataframe=sample_population_df)
    service.compute_metrics("simplex", "euclidean")

    embedding = np.array([[0.0, 0.0], [1.0, 0.5]])
    visual_payload = service.render_visualisations(
        color_mode="pair_exp", exponent=1.0, embedding=embedding
    )

    assert visual_payload["color"]["title"].startswith("Total/")
    assert visual_payload["embedding"] == embedding.tolist()
    assert len(visual_payload["entries"]) == len(sample_population_df)


def test_population_loader_resolves_query_alias(monkeypatch, sample_population_df):
    captured = []

    def fake_resolve(name: str) -> str:
        captured.append(name)
        return "SELECT 1"

    class StubExecutor:
        def as_pandas(self, sql: str) -> pd.DataFrame:
            assert sql == "SELECT 1"
            return sample_population_df

    loader = PopulationLoader(executor=StubExecutor())
    monkeypatch.setattr(
        "services.proposals.data.resolve_query_sql", fake_resolve
    )

    entries = loader.from_queries("QUERY_DUMMY", "", None)

    assert captured == ["QUERY_DUMMY"]
    assert len(entries) == len(sample_population_df)
