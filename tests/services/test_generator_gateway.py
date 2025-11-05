import pandas as pd

from services.data_gateway import GeneratorPopulationGateway, GeneratorRequest


def test_generator_gateway_basic():
    gateway = GeneratorPopulationGateway()
    spec = GeneratorRequest.from_any(
        {
            "mode": "total",
            "alphabet": "0,4,7",
            "octaves": "4-4",
            "cardinalities": "3",
            "limit": 3,
            "label": "GEN_TEST",
        }
    )

    result = gateway.fetch_population([spec], dedupe=False)
    df = result.dataframe

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "__source__" in df.columns
    assert set(df["__source__"]) == {"GEN_TEST"}
    assert result.stats.get("raw_count") == len(df)
    generator_specs = result.stats.get("generator_specs")
    assert generator_specs
    first_spec = generator_specs[0]
    assert first_spec["mode"] == "total"
    assert first_spec["label"] == "GEN_TEST"
