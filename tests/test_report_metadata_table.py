import pandas as pd
import plotly.graph_objects as go
from tools.reporting.report_builder import render_report_html
from pathlib import Path


def test_metadata_tables_render(tmp_path):
    metrics_df = pd.DataFrame([
        {
            "scenario": "MDS:identity | cosine",
            "metric": "cosine",
            "stress_mean": 0.1,
            "stress_std": 0.01,
            "trustworthiness_mean": 0.9,
            "trustworthiness_std": 0.02,
            "mixture_l1_mean_mean": 0.3,
            "mixture_l1_mean_std": 0.01,
            "seeds": [1],
            "preproc_id": "identity",
            "reduction": "MDS",
        }
    ])
    run_metadata = {
        "selection": {"rows_selected": 1, "rows_available": 5, "mode": "compare", "payload_path": "foo.json"},
        "population": {
            "descriptors": [
                {
                    "mode": "combinatorial",
                    "rows": 10,
                    "combinatorial": {
                        "alphabet": ["C", "D#"],
                        "cardinalities": [3],
                        "octave_min": 3,
                        "octave_max": 4,
                        "structural_mode": True,
                    },
                    "filters": {"label": "triadas"},
                },
                {
                    "mode": "database",
                    "rows": 5,
                    "database": {"base_query": "QUERY_DYADS_REFERENCE", "pops_entries": ["A:foo"], "filter_mode": "strict"},
                    "filters": {"label": "db_filtro"},
                }
            ]
        },
    }
    out_path = tmp_path / "report.html"
    render_report_html(
        metrics_df,
        [("MDS:identity | cosine||raw_total", go.Figure())],
        out_path,
        seeds=[1],
        run_metadata=run_metadata,
    )
    text = out_path.read_text(encoding="utf-8")
    assert "meta-section" in text
    assert "triadas" in text
