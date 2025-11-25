import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from tools.proposals_pipeline.metrics import metric_distance
from tools.proposals_pipeline.figures import _apply_color_mode, ColorSettings
from services.combinatorial_generator import generate_combinatorial_chords
from visualisations.proposals import build_scatter_payload
from tools.reporting.contracts import validate_scatter_payload_meta, validate_run_metadata
from tools.reporting.report_builder import render_report_html


class DummyChord:
    def __init__(self, name: str, intervals, notes_abs=None, chroma=None):
        self.name = name
        self.intervals = intervals
        self.notes_abs = notes_abs
        self.chroma = chroma


class DummyEntry:
    def __init__(self, acorde, hist, total, n_notes, family_id=None, is_named=True):
        self.acorde = acorde
        self.hist = np.asarray(hist, dtype=float)
        self.total = float(total)
        self.n_notes = int(n_notes)
        self.identity_name = acorde.name
        self.identity_aliases = ()
        self.is_named = is_named
        self.family_id = family_id
        self.is_inversion = False
        self.inversion_rotation = None
        self.musical_inversion_ids = []
        self.structural_inversion_ids = []


def test_metric_distance_matches_scipy():
    simplex = np.array([
        [0.5, 0.5, 0.0],
        [0.7, 0.3, 0.0],
        [0.0, 0.0, 1.0],
    ])
    vectors = simplex.copy()
    js_res = metric_distance("js", vectors, simplex)
    hellinger_res = metric_distance("hellinger", vectors, simplex)
    cosine_res = metric_distance("cosine", vectors, simplex)

    from scipy.spatial.distance import pdist, jensenshannon

    expected_js = pdist(simplex, lambda u, v: float(jensenshannon(u, v, base=2)))
    expected_hell = pdist(np.sqrt(simplex), metric="euclidean") / np.sqrt(2.0)
    expected_cos = pdist(vectors, metric="cosine")

    np.testing.assert_allclose(js_res, expected_js)
    np.testing.assert_allclose(hellinger_res, expected_hell)
    np.testing.assert_allclose(cosine_res, expected_cos)

    with pytest.raises(ValueError):
        metric_distance("unknown_metric", vectors, simplex)


def test_apply_color_mode_variants():
    totals = np.array([10.0, 20.0])
    totals_adj = np.array([9.0, 12.0])
    pairs = np.array([2.0, 4.0])
    types = np.array([3.0, 5.0])
    settings = ColorSettings(
        per_pair_subtract=0.0,
        per_note_subtract=0.0,
        per_existing_subtract=0.0,
        existing_threshold=1e-6,
        denominator_exponent=1.0,
        output_exponent=1.0,
        exponents=[0.0, 0.5, 1.0],
    )

    vals_raw, title_raw = _apply_color_mode("raw_total", None, totals, totals_adj, pairs, types, settings=settings)
    vals_pair, title_pair = _apply_color_mode("pair_exp", 1.0, totals, totals_adj, pairs, types, settings=settings)
    vals_types, title_types = _apply_color_mode("types_exp", 0.5, totals, totals_adj, pairs, types, settings=settings)

    np.testing.assert_allclose(vals_raw, totals)
    np.testing.assert_allclose(vals_pair, np.array([5.0, 5.0]))
    expected_types = totals_adj / np.power(types, 0.5)
    np.testing.assert_allclose(vals_types, expected_types)

    assert "Total bruto" in title_raw
    assert "Total/Pares" in title_pair
    assert "Tipos" in title_types


def test_generate_combinatorial_chords_basic_columns():
    df = generate_combinatorial_chords([0, 4], 4, 4, [2], structural_mode=False)
    assert not df.empty
    assert df["__source__"].str.contains("GENERATED:COMBINATORIAL").any()
    required = {"id", "interval", "notes", "code", "abs_mask_int", "notes_abs_json", "__structure_id"}
    assert required.issubset(set(df.columns))

    df_struct = generate_combinatorial_chords([0, 4], 4, 4, [2], structural_mode=True)
    assert not df_struct.empty
    assert df_struct["__structure_id"].is_unique


def test_build_scatter_payload_includes_neighbors_and_filters():
    entries = [
        DummyEntry(DummyChord("Cmaj", [4, 3], notes_abs=[60, 64, 67]), hist=[1, 0, 0], total=1.0, n_notes=3, family_id=1),
        DummyEntry(DummyChord("Dm", [3, 4], notes_abs=[62, 65, 69]), hist=[0.6, 0.4, 0], total=1.0, n_notes=3, family_id=1),
        DummyEntry(DummyChord("E", [4, 3], notes_abs=[64, 68, 71]), hist=[0.2, 0.8, 0], total=1.0, n_notes=3, family_id=2),
    ]
    embedding = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    color_values = np.array([1.0, 0.5, 0.2])
    pairs = np.array([3.0, 3.0, 3.0])
    type_counts = np.array([2.0, 2.0, 2.0])
    vectors = np.array([[1, 0, 0], [0.6, 0.4, 0], [0.2, 0.8, 0]])
    adjusted_vectors = vectors.copy()

    payload = build_scatter_payload(
        embedding=embedding,
        entries=entries,
        color_values=color_values,
        pair_counts=pairs,
        type_counts=type_counts,
        vectors=vectors,
        adjusted_vectors=adjusted_vectors,
        title="demo",
        color_title="Color",
        is_proposal=False,
    )
    assert payload["data"]
    meta = payload.get("meta", {})
    assert "filterDataset" in meta
    assert "substitutionNeighbors" in meta
    profiles = meta.get("substitutionProfiles", {})
    assert profiles and profiles.get("default")
    validate_scatter_payload_meta(meta)


def test_render_report_html_writes_file(tmp_path):
    metrics_df = pd.DataFrame(
        [
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
        ]
    )
    fig = go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1])])
    out_path = tmp_path / "report.html"
    render_report_html(
        metrics_df,
        [("MDS:identity | cosine||raw_total", fig)],
        out_path,
        seeds=[1],
        run_metadata={"selection": {"rows_selected": 1, "rows_available": 1}},
    )
    text = out_path.read_text(encoding="utf-8")
    assert "Ranking" in text
    assert "tab-red-0" in text


def test_validate_run_metadata():
    valid = {
        "selection": {"rows_selected": 10, "rows_available": 100},
        "population": {"descriptors": [{"mode": "combinatorial", "rows": 10}]},
    }
    validate_run_metadata(valid)

    invalid = {"selection": "not_a_mapping"}
    with pytest.raises(ValueError):
        validate_run_metadata(invalid)
