import json
from typing import List

import numpy as np
import plotly.graph_objects as go

from visualisations.proposals import (
    FigureSpec,
    build_board_index,
    build_scatter_payload,
)


class DummyChord:
    def __init__(self, name: str, intervals: List[int]):
        self.name = name
        self.intervals = intervals


class DummyEntry:
    def __init__(
        self,
        name: str,
        intervals: List[int],
        hist: List[float],
        *,
        family_id: str = "FamA",
        is_named: bool = True,
        identity: str = "Major",
        aliases: List[str] | None = None,
        is_inversion: bool = False,
        inversion_rotation: int | None = None,
    ) -> None:
        self.acorde = DummyChord(name, intervals)
        self.hist = np.asarray(hist, dtype=float)
        self.total = float(np.sum(self.hist))
        self.n_notes = len(intervals) + 1
        self.identity_name = identity
        self.identity_aliases = tuple(aliases or [])
        self.is_named = is_named
        self.family_id = family_id
        self.is_inversion = is_inversion
        self.inversion_rotation = inversion_rotation


def build_sample_entries() -> List[DummyEntry]:
    return [
        DummyEntry("Cmaj7", [4, 3, 4], [0.5, 0.3, 0.2], family_id="Fam1", aliases=["CM7"]),
        DummyEntry("C7/E", [4, 3, 3], [0.4, 0.4, 0.1], family_id="Fam1", is_inversion=True, inversion_rotation=1),
    ]


def test_build_scatter_payload_returns_serialisable_dict():
    entries = build_sample_entries()
    embedding = np.array([[0.0, 0.1], [0.2, 0.3]])
    vectors = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.1]])
    adjusted = vectors * 1.1
    color_values = np.array([0.8, 0.6])
    pair_counts = np.array([4.0, 5.0])
    type_counts = np.array([3.0, 2.0])

    payload = build_scatter_payload(
        embedding=embedding,
        entries=entries,
        color_values=color_values,
        pair_counts=pair_counts,
        type_counts=type_counts,
        vectors=vectors,
        adjusted_vectors=adjusted,
        title="Demo",
        color_title="Color",
        is_proposal=True,
        meta={"scenario": "demo", "mode": "raw_total", "exponent": None},
    )

    assert set(payload.keys()) == {"data", "layout", "meta"}
    assert payload["data"], "expected traces in payload"
    assert payload["layout"]["meta"]["colorTitle"] == "Color"
    assert payload["meta"]["scenario"] == "demo"
    assert payload["meta"]["mode"] == "raw_total"
    assert payload["data"][0]["customdata"], "customdata should carry hover details"


def test_figurespec_materialises_plotly_figures():
    entries = build_sample_entries()
    payload = build_scatter_payload(
        embedding=np.array([[0.0, 0.0], [1.0, 1.0]]),
        entries=entries,
        color_values=np.array([0.1, 0.9]),
        pair_counts=np.array([3.0, 4.0]),
        type_counts=np.array([2.0, 2.0]),
        vectors=np.array([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]),
        adjusted_vectors=np.array([[0.22, 0.33, 0.55], [0.11, 0.22, 0.77]]),
        title="Spec",
        color_title="Total",
        is_proposal=False,
        meta={"scenario": "spec", "mode": "raw_total", "exponent": None},
    )
    spec = FigureSpec(
        key="scenario||raw_total",
        scenario="scenario",
        metric="cosine",
        reduction="MDS",
        proposal="identity",
        mode="raw_total",
        exponent=None,
        seed=123,
        payload=payload,
    )
    fig = spec.to_figure()
    assert isinstance(fig, go.Figure)
    json.loads(spec.to_json())
    serialised = spec.serializable()
    assert serialised["figure"]["data"] == payload["data"]


def test_board_index_handles_optional_reductions():
    entries = build_sample_entries()
    payload = build_scatter_payload(
        embedding=np.array([[0.0, 0.0], [0.1, 0.2]]),
        entries=entries,
        color_values=np.array([0.1, 0.2]),
        pair_counts=np.array([3.0, 3.0]),
        type_counts=np.array([1.0, 1.0]),
        vectors=np.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.2]]),
        adjusted_vectors=np.array([[0.11, 0.22, 0.33], [0.22, 0.22, 0.22]]),
        title="Board",
        color_title="Total",
        is_proposal=True,
        meta={"scenario": "board", "mode": "raw_total", "exponent": None},
    )
    spec = FigureSpec(
        key="scenario||raw_total",
        scenario="scenario",
        metric="cosine",
        reduction="MDS",
        proposal="identity",
        mode="raw_total",
        exponent=None,
        seed=42,
        payload=payload,
    )

    boards = build_board_index([spec])
    assert "identity" in boards["by_proposal"]
    assert "cosine" in boards["by_metric"]
    assert "MDS" in boards["by_reduction"]
    assert "UMAP" not in boards["by_reduction"], "UMAP board should be optional"
