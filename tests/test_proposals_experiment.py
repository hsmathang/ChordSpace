from tools.proposals_pipeline.experiment import build_scenarios
from tools.compare_proposals import PREPROCESSORS


def test_build_scenarios_includes_identity_when_requested():
    scenarios = build_scenarios(["simplex"], ["cosine"], PREPROCESSORS, include_identity=True)
    names = {s["name"] for s in scenarios}
    assert "simplex | cosine" in names
    assert "identity | cosine" in names
    ids = {(s["preproc_id"], s["metric"]) for s in scenarios}
    assert ("identity", "cosine") in ids


def test_build_scenarios_respects_disable_identity():
    scenarios = build_scenarios(["simplex"], ["cosine"], PREPROCESSORS, include_identity=False)
    names = {s["name"] for s in scenarios}
    assert "identity | cosine" not in names


def test_build_scenarios_skips_unknown():
    scenarios = build_scenarios(["unknown_proposal"], ["cosine"], PREPROCESSORS, include_identity=False)
    assert not scenarios
