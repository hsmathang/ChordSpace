import argparse
from pathlib import Path

import pytest

from ui.launcher.controllers import (
    ExperimentLauncherController,
    ExperimentRunRequest,
    ExperimentDataGateway,
    MissingParametersError,
)
from ui.launcher.state import LauncherState


class DummyGateway(ExperimentDataGateway):
    def __init__(self) -> None:
        self.received = []

    def run(self, args, df_override=None, descriptor=None):  # type: ignore[override]
        self.received.append((args, df_override, descriptor))
        return {"output_dir": "/tmp/out"}


def make_state(tmp_path: Path) -> LauncherState:
    state = LauncherState()
    state.update(
        chord_type="B",
        reduction="MDS",
        base_query="QUERY_BASE",
        model="sethares",
        metric="euclidean",
        ponderation="ninguna",
    )
    state.set_population_sources(["A:foo"])
    state.set_output_dir(tmp_path)
    return state


def test_build_experiment_request_includes_state(tmp_path):
    state = make_state(tmp_path)
    controller = ExperimentLauncherController(state)

    request = controller.build_experiment_request(
        default_output_dir=tmp_path / "default",
        selected_ids=[1, 2],
        df_override=None,
        descriptor="preview",
    )

    assert request.output_dir == tmp_path
    assert request.args.query == "QUERY_BASE"
    assert request.args.pops == ["A:foo"]
    assert request.args.model == "sethares"
    assert request.descriptor == "preview"


def test_build_experiment_request_requires_parameters(tmp_path):
    state = LauncherState()
    controller = ExperimentLauncherController(state)

    with pytest.raises(MissingParametersError):
        controller.build_experiment_request(
            default_output_dir=tmp_path / "default",
            selected_ids=[],
            df_override=None,
            descriptor=None,
        )


def test_run_experiment_uses_gateway_and_logs(tmp_path):
    state = make_state(tmp_path)
    gateway = DummyGateway()
    controller = ExperimentLauncherController(state, data_gateway=gateway)

    args = argparse.Namespace(
        out=tmp_path,
        type="B",
        query="QUERY_BASE",
        reduction="MDS",
        pops=["A:foo"],
        pops_csv=None,
        pops_file=None,
        model="sethares",
        metric="euclidean",
        ponderation="ninguna",
    )
    request = ExperimentRunRequest(args=args, df_override=None, descriptor=None, output_dir=tmp_path)

    result = controller.run_experiment(request)

    assert gateway.received and gateway.received[0][0] is args
    assert result == {"output_dir": "/tmp/out"}
    category, message = state.log_queue.get_nowait()
    assert category == "exp_log"
    assert "Iniciando" in message
