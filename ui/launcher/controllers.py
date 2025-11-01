"""Controllers for the experiment launcher UI."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from tools import experiment_inversions

from .state import LauncherState


@dataclass
class ExperimentRunRequest:
    args: argparse.Namespace
    df_override: Optional[pd.DataFrame]
    descriptor: Optional[str]
    output_dir: Path


class ControllerError(Exception):
    """Base error for controller failures."""


class MissingParametersError(ControllerError):
    """Raised when required parameters are not provided."""


class ExperimentDataGateway:
    """Boundary responsible for executing experiments."""

    def run(
        self,
        args: argparse.Namespace,
        df_override: Optional[pd.DataFrame] = None,
        descriptor: Optional[str] = None,
    ) -> dict:
        return experiment_inversions.run_experiment_with_args(
            args,
            df_override=df_override,
            descriptor=descriptor,
        )


class ProposalsComparisonService:
    """Placeholder service for comparison runs."""

    def run(self, *args, **kwargs):  # pragma: no cover - behaviour delegated to legacy code
        raise NotImplementedError


class ExperimentLauncherController:
    """High level coordinator between the launcher views and services."""

    def __init__(
        self,
        state: LauncherState,
        data_gateway: Optional[ExperimentDataGateway] = None,
        comparison_service: Optional[ProposalsComparisonService] = None,
    ) -> None:
        self.state = state
        self.data_gateway = data_gateway or ExperimentDataGateway()
        self.comparison_service = comparison_service or ProposalsComparisonService()

    # ------------------------------------------------------------------ build
    def build_experiment_request(
        self,
        *,
        default_output_dir: Path,
        selected_ids: list[int],
        df_override: Optional[pd.DataFrame],
        descriptor: Optional[str],
    ) -> ExperimentRunRequest:
        output_dir = self.state.output_dir or default_output_dir
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        base_query = self.state.base_query
        if base_query == "<Ninguna>":
            base_query = None

        pops = [p for p in self.state.population_sources if p]
        filters_enabled = self.state.filters_enabled

        if not selected_ids and df_override is None:
            if not pops and not base_query and not filters_enabled:
                raise MissingParametersError(
                    "Selecciona una consulta base, poblaciones adicionales o activa los filtros personalizados.",
                )

        args = argparse.Namespace(
            out=Path(output_dir),
            type=self.state.chord_type,
            query=base_query,
            reduction=self.state.reduction,
            pops=pops if pops else None,
            pops_csv=None,
            pops_file=None,
            model=self.state.model,
            metric=self.state.metric,
            ponderation=self.state.ponderation,
        )

        return ExperimentRunRequest(
            args=args,
            df_override=df_override,
            descriptor=descriptor,
            output_dir=Path(output_dir),
        )

    # ------------------------------------------------------------------ run
    def run_experiment(self, request: ExperimentRunRequest) -> dict:
        writer = QueueWriter(self.state.log_queue, category="exp_log")
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                print(
                    f"[experimento] Iniciando… modelo={request.args.model}, "
                    f"metrica={request.args.metric}, reduccion={request.args.reduction}"
                )
                result = self.data_gateway.run(
                    request.args,
                    df_override=request.df_override,
                    descriptor=request.descriptor,
                )
        finally:
            writer.flush()
        return result


class QueueWriter:
    """Redirect stdout/stderr writes into a queue for the GUI to consume."""

    def __init__(self, bucket, category: str = "log"):
        self.bucket = bucket
        self.category = category

    def write(self, msg: str) -> int:
        if msg:
            self.bucket.put((self.category, msg))
        return len(msg)

    def flush(self) -> None:  # pragma: no cover - nothing to flush
        return None
