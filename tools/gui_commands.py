"""Extensibility primitives for the GUI experiment launcher.

This module defines a light-weight command protocol that extensions can
implement in order to plug additional behaviours into the Tk GUI without
touching the main window implementation.  Commands operate on a
``LauncherState`` snapshot (built by the GUI) and return a
``CommandResult`` structure with datasets, metrics and visualisations that
can be chained by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Protocol

import pandas as pd


@dataclass(frozen=True)
class LauncherState:
    """Snapshot of the GUI state passed to commands.

    Parameters
    ----------
    params:
        Scalar parameters describing the current launcher configuration
        (query names, execution mode, etc.).
    datasets:
        Mapping of logical dataset names (for example ``"population"`` or
        ``"selection"``) to pandas DataFrames.
    selection:
        Small mapping that enumerates indices/identifiers of the current
        selection performed by the user.
    metadata:
        Additional contextual information (output directories, timestamps,
        user-provided descriptors, ...).
    """

    params: Mapping[str, Any] = field(default_factory=dict)
    datasets: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    selection: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def get_param(self, key: str, default: Any | None = None) -> Any:
        """Convenience accessor for scalar parameters."""

        return self.params.get(key, default)

    def get_dataset(self, key: str) -> Optional[pd.DataFrame]:
        """Return a dataset if it exists in the snapshot."""

        return self.datasets.get(key)


@dataclass
class CommandResult:
    """Structured response returned by experiment commands."""

    datasets: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    visualizations: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


class ExperimentCommand(Protocol):
    """Executable unit used by the GUI to perform an action."""

    def __call__(self, state: LauncherState) -> CommandResult:
        ...


class ComparisonService(Protocol):
    """Service responsible for running comparison experiments."""

    def run(self, state: LauncherState) -> CommandResult:
        ...


class NeighbourService(Protocol):
    """Service that generates neighbours for a given selection."""

    def generate(self, state: LauncherState) -> CommandResult:
        ...


class PopulationAugmentationService(Protocol):
    """Service that augments populations with new entries."""

    def augment(self, state: LauncherState) -> CommandResult:
        ...


class RunComparisonCommand:
    """Command that delegates to a :class:`ComparisonService`."""

    def __init__(self, service: ComparisonService) -> None:
        self._service = service

    def __call__(self, state: LauncherState) -> CommandResult:
        return self._service.run(state)


class GenerateNeighboursCommand:
    """Command that obtains neighbours through ``NeighbourService``."""

    def __init__(self, service: NeighbourService) -> None:
        self._service = service

    def __call__(self, state: LauncherState) -> CommandResult:
        return self._service.generate(state)


class AugmentPopulationCommand:
    """Command that extends populations via ``PopulationAugmentationService``."""

    def __init__(self, service: PopulationAugmentationService) -> None:
        self._service = service

    def __call__(self, state: LauncherState) -> CommandResult:
        return self._service.augment(state)


class CommandRegistry:
    """Container that stores experiment commands by name."""

    def __init__(self) -> None:
        self._commands: MutableMapping[str, ExperimentCommand] = {}

    def register(self, name: str, command: ExperimentCommand) -> None:
        """Register a new command under ``name``.

        Parameters
        ----------
        name:
            Identifier used to later resolve the command.
        command:
            Object implementing :class:`ExperimentCommand`.
        """

        if not name:
            raise ValueError("El nombre del comando no puede estar vacío.")
        self._commands[name] = command

    def unregister(self, name: str) -> None:
        """Remove a previously registered command if present."""

        self._commands.pop(name, None)

    def get(self, name: str) -> ExperimentCommand:
        """Retrieve a command by name, raising ``KeyError`` if missing."""

        if name not in self._commands:
            raise KeyError(f"No se encontró un comando llamado '{name}'.")
        return self._commands[name]

    def dispatch(self, name: str, state: LauncherState) -> CommandResult:
        """Execute the command ``name`` with ``state``."""

        command = self.get(name)
        return command(state)

    def available(self) -> Iterable[str]:
        """Return an iterable with the registered command names."""

        return tuple(self._commands.keys())

