"""State management primitives for the experiment launcher UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Set

import queue


StateListener = Callable[[str, Any, Any], None]


@dataclass
class LauncherState:
    """Observable state container for the experiment launcher."""

    chord_type: str = "B"
    reduction: str = "MDS"
    base_query: Optional[str] = None
    output_dir: Optional[Path] = None
    population_sources: list[str] = field(default_factory=list)
    filters_enabled: bool = False
    model: str = "Sethares"
    metric: str = "euclidean"
    ponderation: str = "ninguna"
    execution_mode: str = "deterministic"
    n_jobs: Optional[str] = None
    selected_proposals: Set[str] = field(default_factory=set)
    selected_metrics: Set[str] = field(default_factory=set)
    selected_reductions: Set[str] = field(default_factory=set)
    comparison_seeds: str = "42"
    log_queue: "queue.Queue[tuple[str, Any]]" = field(default_factory=queue.Queue)
    running_thread: Any = None
    status_text: str = "Listo"
    comparison_status_text: str = "Listo"

    _listeners: list[StateListener] = field(default_factory=list, init=False, repr=False)

    def subscribe(self, listener: StateListener) -> None:
        self._listeners.append(listener)

    # Generic update helpers -------------------------------------------------
    def _notify(self, field_name: str, old: Any, new: Any) -> None:
        if old == new:
            return
        for listener in list(self._listeners):
            listener(field_name, old, new)

    def update(self, **changes: Any) -> None:
        for field_name, value in changes.items():
            old = getattr(self, field_name)
            setattr(self, field_name, value)
            self._notify(field_name, old, value)

    # Domain specific helpers ------------------------------------------------
    def set_output_dir(self, value: str | Path | None) -> None:
        path = Path(value).expanduser() if value else None
        self.update(output_dir=path)

    def set_population_sources(self, sources: Iterable[str]) -> None:
        self.update(population_sources=list(sources))

    def toggle_filter(self, enabled: bool) -> None:
        self.update(filters_enabled=enabled)

    def set_proposal(self, name: str, enabled: bool) -> None:
        self._update_set("selected_proposals", name, enabled)

    def set_metric(self, name: str, enabled: bool) -> None:
        self._update_set("selected_metrics", name, enabled)

    def set_reduction(self, name: str, enabled: bool) -> None:
        self._update_set("selected_reductions", name, enabled)

    def _update_set(self, field_name: str, element: str, enabled: bool) -> None:
        values: Set[str] = getattr(self, field_name)
        if enabled:
            new = set(values)
            new.add(element)
        else:
            new = set(values)
            new.discard(element)
        self.update(**{field_name: new})

    def mark_running(self, thread: Any) -> None:
        self.update(running_thread=thread)

    def set_status(self, text: str) -> None:
        self.update(status_text=text)

    def set_comparison_status(self, text: str) -> None:
        self.update(comparison_status_text=text)
