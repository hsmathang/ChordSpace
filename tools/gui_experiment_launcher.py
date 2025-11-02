"""Entry point for the Tk-based experiment launcher.

Thin wrapper around :class:`ui.launcher.views.app.ExperimentLauncher` so the
GUI can be started with ``python -m tools.gui_experiment_launcher`` while the
implementation lives in the ``ui`` package.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from tools.query_registry import get_all_queries
from ui.launcher.views.app import ExperimentLauncher


def _print_queries() -> None:
    queries = get_all_queries()
    for name, meta in sorted(queries.items()):
        source = meta.get("source", "")
        if source:
            print(f"{name} [{source}]")
        else:
            print(name)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI launcher for ChordSpace experiments")
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="Lista las consultas registradas y finaliza.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_queries:
        _print_queries()
        return

    app = ExperimentLauncher()
    app.mainloop()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
