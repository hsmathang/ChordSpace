"""Compatibility shim for launcher entrypoints.

Historically the GUI class lived in ``ui.launcher.views.app``.
After refactors the implementation moved to ``app_new``; this module
re-exports the public entrypoints so existing commands keep working.
"""

from __future__ import annotations

from .app_new import ExperimentLauncher, main

__all__ = ["ExperimentLauncher", "main"]


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
