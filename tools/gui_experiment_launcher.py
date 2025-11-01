"""Entry point for the Tk-based experiment launcher."""

from __future__ import annotations

from ui.launcher.views.app import ExperimentLauncher


def main() -> None:
    app = ExperimentLauncher()
    app.mainloop()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
