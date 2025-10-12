# GUI Launcher (Tk)

- Ensure you are using the project virtual environment:
  - Windows: `py -m venv .venv && .\\.venv\\Scripts\\activate`
  - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`

Run the GUI:
- Windows: ` .\\.venv\\Scripts\\python -m tools.gui_experiment_launcher`
- macOS/Linux: `python -m tools.gui_experiment_launcher`

Troubleshooting:
- Verify the Python used is the venv one:
  - Windows: `Get-Command python` should point to `.venv\\Scripts\\python.exe`
  - macOS/Linux: `which python` should point to `.venv/bin/python`
- If the window does not open and the command returns immediately, you likely used the system Python. Run with the explicit venv path as shown above.
- Plotly in notebooks: set `PLOTLY_RENDERER=browser` if figures do not display.
- Windows + `Illegal instruction (0xc000001d)` when launching the GUI:
  1. `pip uninstall -y polars polars-runtime-32 polars-lts-cpu`
  2. `pip install --no-cache-dir polars-lts-cpu==1.32.3`
  3. `python -c "import polars; print(polars.__version__)"`
