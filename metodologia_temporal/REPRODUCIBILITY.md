Reproducibility & Peer Review Guide
===================================

Objetivo
- Permitir que revisores reproduzcan resultados sin depender de notebooks y validar diferencias en el modelo de Sethares.

Preparación del entorno (Windows PowerShell)
- `py -m venv venv`
- `./venv/Scripts/Activate`
- `pip install -r requirements.txt`

Runner del laboratorio (sin DB)
- `python -m tools.run_lab --model Sethares --metric cosine --reduction MDS --out outputs/demo`
- Artefactos en `outputs/demo`:
  - `embeddings.npy`, `distances.npy`, `report.txt`
  - `fig_scatter.html`, `fig_heatmap.html`, `fig_shepard.html`

Comparación controlada: Sethares canónico vs repo
- `python -m tools.compare_sethares --base-freq 500 --n-harmonics 10 --decay 0.8 --plots --save-dir outputs/sethares`
- Genera CSV + PNG con curvas de disonancia por intervalo (1..12) normalizadas para ambas implementaciones.

Notas / Limitaciones conocidas
- Evite las métricas "timbre" y "gower" en el laboratorio: no están implementadas en este repo.
- `pre_process.py` contiene dos clases `ModeloSethares`; Python usa la segunda (solo fundamentales). Esto explica diferencias con literatura. Use los scripts para comparar.
- Para ejecutar notebooks con DB, configure `.env` y ejecute `python -m notebook`.

Sugerencias de validación
- Verificar forma de la curva de disonancia de díadas (mínimos relativos cerca de 0, 3, 4, 7 semitonos; picos alrededor de 1, 2, 6, 10–11 según parámetros), variando `--n-harmonics`.
- Testear invariancia aproximada al trasladar `--base-freq` (p. ej., 220 vs 500 Hz) y observar cambios esperados por la dependencia de `s` en `f_min`.
- Registrar configuración exacta (seed, parámetros, versiones) en el `report.txt` del runner.

