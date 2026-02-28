---
description: Crear un nuevo experimento MDS estandarizado (generación combinatorial → rugosidad → MDS → métricas completas → figuras del report.html)
---

# Workflow: Nuevo Experimento MDS

Genera un experimento reproducible con **métricas completas** y **figuras idénticas al report.html**.

## Paso 1: Describir la población en lenguaje natural

El usuario describe: qué acordes, qué octava/s, qué restricciones.

Ejemplos:
- "Todas las díadas y triadas en C3-B3"
- "21 tríadas diatónicas de Do Mayor con inversiones"
- "Todas las tétradas dentro de dos octavas"
- "Los 7 modos gregorianos sobre C"

## Paso 2: Crear el script del experimento

Crear el archivo en:
```
experiments/<tema>/<nombre_desc>.py
```

Plantilla mínima (usar `exp_utils.py`):

```python
import sys
from pathlib import Path
from itertools import combinations  # o la generación que aplique

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_utils import chords_from_midi, build_entries, run_experiment

OUTPUT_DIR = Path(__file__).parent / "outputs" / "<nombre_experimento>"

# 1. Definir la población (MIDI o directamente (name, intervals, freqs, notes_abs))
combos = list(combinations(range(48, 60), 3))   # ejemplo: triadas C3-B3
chords_raw = chords_from_midi(combos)

# 2. Construir ChordEntry + vectores 12-D
entries, X = build_entries(chords_raw)

# 3. Correr experimento (métricas completas + figuras del report.html)
for metric in ("euclidean", "cosine"):   # ajustar según experimento
    run_experiment(
        entries=entries,
        X=X,
        metric=metric,
        output_dir=OUTPUT_DIR,
        experiment_name="<nombre>",
        n_init=8,
    )
```

**Notas importantes:**
- Cada experimento va en su **propia subcarpeta** dentro de `outputs/`
- `exp_utils.build_entries()` usa **ModeloSetharesVec canónico** (mismo que el GUI)
- `exp_utils.run_experiment()` llama `summarise_embedding_metrics()` del proyecto
- Las figuras generadas son idénticas al `report.html`: `build_scatter_figure`, `build_heatmap_figure`, `build_shepard_figure`

## Paso 3: Ejecutar el script

```bat
cd d:\Documents\GitHub\ChordSpace
.venv\Scripts\python.exe experiments\<tema>\<nombre_desc>.py
```

## Paso 4: Verificar salidas

En `experiments/<tema>/outputs/<nombre_experimento>/`:

| Archivo | Contenido |
|---------|-----------|
| `scatter_<exp>_<metric>.html` | MDS scatter interactivo (mismo que report.html) |
| `heatmap_<exp>_<metric>.html` | Heatmap de distancias |
| `shepard_<exp>_<metric>.html` | Diagrama de Shepard |
| `metricas_<exp>_<metric>.txt` | Todas las métricas (ver abajo) |

## Métricas reportadas (automáticas)

`summarise_embedding_metrics()` calcula:

| Métrica | Descripción |
|---------|-------------|
| `trustworthiness` | Preservación de vecinos locales (0→1, mejor=1) |
| `continuity` | Inverso: vecinos del embedding estaban cerca en el original |
| `knn_recall` | Recuperación de k-NN |
| `rank_corr` | Correlación de rankings de distancias |
| `stress` | Stress de Kruskal-1 (menor=mejor) |
| `shepard_r2` | R² del diagrama de Shepard |
| `shepard_slope` | Pendiente de la regresión Shepard |
| `silhouette` | Separabilidad por cardinalidad |
| `davies_bouldin` | Compacidad de clusters por cardinalidad |
| `relative_rank_error` | Error relativo medio de rankings |
| `var_ratio_dim1/2` | Varianza explicada por cada dimensión |
| `cardinality_logreg_acc` | Separabilidad lineal por cardinalidad |
| `knn_hit_card_N` | % vecinos con misma cardinalidad N |

## Personalización

Para poblaciones no-MIDI (definición manual por intervalos):
```python
chords_raw = [
    ("Cmaj-fund", [4, 3], [261.63, 329.63, 392.0], [60, 64, 67]),
    ...
]
entries, X = build_entries(chords_raw)
```

Para un solo métrica:
```python
run_experiment(entries, X, "euclidean", output_dir, experiment_name="mi_exp")
```

Para usar highlight personalizado:
```python
from experiments.exp_utils import DEFAULT_HIGHLIGHT
from tools.proposals_pipeline.figures import HighlightSettings

mi_highlight = HighlightSettings(threshold=20, size_scale=1.5, size_delta=5.0,
                                  selected_opacity=1.0, fade_factor=0.2)
run_experiment(..., highlight=mi_highlight)
```
