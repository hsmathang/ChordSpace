## Canal de generación de reportes (`report.html`)

Este documento resume cómo el repositorio arma el reporte HTML mostrado en la GUI (pestaña “Compare Proposals”) y en el CLI `tools/compare_proposals.py`. Sirve de guía para la futura modularización.

### 1. Entradas (GUI y CLI)

- **GUI**  
  `ui.py` invoca comandos registrados en `tools/gui_commands.py`. El comando de comparación (`RunComparisonCommand`) recibe un `LauncherState` con:
  - `params`: valores seleccionados en el panel (queries, modo combinatorial/DB, flags de baseline, etc.).
  - `datasets`: poblaciones resultantes (DataFrame con acordes y metadatos combinatorios).
  - `metadata`: rutas de salida, timestamp, filtros aplicados.
  El servicio configurado termina llamando a `tools.compare_proposals.main` con argumentos serializados (JSON para la población cuando no proviene de la BD).

- **CLI**  
  `tools/compare_proposals.py` (líneas ~1896-2115) parsea flags (`--population-json`, `--disable-baseline-identity`, `--run-metadata`, etc.), construye/extrae datasets y entrega todo a `run_experiment(...)`. El CLI y el GUI convergen en la misma función para generar métricas, figuras y el HTML final.

### 2. Flujo interno (`tools/compare_proposals.py`)

1. **Carga de población** (líneas 320-620):  
   - `build_population_from_queries` carga acordes desde SQL (`config.QUERY_*`).  
   - `build_population_from_json` se usa cuando la GUI exporta combinatorias personalizadas.  
   - Se instancian `ChordEntry` con histograma crudo (`ModeloSetharesVec`), totales y banderas (identidad, familia, inversión).

2. **Preprocesadores y escenarios** (líneas 650-1050, 2170-2210):  
   - `PREPROCESSORS` define funciones como `preprocess_identity`, `preprocess_global`, etc.  
   - `SCENARIOS` describe combinaciones `[proposal, metric, reduction]`.  
   - `build_scenarios(..., include_identity=not args.disable_baseline_identity)` asegura que el baseline “identity” solo se agrega cuando no se desactiva (el flag de GUI/CLI controla esto).

3. **Embeddings y métricas** (líneas 1050-1650):  
   - Por cada semilla se calculan matrices de distancia (`build_distance_matrix`), reducciones (`generate_embedding`) y métricas de calidad (`compute_trustworthiness`, `kruskal_stress_1`, etc.).  
   - `aggregate_seed_results` combina resultados por escenario en un `metrics_df` que luego se ordena con `compute_rank`.

4. **Figuras Plotly** (líneas 1200-1670 + `visualisations/proposals.py`):  
   - `generate_proposal_figures` crea `FigureSpec` por escenario/modo de color.  
   - Cada `FigureSpec` serializa la figura (para `<div>` + JSON) y se agrupa por pestañas (reducción → métrica).

5. **Generador del reporte** (`build_report_html_v2`, líneas 1433-1890):  
   - Construye el ranking global (`table_html`).  
   - Compone pestañas exteriores (reducciones) e interiores (métricas), renderizando tarjetas para baseline/propuestas.  
   - Ensambla metadatos (`run_metadata`): selección actual, detalles de población combinatorial (alfabeto, cardinalidades, filtros), fuentes SQL, etc.

### 3. Plantilla externa (`tools/report_assets/`)

Durante esta etapa se extrajo todo el HTML estático, CSS y JS del script principal:

- `template.html` define esqueleto y placeholders:  
  - `__CSS__`: contenido de `styles.css`.  
  - `__TABLE_HTML__`: ranking global.  
  - `__METADATA_HTML__`: tablas con parámetros de población.  
  - `__TABS_HTML__`: pestañas anidadas con figuras.  
  - `__SCRIPT_JS__`: lógica de tabs e interacciones.

- `styles.css` (≈6 KB) ahora describe tablas, tarjetas, layout responsivo, y reutiliza la estética de la “tablita” del ranking.
- `script.js` mantiene la lógica de tabs y toggles que ya existía (no se mostró en esta etapa).

`compare_proposals.py` carga estos recursos al inicio (`REPORT_TEMPLATE`, `REPORT_CSS`, `REPORT_JS`) y, si faltan, usa un fallback HTML mínimo. De esta forma el CLI/GUI siguen funcionando aunque el paquete se distribuya sin los assets.

### 4. Metadata mostrada

`run_metadata` (CLI flag `--run-metadata` y objeto que la GUI escribe en disco) debe contener:

```json
{
  "selection": {
    "rows_selected": 1200,
    "rows_available": 3400,
    "mode": "compare",
    "payload_path": ".../population.json",
    "payload_reason": "combinatorial mix"
  },
  "population": {
    "descriptors": [
      {
        "mode": "combinatorial",
        "rows": 1200,
        "combinatorial": {
          "alphabet": ["C", "D#", "F#"],
          "cardinalities": [3, 4],
          "octave_min": 3,
          "octave_max": 5,
          "structural_mode": true
        },
        "filters": {
          "label": "triadas consonantes"
        }
      }
    ]
  }
}
```

`build_report_html_v2` transforma esa estructura en dos tablas (“Selección” y “Fuente n”) dentro del bloque de metadatos.

### 5. Próximos pasos (Etapa 2+)

Con la plantilla aislada, el siguiente objetivo es:

1. Extraer el pipeline de población/métricas/figuras en módulos dedicados para poder probar cada parte sin abrir la GUI.
2. Añadir una sección de “Parámetros combinatoriales” en el reporte reutilizando la estructura anterior, evitando mostrar JSON puro y adoptando el estilo de tabla existente.
3. Aprovechar el flag `--disable-baseline-identity` (y su contraparte en la GUI) para no generar la figura MDS Identity cuando no se requiere.

Este documento debe mantenerse actualizado conforme avancemos en la refactorización.

