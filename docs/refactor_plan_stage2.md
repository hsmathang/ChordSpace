## Plan de modularización (Etapa 2)

Objetivo: dividir `tools/compare_proposals.py` (≈2.4k líneas) en módulos cohesivos para facilitar pruebas, mantenimiento y futuras mejoras del `report.html`. La idea es introducir una fachada ligera (`compare_proposals.py`) que orqueste componentes especializados.

> **Momento seguro para lanzar la GUI:** Después de los pasos 1‑2 (creación de módulos + adaptadores), antes de migrar lógica sensible como `generate_proposal_figures`. En ese punto el pipeline sigue usando las funciones originales y la GUI/CLI se comportarán igual.

### Orden recomendado

1. **Crear paquete `tools/reporting`**
   - Archivos:  
     - `__init__.py`  
     - `templates.py`: funciones `_load_asset`, `build_report_html`.
     - `metadata.py`: helpers para formatear `run_metadata`.
   - Pasos:
     1. Mover `_load_asset`, `REPORT_TEMPLATE/CSS/JS` y `build_report_html_v2` a `reporting/templates.py`.
     2. Exponer una función `render_report(metrics_df, figures, seeds, run_metadata, output_path)` que encapsule la escritura.
     3. Ajustar `compare_proposals.py` para importar esta función.  
   - Riesgos: rutas relativas de assets. Usar `Path(__file__).with_suffix("")` para resolver `report_assets`.

2. **Separar la capa de “experimento” (`experiment.py`)**
   - Contenido: funciones que coordinan población → escenarios → métricas → figuras.
   - Pasos:
     1. Introducir dataclasses ligeras (p. ej., `PopulationBundle`, `ScenarioSpec`, `SeedMetrics`, `ExperimentResult`).
     2. Mover `build_scenarios`, `generate_seed_results`, `aggregate_seed_results`, `compute_rank`, `generate_proposal_figures`, etc., manteniendo dependencias desde `compare_proposals.py`.
     3. Hacer que `compare_proposals.run_experiment` solo valide args y delegue al nuevo módulo.
   - Riesgos: referencias circulares con `visualisations.proposals.FigureSpec`. Solución: importar tipos solo con `TYPE_CHECKING`.

3. **Aislar carga de poblaciones (`population_pipeline.py`)**
   - Funciones a mover: `load_population_from_queries`, `load_population_from_json`, `build_population_entries`, utilidades para familias/identidades.
   - Exponer API:
     ```python
     def build_population(params: PopulationParams) -> PopulationBundle:
         return PopulationBundle(entries=..., metadata=...)
     ```
   - Riesgos: dependencias con `config`, `pre_process`, `ChordEntry`. Mantener `ChordEntry` en este módulo y exportarlo para otros.

4. **Módulo de métricas y reducciones (`metrics_pipeline.py`)**
   - Separar funciones puras como `build_distance_matrix`, `generate_embedding`, `compute_metrics_for_seed`, `format_value_with_std`.
   - Centralizar importes de `numpy`, `sklearn`, `plotly`, `kruskal_stress_1`.
   - Definir `EmbeddingResult`, `MetricBundle`.

5. **Fábrica de figuras (`figure_factory.py`)**
   - Trasladar `render_card`-like logic y la orquestación de `FigureSpec`.
   - Dejar `visualisations/proposals.py` solo con utilidades específicas de Plotly (hover/text). El nuevo módulo se encargará de agrupar `FigureSpec` por pestañas.

6. **Facade final**
   - `compare_proposals.py` quedará con:
     - `parse_args`, composición de `PopulationParams`, lectura de JSON.
     - Llamada a `population_pipeline.build_population` → `experiment.run_experiment`.
     - Render final: `reporting.render_report`.
   - Añadir pruebas unitarias ligeras para cada módulo (p. ej., `tests/test_population_pipeline.py` con fixtures pequeños).

### Detalle por paso

| Paso | Acción | Resultado esperado |
| --- | --- | --- |
| 1a | Crear paquete `tools/reporting/` con `__init__.py`. | Mantener importaciones limpias. |
| 1b | Mover `_load_asset` y constantes a `reporting/assets.py`. | Se eliminan duplicados al cargar template/CSS/JS. |
| 1c | Extraer `build_report_html_v2` como `render_report(...)`. | `compare_proposals.py` usa una sola llamada. |
| 1d | Ejecutar `python -m compileall tools/reporting`. | Validar que la GUI sigue funcionando aquí (punto seguro). |
| 2a | Definir dataclasses (`PopulationBundle`, `ScenarioResult`). | Normalizar valores compartidos entre funciones. |
| 2b | Mover `build_scenarios`, `generate_seed_results`, `aggregate_seed_results`. | `compare_proposals.run_experiment` delega. |
| 2c | Añadir pruebas unitarias mínimas para `build_scenarios`. | Garantiza que `include_identity` respete flags. |
| 3a | Crear `population_pipeline.py` y mover funciones de carga. | Claridad sobre fuentes (queries, JSON). |
| 3b | Inyectar `population_bundle.metadata` en `run_metadata`. | Reporte muestra alfabeto/filtros sin JSON crudo. |
| 4a | Portar funciones de métricas y embeddings. | Simplifica mocking durante tests. |
| 4b | Añadir caché ligera para distancias (opcional). | Optimización futura. |
| 5a | Implementar `figure_factory.build_tabs(...)`. | `build_report_html_v2` solo recibe HTML ya tabulado. |
| 6a | Limpiar `compare_proposals.py` (<500 líneas). | Más mantenible, apto para publicación académica. |

### Riesgos y mitigaciones

- **Regresiones en GUI:** validar después del Paso 1d y Paso 6a lanzando la GUI (`python ui.py`) y ejecutando un experimento corto.
- **Duplicación de imports pesados:** al crear módulos, mover importaciones top-level a los nuevos archivos para mantener tiempos de carga.
- **Paquete distribuible:** asegurarse de incluir `tools/report_assets/` y los nuevos módulos en `setup.cfg`/`MANIFEST` (si aplica).

Con este plan, la próxima acción es iniciar el Paso 1 (crear `tools/reporting` y mover el renderer). Cuando confirmemos que la GUI sigue operando tras ese paso, continuamos con los módulos de experimento.

