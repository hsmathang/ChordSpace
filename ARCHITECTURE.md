# Arquitectura del Proyecto

Este documento describe la arquitectura actual y la arquitectura objetivo del proyecto ChordSpace, diseñada para garantizar reproducibilidad, modularidad y rigor científico.

## 1. Visión General: Capas del Sistema

El sistema se organiza en cuatro capas estrictas. El flujo de datos siempre viaja desde la población hacia el reporte, pasando por transformaciones de dominio y servicios de orquestación.

### Diagrama de Flujo Lógico

```mermaid
graph TD
    A[Population Source (File/DB)] -->|Fetch| B(Domain: Population)
    B -->|Vectorize| C(Domain: Roughness Models)
    C -->|Calculate| D(Domain: Distance Matrices)
    D -->|Reduce| E(Domain: Embeddings / Reductions)
    E -->|Measure| F(Domain: Quality Metrics)
    F -->|Visualize| G(Services: Reporting / Visualization)
    G --> H[Report Artifacts (HTML, CSV, Plots)]
```

## 2. Definición de Capas

### A. Dominio Puro (Core Domain)
*Ubicación: `services/domain/` (propuesto) o módulos específicos.*
Contiene la lógica científica y las definiciones de datos. No depende de bases de datos, GUI ni CLI.

*   **Generación Combinatoria**: Reglas para crear poblaciones (`services/combinatorial_generator.py`).
*   **Modelos de Rugosidad**: Implementación de Sethares y variantes (`pre_process.py`, `rugosidad_model/`).
*   **Vectorización**: Transformación de acordes a vectores matemáticos.
*   **Reducciones y Métricas**: Algoritmos (MDS, UMAP) y cálculos de calidad (Stress, Trustworthiness).
*   **Configuraciones (Tipos)**:
    *   `PopulationConfig`: Parámetros para generar/cargar una población.
    *   `ExperimentConfig`: Configuración completa de un experimento (población + modelo + reducción + parámetros).
    *   `VisualizationConfig`: Qué mostrar en el reporte.
*   **Resultados (Tipos)**:
    *   `ExperimentResult`: Contenedor estructurado con la población, embeddings, métricas y metadatos.

### B. Servicios de Aplicación
*Ubicación: `services/`*
Orquestan el dominio para cumplir casos de uso. Son el punto de entrada para GUI y CLI.

*   **`services/population_store.py`**: Abstracción para cargar/guardar poblaciones.
    *   Interfaz: `fetch_population(sources)`, `ingest_population(df, metadata)`.
    *   Implementaciones: `FilePopulationStore` (default, JSONL/CSV), `DatabasePopulationStore` (opcional).
*   **`services/space_experiments.py`**:
    *   `run_experiment(config: ExperimentConfig) -> ExperimentResult`
    *   Encapsula el pipeline: Carga -> Vectorización -> Distancias -> Reducción -> Métricas.
*   **`services/space_visualization.py`**:
    *   Generación de figuras puras (Plotly objects) a partir de `ExperimentResult`.
    *   Ensamblaje de `report.html` usando `VisualizationConfig`.

### C. Infraestructura
Implementaciones concretas de persistencia y acceso a recursos.
*   Manejo de archivos (JSONL, CSV, NPY).
*   Conexión a PostgreSQL (solo si se usa `DatabasePopulationStore`).

### D. Interfaces (Front-ends)
Capas finas que capturan la intención del usuario y llaman a los Servicios.

*   **CLI (`tools/`)**: Scripts que parsean argumentos, construyen Configs y llaman a `run_experiment`.
*   **GUI (`ui/`)**: Interfaz Tkinter que construye Configs visualmente y delega la ejecución a los mismos servicios que el CLI. **No contiene lógica de negocio.**

## 3. Entidades Clave (Contratos)

### `ExperimentConfig`
```python
@dataclass
class ExperimentConfig:
    population: PopulationConfig  # Definición o ID de la población
    roughness: RoughnessConfig    # Parámetros del modelo (curvas, normalización)
    reduction: ReductionConfig    # Algoritmo (MDS), dimensiones, n_init
    execution: ExecutionConfig    # Semillas, n_jobs, determinismo
```

### `ExperimentResult`
```python
@dataclass
class ExperimentResult:
    config: ExperimentConfig
    population_data: pd.DataFrame
    embeddings: Dict[str, np.ndarray] # Por semilla/método
    metrics: pd.DataFrame             # Stress, etc.
    artifacts_path: Path              # Directorio con resultados en disco
```

## 4. Plan de Refactorización y Protocolo de Seguridad

### Protocolo de No-Regresión
1.  **Congelar Referencia**: Antes de empezar, ejecutar un experimento estándar (e.g., `diadas_estructurales_octava3`, MDS, Sethares) y guardar:
    *   `metrics.csv`
    *   `report.html` (layout visual)
2.  **Verificación Continua**: Tras cada cambio de arquitectura:
    *   Ejecutar el mismo experimento vía el nuevo código.
    *   Comparar métricas (deben ser idénticas o dentro de tolerancia flotante).
    *   Verificar que el reporte se genera correctamente.

### Pasos
1.  **Consolidar Dominio**: Definir las clases de Configuración y Resultado.
2.  **Interfaz de Datos**: Implementar `PopulationStore` con soporte de metadatos.
3.  **Servicio de Experimentos**: Migrar lógica de `compare_proposals.py` a `space_experiments.py`.
4.  **Servicio de Visualización**: Desacoplar generación de figuras de la escritura de HTML.
5.  **Refactor CLI**: `compare_proposals.py` pasa a ser un consumidor de `space_experiments`.
6.  **Refactor GUI**: Conectar GUI a los nuevos servicios.
