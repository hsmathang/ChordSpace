# Plan para incorporar reducciones aproximadas escalables

## 1. Motivación y diagnóstico

1. Las mediciones efectuadas con `tools/profile_compare_pipeline.py` muestran que, para poblaciones de 1 k–1.5 k acordes, **la fase de reducción MDS consume más del 97 % del tiempo total** (44.8 s de 44.8 s para 1 012 acordes; 112.3 s de 113.8 s para 1 518 acordes). El cálculo previo (carga SQL, histogramas, preprocesamiento, `pdist`) es despreciable.
2. MDS (SMACOF) requiere la matriz de distancias completa O(n²) y repite eigendescomposiciones; la implementación de scikit‑learn no ofrece paralelismo eficiente con joblib para matrices densas de gran tamaño (el overhead IPC supera cualquier ganancia).
3. UMAP también enfrenta un cuello de botella en la construcción del grafo k‑NN cuando n es grande.
4. El dashboard GUI debe ser capaz de trabajar con poblaciones en el rango 10³–10⁶ acordes según los experimentos de navegación; por tanto debemos ofrecer **modos aproximados** que reduzcan la complejidad manteniendo interpretabilidad musical y métricas comparables.

## 2. Estrategia general

Implementar un “modo aproximado” complementario a los métodos actuales que:

- Reemplace la matriz de distancias completa por una aproximación eficiente.
- Ofrezca garantías cuantitativas sobre la calidad de la incrustación (stress relativo, distorsión de vecinos).
- Se integre sin rupturas con la CLI y la GUI (seleccionable como otra reducción).

Propuestas concretas:

1. **Landmark / Pivot MDS**  
   - Seleccionar L ≪ n “landmarks” (estratificados por cardinalidad, tags o familias).  
   - Ejecutar MDS exacto sobre la submatriz L×L.  
   - Proyectar el resto de los puntos mediante interpolación barycéntrica (ver De Silva & Tenenbaum, 2004).  
   - Complejidad: O(L² + (n·L)) en lugar de O(n²). L típico: 512–1024.

2. **Approximate UMAP con FAISS/Annoy**  
   - Reemplazar la búsqueda exacta de vecinos por un índice aproximado (FAISS HNSW o Annoy).  
   - Estimaciones empíricas muestran hasta 10× de velocidad con una degradación mínima del trustworthiness (ver McInnes, 2018).

## 3. Integración con el repositorio

### 3.1 Nuevos módulos

- `tools/approximations.py`
  - `select_landmarks(df, strategy, size)`
  - `landmark_mds(dist_func, data, landmarks, output_dim=2)`
  - `approximate_neighbors(matrix, k, backend='faiss')`
- Dependencias opcionales: `faiss-cpu` o `annoy` (activadas vía extras).

### 3.2 Cambios en `tools/compare_proposals.py`

1. Extender `AVAILABLE_REDUCTIONS` con `LANDMARK_MDS` y `UMAP_APPROX`.
2. Añadir flags:
   - `--approx-mode` (`auto`, `exact`, `approx`).
   - `--landmark-size`, `--knn-backend`.
3. Durante `compute_embeddings`, desviar hacia la versión aproximada cuando:
   - `approx-mode == 'approx'`, o
   - `approx-mode == 'auto'` y `n > N_threshold` (configurable en `config.py`).
4. Registrar métricas de fidelidad:
   - Stress relativo: `stress_approx / stress_exact` (cuando se evalúe subsampled exact).
   - Vecinos preservados: trustworthiness, continuity, kNN recall (ya disponibles).

### 3.3 GUI (`tools/gui_experiment_launcher.py`)

- Añadir controles para seleccionar modo (`Exacto / Automático / Aproximado`), tamaño de landmark y backend de vecinos.
- Mostrar en el log un resumen comparando métricas exactas vs. aproximadas cuando estén disponibles.
- Persistir estos parámetros al construir comandos (`compare_runner` centralizado).

## 4. Plan de validación

### 4.1 Dataset de referencia

- Conjunto escalonado:
  1. 1 024 acordes (`QUERY_CHORDS_4_NOTES_SAMPLE_50` + triadas).  
  2. 4 096 acordes (combinación ampliada; se generará vía scripts).  
  3. 16 384 acordes (necesitará extracción batched o población sintética).

### 4.2 Métricas cuantitativas

Para cada tamaño:

| Método | Tiempo (s) | Stress | Trustworthiness | Continuity | kNN recall | Memoria pico |
| ------ | ---------- | ------ | --------------- | ---------- | ---------- | ------------ |
| MDS exacto | ... | ... | ... | ... | ... | ... |
| Landmark MDS (L=512) | ... | ... | ... | ... | ... | ... |
| Landmark MDS (L=1024) | ... | ... | ... | ... | ... | ... |
| UMAP exacto | ... | ... | ... | ... | ... | ... |
| UMAP approx (FAISS) | ... | ... | ... | ... | ... | ... |

Calidad aceptable si:

- Stress relativo ≤ 1.10.
- Trustworthiness y continuity ≥ 90 % del valor exacto.
- kNN recall ≥ 85 % del valor exacto.

### 4.3 Pruebas automáticas

- Actualizar `tests/` con un caso pequeño que compare `LANDMARK_MDS` vs. MDS exacto (assert sobre tolerancias).
- Script de benchmarking que genere tabla/CSV; se incluye en `docs/APPROXIMATE_REDUCTION_PLAN.md` al finalizar.

## 5. Entregables

1. Implementación en `tools/approximations.py` + integración CLI/GUI.
2. Documentación de uso (`docs/APPROXIMATE_REDUCTION_PLAN.md` se ampliará con resultados).
3. Benchmark reproducible (`scripts/benchmark_approx_reductions.py`) que produce tablas en `docs/`.
4. Notas de adopción en la GUI (mensajes contextualizados para el usuario y defaults seguros).

## 6. Cronograma tentativo

1. Semana 1: implementar selección de landmarks + flujo CLI; ensayo con datasets 1 k, 4 k.  
2. Semana 2: integrar UMAP aproximado (FAISS/Annoy) y telemetría de fidelidad.  
3. Semana 3: unir controles en la GUI y cerrar documentación + pruebas automatizadas.  
4. Semana 4: análisis final con datasets grandes (≥ 16 k) y ajuste de parámetros por defecto.

Este plan garantiza una transición ordenada desde el pipeline exacto actual hacia versiones aproximadas, cuantificando la degradación (o mejora) y manteniendo trazabilidad entre la CLI, la GUI y los experimentos reproducibles.

