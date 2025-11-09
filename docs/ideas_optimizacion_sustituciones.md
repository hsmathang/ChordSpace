# Ideas para optimizar el cálculo de sustituciones

Este documento resume propuestas para reducir el tiempo de generación del reporte al calcular sustituciones armónicas. Las referencias de línea corresponden al commit actual.

## 1. Vectorización completa (ya aplicada parcialmente)

- **Archivo:** `visualisations/proposals.py:466-570`
- **Descripción:** Eliminamos los bucles doble `for` al calcular las matrices JSD y Jaccard usando broadcasting NumPy. Se puede seguir optimizando:
  - Reutilizar `hist_probs` normalizados entre escenarios (serializarlos en disco).
  - Evitar el almacenamiento de dos matrices completas cuando sólo se necesitan filas (usar `np.memmap` o cálculo por bloques).

## 2. Filtrado previo por cardinalidad

- **Archivo:** `visualisations/proposals.py:533-570`
- Actualmente calculamos distancias para todos los pares y luego filtramos por cardinalidad. Se puede reducir memoria y tiempo calculando JSD/Jaccard por grupos (una matriz por cardinalidad).
- Pseudocódigo:
  ```python
  for card, idxs in card_groups.items():
      sub_hist = hist_probs[idxs]
      sub_jsd = pairwise_jsd(sub_hist)
      # Rellenar dist_matrix solo para (idx_i, idx_j)
  ```

## 3. Amortizar resultados repetidos

- **Situación:** distintos escenarios (mismos acordes con distintos colores) vuelven a calcular las mismas matrices.
- **Idea:** cachear en `outputs/.../cache/` un archivo comprimido con `hist_probs`, `pc_vectors` y matrices resultantes usando el hash de `entries`.
  - Hook: justo antes de la sección `# --- Sustituciones` se podría comprobar si existe un archivo `cache/<scenario_hash>.npz`.

## 4. Aproximación con ANN

- **Idea:** en lugar de calcular toda la matriz `N×N`, construir un índice Annoy/FAISS sobre el vector `features = concat(p_hist, pc_vec)` con pesos adecuados.
- **Impacto:** tiempo O(N log N), consultas O(log N). El resultado aproxima la métrica compuesta pero puede calibrarse.
- **Integración:** generar el índice en la fase previa al reporte y persistirlo junto con los datos del escenario.

## 5. Reducir N evaluado

- **Estrategias:**
  - Limitar sustituciones al baseline + propuesta activa (ignorar otros paneles).
  - Permitir un parámetro CLI `--max-substitution-points` para muestrear si `N > L`.

## 6. Reutilizar en front-end

- **Idea:** exportar una matriz de vecinos en JSON independiente por escenario para no recalcular si el reporte se vuelve a generar sin cambios en la población.
- **Archivo:** `tools/compare_proposals.py` podría leer `substitution_neighbors.json` desde el directorio del run.

---

Estas ideas están pensadas para iterarse después de verificar el MVP. Actualmente, la vectorización (1) ya redujo el costo principal; los siguientes pasos naturales serían (2) y (3) para escenarios grandes.
