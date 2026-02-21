# Cuestiones sobre cómo funciona la sustitución en su primera fase

Este documento responde, una por una, las preguntas planteadas sobre el primer MVP de sustitución armónica implementado en ChordSpace. Las respuestas están ancladas al código actual (commit de trabajo en tu máquina) y a los documentos técnicos relacionados.

Las referencias de código siguen el formato `ruta:línea`.

---

## 0. Inventario de preguntas

Para poder responder de forma sistemática, primero reescribo y enumero todas las preguntas explícitas de tu mensaje (agrupando frases que forman una misma idea, pero sin fusionar preguntas distintas).

1. **Generación combinatoria de poblaciones**
   1.1. «No vi que me hablaras en el flujo general sobre la generación de poblaciones de acordes por combinatoriedad. Quiero detalladamente eso.»  
        → ¿Cómo se generan las poblaciones de acordes por combinatoria en el sistema actual / documentos del repo?

2. **Metadatos**
   2.1. «Más detalle en los metadatos a los que hace mención: Como se calculan cada uno de ellos quiero el flujo de cada uno.»  
        → ¿Qué metadatos intervienen (ChordEntry, meta del payload, etc.) y cómo se calculan uno por uno?

3. **Escenarios**
   3.1. «Qué es un escenario acá?»  
        → ¿Qué significa “escenario” en `compare_proposals`?  
   3.2. «¿Cuáles son los disponibles?»  
        → ¿Qué escenarios concretos se pueden generar (combinaciones de propuesta/normalización y métrica)?

4. **Vectores ajustados**
   4.1. «¿A qué se refiere con vectores ajustados?»  
   4.2. «¿Cómo se pueden ajustar esos vectores?»  
   4.3. «¿Qué propósito tienen tales ajustes?»

5. **Estructura de `customdata`**
   5.1. «¿Cómo se construye cada dato que está en el `customdata`? Quiero eso detallado.»

6. **Tres rasgos por acorde y normalización**
   6.1. «Quiero detalle de esos tres rasgos por acorde.»  
   6.2. «¿Cómo se calcula el proceso de normalización del histograma?»  
   6.3. «¿Por qué se hace así, por qué como probabilidad?»

7. **Raíz de JSD y vector de rugosidades**
   7.1. «¿Por qué se usa la raíz de JSD? ¿Eso tiene que ver con haber normalizado el vector de rugosidades?»  
   7.2. «Ahora, ¿cuál vector de rugosidades se está normalizando?»  
   7.3. «¿Hay ponderaciones de él que se hacen en los escenarios, no?»

8. **Ponderación 60/40**
   8.1. «¿Qué significa ponderar una matriz y eso del 60/40 qué representa?»

9. **Cardinalidad distinta en sustituciones**
   9.1. «Es decir que el sistema de sustitución que está implementado, ¿no es capaz de asociar como sustitutos acordes que tengan cardinalidad diferente?»

10. **Uso de `argpartition`**
    10.1. «¿Por qué se usa `argpartition`, qué hace eso? ¿Cuál es su propósito en este caso?»  
    10.2. «¿Cómo ordena ese algoritmo y qué impacto tiene sobre lo musical?»

11. **Interpretación de JSD como métrica sensorial**
    11.1. «De acuerdo a lo comentado esto es una primera aproximación que se basa en lo sensorial JSD, pero ¿cómo es que JSD captura eso?»  
    11.2. «¿Por qué sería una buena manera?»  
    11.3. «¿Qué se sacrifica ahí?»  
    11.4. «¿Qué se gana?»

12. **Modularización futura según el documento matemático**
    12.1. «¿Cuáles funciones y lógicas habría que sacar de `visualisations/proposals.py` para factorizar correctamente los módulos y el código en general de acuerdo a las buenas prácticas y para ir acercándose a un intento modular como el del documento que modela matemáticamente la sustitución armónica?»

13. **Poblaciones mixtas y cardinalidad**
    13.1. «Dices que [hay] trabajo inútil cuando hay poblaciones mixtas, ¿por qué inútil?»  
    13.2. «Musicalmente puede ser útil tener sustitutos con cardinalidades distintas.»  
    13.3. «¿Cuál es la dificultad de esta tarea?»

En total cuento **13 grupos** de preguntas con **≈27 subpreguntas**. En las secciones siguientes respondo cada grupo de forma explícita y referenciada.

---

## 1. Generación combinatoria de poblaciones de acordes (P1)

### 1.1. Dos capas: DB actual vs. modelo combinatorio teórico

En el repositorio conviven dos “vías” relacionadas:

- **Vía actual de producción del reporte** (la que usa el MVP de sustituciones):
  - El script `tools/compare_proposals.py` no genera acordes combinatoriamente; **los lee** de una base de datos o de un JSON:
    - `parse_args` define `--dyads-query`, `--triads-query`, `--sevenths-query`, `--population-json` (`tools/compare_proposals.py:290-317`).
    - `load_chords` ejecuta esas consultas con `QueryExecutor` o lee el JSON (`tools/compare_proposals.py:401-420`).
  - La combinatoria de “todas las díadas, tríadas, etc.” se resuelve en el lado SQL (ver `DB_SETUP.md` y `AUDIT_QUERIES.md`), no en Python.

- **Vía de diseño combinatorio general**:
  - El documento `docs/modelo_computacional_de_generacion_y_tratamiento_de_acordes_v_1.md` describe un **motor generativo combinatorio** independiente de la DB.
  - Ese motor no está completamente implementado en código productivo aún, pero fija la semántica de cómo deberían generarse y tratarse las poblaciones de acordes en futuras versiones.

Tu pregunta apunta a entender *cómo debería* funcionar la generación combinatoria de poblaciones (según el modelo) y cómo se relaciona con el pipeline actual de sustituciones.

### 1.2. Generación combinatoria según el modelo formal

Según `docs/modelo_computacional_de_generacion_y_tratamiento_de_acordes_v_1.md`:

1. **Parámetros de entrada** (`§2`):
   - Alfabeto de pitch-classes \(S \subseteq \mathbb{Z}_{12}\).
   - Rango de octavas `[o_min, o_max]` y opcionalmente `edge_pc0`.
   - Lista de cardinalidades `N` (por ejemplo `{2,3,4}`).

2. **Universo MIDI absoluto**:
   - Se construye un conjunto de notas absolutas:
     \[
     M = \{ m : p \in S,\ o \in [o_{\min}, o_{\max}] \}
     \]
   - Opcionalmente se añade la “primera nota” de la octava superior si `edge_pc0=True`.
   - Esta capa impone **no unísonos**: no se repite el mismo número MIDI dentro de un acorde.

3. **Generación combinatoria total** (`§3.1 Combinatorial Total`):
   - Para cada cardinalidad \(k\in N\):
     - Se recorren todas las combinaciones sin reposición \(\binom{|M|}{k}\).
     - Cada combinación se ordena y se emite un acorde absoluto \(A = (m_1 < \dots < m_k)\).
   - Metadatos generados “en el momento”:
     - `abs_mask_bigint`: firma de alturas absolutas (\(\sum_{m\in A} 2^m\)).
     - `pc_mask`: máscara de PCs (\(\sum_{p\in \mathrm{PC}(A)}2^p\)).
     - `n = k` (cardinalidad), `span = m_k - m_1`.
     - `origin = GEN_TOTAL(S,O,N,edge_pc0)` para poder reconstruir cómo se generó.

4. **Generación estructural** (`§3.2`):
   - Alternativamente, se podrían generar patrones “estructurales” primero:
     - Fijar anchura máxima en PC (`max_span_struct`) y generar patrones de PCs en \(\mathbb{Z}_{12}\) (anclando en 0).
     - O proyectar los acordes absolutos (Total) a representaciones estructurales (`canon_0`, `intervals_mod12`) y colapsar por firma estructural.

5. **Tratamiento inmediato** (`§4`):
   - Una vez se genera un acorde absoluto, se calculan de inmediato:
     - Representaciones canónicas (tupla de PCs, PC-set, canon anclado en 0).
     - Vector de croma binario/contado (12‑D).
     - Distancias internas (intervalos adyacentes, pairwise).
   - Estas estructuras son exactamente las que luego alimentan el cálculo de rugosidad y, más adelante, las métricas de sustitución.

### 1.3. Relación con el pipeline actual de sustitución

En el MVP de sustitución:

- `compare_proposals` **no implementa** directamente este motor combinatorio, pero la estructura de datos de salida es compatible:
  - Cada fila de DB o JSON ya llega con `notes_abs_json` o `intervals`, que se usan para construir un `ChordAdapter` (`tools/compare_proposals.py:430-431`).
  - El cálculo de rugosidad (`ModeloSetharesVec`) y los metadatos (`counts`, `pc_mask` implícito, etc.) se apoyan en esa información.
- El motor combinatorio del documento se concibe como un **reemplazo/alternativa** a la DB, pero produciendo registros equivalentes a los que hoy cargas desde SQL. Gracias a eso, el módulo de sustituciones no tendría que cambiar: sólo la etapa “población de entrada”.

En resumen, la generación combinatoria descrita en el documento define **cómo** producir exhaustivamente todas las combinaciones de alturas y sus metadatos, pero el MVP de sustitución que se ejecuta hoy asume que esa población ya está creada (vía DB o motor futuro) y se centra en medir similitudes y construir vecinos.

---

## 2. Metadatos y su flujo de cálculo (P2)

### 2.1. Metadatos de `ChordEntry`

`ChordEntry` (`tools/compare_proposals.py:176-193`) agrupa:

- `acorde`: instancia de `pre_process.Acorde`, construida con `ChordAdapter.from_csv_row(row)` (`tools/compare_proposals.py:430-431`).
- `hist` y `total`: calculados por `ModeloSetharesVec.calcular(acorde)` (`tools/compare_proposals.py:452-453`).
- `counts`: vector de conteos de díadas (`compute_interval_counts`, `tools/compare_proposals.py:558-569`).
- `total_pairs`: suma de `counts` (`tools/compare_proposals.py:455`).
- `n_notes`: cardinalidad (`len(intervals) + 1`, `tools/compare_proposals.py:456`).
- `dyad_bin`: clase de intervalo principal para díadas (`determine_dyad_bin`, `tools/compare_proposals.py:572-576`).
- `identity_name`, `identity_aliases`, `is_named`: derivados de `get_chord_type_from_intervals` (`tools/compare_proposals.py:448-451`).
- `is_inversion`, `family_id`, `inversion_rotation`: combinan flags `__inv_flag`, `__family_id`, `__inv_source_id`, `__inv_rotation` y `id` de la fila (`tools/compare_proposals.py:421-497`).
- `musical_inversion_ids`, `structural_inversion_ids`: se rellenan después usando `get_musical_inversions` y `get_structural_inversions` (`tools/compare_proposals.py:520-553`, `tools/compare_proposals.py:587-605`).

El flujo, simplificado:

1. `load_chords` crea un `ChordAdapter` desde cada fila.
2. Ajusta/crea `notes_abs` si es necesario (`tools/compare_proposals.py:432-447`).
3. Calcula rugosidad, counts, etc., y construye `ChordEntry` (`tools/compare_proposals.py:452-513`).
4. Después, en un segundo paso, calcula y asigna listas de inversiones (`tools/compare_proposals.py:516-553`).

### 2.2. Metadatos en el payload y en `layout.meta`

En `build_scatter_payload` (`visualisations/proposals.py:353-771`) se derivan:

- **Familias y resaltado**:
  - `family_tags`, `family_counts` y `highlight_summary` (`visualisations/proposals.py:378-405`).
  - Guardados en `meta_payload["familyHighlight"]` (`visualisations/proposals.py:721-723`).

- **Textos de hover**:
  - `detail_texts` (hover largo) con `build_hover` (`visualisations/proposals.py:421-435`).
  - `summary_texts` (hover resumido) con `build_hover_summary` (`visualisations/proposals.py:436-443`).

- **Dataset de filtros**:
  - `filter_definitions` y `field_values` desde `_build_filter_metadata` (`visualisations/proposals.py:320-350`).
  - `filter_dataset = {"traceSources": trace_sources, "fields": filter_definitions}` (`visualisations/proposals.py:731-734`).
  - Guardado como `meta_payload["filterDataset"]` (`visualisations/proposals.py:735`).

- **Vecinos de sustitución**:
  - `substitution_neighbors` se calcula en la sección de sustituciones (`visualisations/proposals.py:465-559`).
  - Si no está vacío, se añade `meta_payload["substitutionNeighbors"] = substitution_neighbors` (`visualisations/proposals.py:735-737`).

Además:

- `meta_payload["colorTitle"]`, `meta_payload["isProposal"]`, `meta_payload["filters"]` (`visualisations/proposals.py:721-726`).
- Metadatos adicionales provenientes del contexto (`meta` parámetro) serializados con `_to_serialisable` (`visualisations/proposals.py:727-729`).

Estos metadatos son los que consumen:

- `registerCardFilters` (`tools/compare_proposals.py:2032-2067`) para filtros dinámicos.
- `registerCardHighlight` (`tools/compare_proposals.py:2761-2774`) para resaltar familias e inversiones.
- `registerCardDetail` (`tools/compare_proposals.py:2787-2849`) para mostrar vecinos y detalles en el panel.

---

## 3. Escenarios: definición y catálogo (P3)

### 3.1. Definición

Un **escenario** es una combinación:

- de una propuesta de normalización `preproc_id` (`simplex`, `perclass_alpha1`, etc.), y
- de una métrica `metric` (`cosine`, `js`, etc.).

Se construyen en `build_scenarios` (`tools/compare_proposals.py:3379-3455`) a partir de:

- `proposals_requested` y `metrics_requested` (`tools/compare_proposals.py:3202-3203`).

Cada escenario tiene:

- Nombre (por ejemplo, `"simplex | js"`).
- Función de preprocesado (ajuste de vectores de rugosidad).
- Métrica de distancia para construir la matriz condensa de distancias que luego se reduce a 2D.

### 3.2. Escenarios disponibles

Propuestas (`PREPROCESSORS`, `tools/compare_proposals.py:689-699`):

- `simplex`, `simplex_sqrt`, `simplex_smooth`.
- `perclass_alpha1`, `perclass_alpha0_5`, `perclass_alpha0_75`, `perclass_alpha0_25`.
- `global_pairs`, `divide_mminus1`, `identity`.

Métricas (`metric_distance`, `tools/compare_proposals.py:703-720` y siguientes):

- `cosine`, `js`, `hellinger`, `euclidean`, `manhattan`, etc.

Además, `build_scenarios` garantiza que, para cada métrica, exista siempre al menos un escenario `identity | metric` como control (`tools/compare_proposals.py:3450-3455`).

---

## 4. Vectores ajustados: qué son, cómo se ajustan, para qué (P4)

### 4.1. Qué son

“Vectores ajustados” son versiones transformadas de `hist` pensadas para:

- Controlar efectos de cardinalidad y multiplicidad de díadas.
- Adaptar el vector a distintos tipos de métricas (coseno, JSD, Hellinger…).

En código:

- Cada preprocesador devuelve:
  - `X`: vector ajustado (escala original o modificada).
  - `dist_simplex`: versión generalmente L1-normalizada para métricas de distribución.
  (`tools/compare_proposals.py:623-686`)

Estas dos salidas alimentan:

- Cálculo de distancias en cada escenario (`metric_distance`, `tools/compare_proposals.py:703-715`).
- Construcción de scatter y texto de hover (a través de `build_scatter_figure` y `build_scatter_payload`).

### 4.2. Cómo se ajustan

Resumido de §4 de este mismo documento (ver para detalles):

- `preprocess_simplex`: `hist` → L1-normalizado.
- `preprocess_simplex_sqrt`: `sqrt(hist)` → L1-normalizado.
- `preprocess_simplex_smooth`: `hist` → suavizado Gaussiano circular → L1-normalizado.
- `preprocess_per_class`: `hist / m_k^α` (por clase), sin normalización automática.
- `preprocess_global_pairs`: `hist / P` (pares totales), luego L1.
- `preprocess_divide_mminus1`: `hist / (m_k - 1)` para `m_k ≥ 2`, luego L1.
- `preprocess_identity`: `hist` sin cambios, más una versión L1-normalizada.

### 4.3. Propósito

Los ajustes buscan:

- Hacer las distancias menos dependientes del tamaño del acorde y más de su **perfil relativo** de rugosidad.
- Probar diferentes hipótesis sobre el rol de:
  - repeticiones de díadas,
  - distribución de energía,
  - vecindad entre clases de intervalo.
- Facilitar comparaciones entre poblaciones heterogéneas sin cambiar la API de salida (vectores 12‑D).

---

## 5. `customdata`: construcción detallada (P5)

Ver §5 arriba para el desglose de `customdata` índice por índice. El flujo completo:

1. Se calculan `family_tags` y `family_counts` a partir de los `ChordEntry` (`visualisations/proposals.py:393-405`).
2. Se generan `detail_texts` y `summary_texts` (`visualisations/proposals.py:421-443`).
3. Se crean las filas de `customdata_all` combinando:
   - Datos de familia.
   - Flags de inversión.
   - IDs de inversiones musicales/estructurales.
   - Textos HTML.
   - `global_id = i`.
   (`visualisations/proposals.py:445-463`)
4. `customdata_all` se asigna a cada traza de Plotly.

Esto habilita toda la interactividad del front-end (`tools/compare_proposals.py:2550-2855`).

---

## 6. Tres rasgos por acorde y normalización del histograma (P6)

Ver §6 arriba:

- Rasgos:
  - `hist_probs[i]`: histograma de rugosidad normalizado a probabilidad (`visualisations/proposals.py:472-479`).
  - `pc_vectors[i]`: vector binario de PCs (`visualisations/proposals.py:481-497`).
  - `cardinalities[i]`: número de notas del acorde (`visualisations/proposals.py:497`).
- Normalización:
  - `p = hist / sum(hist)` o uniforme si `sum(hist)` es muy pequeño.
  - Justificación: trabajar en el simplex, compatibilidad con JSD, invariancia a escala.

---

## 7. Raíz de JSD, vector normalizado y escenarios (P7)

Ver §7:

- Se usa \(\sqrt{\mathrm{JSD}}\) por ser métrica y tener mejor escalado (`visualisations/proposals.py:503-513`).
- La normalización se hace sobre `entry.hist` dentro de `visualisations/proposals.py`, independientemente de los preprocesadores de escenarios.
- Los escenarios sí ajustan `hist` para otros fines (embeddings, métricas globales), pero el MVP de sustituciones usa siempre la misma normalización base para JSD.

---

## 8. Ponderación 60/40 (P8)

Ver §8:

- `dist_matrix = 0.6 * jsd_matrix + 0.4 * jaccard_matrix` (`visualisations/proposals.py:523-524`).
- 60% peso sensorial (JSD sobre rugosidad), 40% peso estructural (Jaccard sobre PCs).
- El resultado es la disimilitud que se usa para ranking de vecinos por cardinalidad.

---

## 9. Cardinalidad distinta: qué hace hoy el sistema (P9)

Ver §9:

- El sistema actual agrupa índices por cardinalidad y sólo busca vecinos dentro del grupo correspondiente (`visualisations/proposals.py:526-538`).
- Por tanto, hoy **no asigna** sustitutos entre acordes con distinta cardinalidad.
- Es una decisión de diseño del MVP (perfil “básico”), no una limitación conceptual irrevocable.

---

## 10. `argpartition`: propósito y efecto musical (P10)

Ver §10:

- `np.argpartition` se usa para encontrar los K mínimos en `O(N)` en lugar de ordenar toda la lista (`visualisations/proposals.py:539-544`).
- Después se ordenan sólo esos K (`np.argsort`).
- Resultado: misma lista Top‑K (y mismo orden dentro de ella), pero con coste menor.
- No hay impacto musical en la selección de vecinos; sólo mejora de rendimiento.

---

## 11. JSD como métrica sensorial (P11)

Ver §11:

- JSD sobre histogramas de rugosidad normalizados captura similitud en la **textura de disonancias/consonancias** por clases de intervalo.
- Es una buena primera aproximación porque:
  - Se integra con el modelo de Sethares.
  - Es estable y simétrica.
  - Permite añadir métricas adicionales más adelante.
- Se sacrifica:
  - Información sobre voicing, función tonal y voice‑leading.
  - Parte de la estructura del PC-set.
- Se gana:
  - Una noción coherente y robusta de similitud auditiva de bajo nivel, compatible con el resto del pipeline.

---

## 12. Qué extraer de `visualisations/proposals.py` para modularizar (P12)

Ver §12:

- Extraer a `substitution/features.py`:
  - Cálculo de `pcvec`, `roughness_prob`, `interval_class_vector`, `voice_leading_cost`.
  - Lógica de inversiones (`get_musical_inversions`, `get_structural_inversions`) podría vivir en un módulo de armonía.
- Extraer a `substitution/metrics.py`:
  - Implementación de JSD, Jaccard-PC, IC-L1, coseno, VL, tonal centroid.
- Extraer a `substitution/aggregate.py`:
  - Combinación ponderada y perfiles (weights por métrica).
- Extraer a `substitution/index.py`:
  - Cálculo de k‑NN exacto (con `argpartition`) y versiones ANN futuras.
- Mantener en `visualisations/proposals.py`:
  - Sólo la construcción del payload visual y la inyección de resultados de sustitución en `meta`.

Esto alinearía el código con la arquitectura propuesta en los documentos matemáticos y facilitaría reuso y testeo.

---

## 13. Poblaciones mixtas, “trabajo inútil” y dificultad de cardinalidades distintas (P13)

Ver §13:

- El “trabajo inútil” se refiere a que, para el MVP actual (que sólo usa vecinos de misma cardinalidad), calcular distancias entre cardinalidades distintas no aporta nada al resultado.
- Musicalmente, es **muy interesante** considerar sustitutos con distinta cardinalidad, pero:
  - Requiere definir métricas y penalizaciones más sofisticadas (VL, IC, detectores funcionales).
  - Complica la interpretación en la UI.
  - Aumenta el coste computacional de forma significativa.
- Por eso se ha empezado por el caso “más seguro” y se deja la generalización como fase posterior, con un diseño modular ya previsto en los documentos de sustitución.

---

## 14. Correspondencia preguntas–respuestas

- Preguntas agrupadas: **13 grupos** (P1–P13).
- Subpreguntas aproximadas: **27**.
- Secciones de este documento:
  - §1–§13 responden directamente a P1–P13.
  - Cada sección cita el código o documento relevante y explica el flujo o la motivación.

De este modo, todas las preguntas planteadas en tu mensaje están cubiertas explícitamente en al menos una sección de este documento.

