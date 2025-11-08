# ChordSpace – Flujo de datos y funcionamiento de la GUI

Este documento describe, de forma práctica, cómo fluye la información en la interfaz gráfica (Experiment Launcher) desde la elección/generación de la población hasta la construcción del reporte HTML de comparación (scatter plots) en la pestaña “Parámetros de comparación”. También detalla las formas de combinar y filtrar poblaciones, y los pasos internos que ejecuta la herramienta.

## 1. Conceptos generales

- Población: conjunto de acordes con columnas compatibles con la tabla `chords` de la DB (id, n, interval, notes, bass, octave, tag, code, span, abs_mask_int, abs_mask_hex, notes_abs_json, etc.).
- Población temporal: vista de previsualización con la configuración actual (DB o combinatorial, más filtros). No afecta a los experimentos hasta que se añade a la población final.
- Población final: unión (con deduplicado) de una o más poblaciones temporales. Es la que se usa para ejecutar comparaciones o experimentos.
- Vista rápida: modo de previsualización que limita la cantidad de filas por fuente (LIMIT N) y reduce drásticamente el consumo de memoria.

## 2. Modos de construcción de la población

### 2.1. Desde la Base de Datos (DB)

1) Selección de fuentes:
   - Consulta base (p. ej. `QUERY_DYADS_REFERENCE`, `QUERY_CHORDS_3_NOTES_ALL`).
   - Poblaciones conjuntas A/B/C (presets o consultas personalizadas del registro).
   - Filtros dinámicos (Cardinalidades, Span, Máx. intervalo interno, Pitch classes incl./excl., patrones de intervalos, códigos, etc.).

2) Vista rápida (opcional, por defecto activa):
   - Añade `LIMIT N` al SQL de cada fuente.
   - Pide solo el perfil de columnas configurado (por defecto, `VISUAL`).
   - Muestra el estado en el “Registro de población” con líneas `[db]` (tiempos por etapa y memoria aproximada).

3) Combinación: se concatenan base + pops A/B/C + filtros y luego se aplican transposiciones (si están activadas) y deduplicado.

### 2.2. Generación combinatorial

1) Universo MIDI: se arma con el alfabeto de pitch classes (0–11) y el rango de octavas (e.g. 3–4), produciendo todas las notas MIDI posibles.
2) Combinación: se generan todas las combinaciones (sin repetición) para las cardinalidades solicitadas (e.g. triadas y cuatríadas).
3) Vista REAL (por defecto):
   - `notes`: PCs reales (mod 12) en el orden del voicing.
   - `bass`: raíz real (PC real) y `octave`: octava real del bajo.
   - `interval`: diferencias adyacentes en MIDI (describe el voicing).
   - `code`: código por PCs reales (alfabeto 0123456789AB).
   - `frequencies`: Hz para las notas MIDI.
   - Metadatos: `__root_midi`, `abs_mask_midi` (máscara de MIDI), `__source__='GENERATED:COMBINATORIAL'`.
4) Vista normalizada (para compatibilidad interna):
   - Se conserva como columnas auxiliares `__norm_interval`, `__norm_notes`, `__norm_code`, `__norm_bass` (ancladas a 0).
   - `abs_mask_int` se calcula sobre la representación normalizada para estabilizar dedupe y evitar enteros gigantes.
5) Toggle en la GUI: “Anclar vista a 0 (normalizada)” permite alternar entre vista REAL y normalizada sin regenerar.
6) Filtros de voicing (cerrado, etc.) se aplican desde Filtros dinámicos (no como opción separada en combinatorial).

## 3. Filtros dinámicos (DB y combinatorial)

- Cardinalidades (n), Span (min/max), Máx. intervalo interno (voicing cerrado), Inclusión/Exclusión de pitch classes.
- Modos de intervalos: `exact` (patrones completos), `subseq` (subsecuencia), `any_value` (cualquier valor en la lista).
- Códigos absolutos (p. ej., 0135679AB0).
- Para combinatorial y DB, los filtros se aplican sobre la población seleccionada en memoria; en modo DB, siempre que es posible se empujan condiciones al SQL para reducir el volumen de datos.

## 4. Población temporal y final

1) “Generar Población Temporal”: construye y muestra la tabla con la vista seleccionada.
2) “=> Añadir a Población Final”: concatena a la final y ejecuta deduplicado.
3) Deduplicado (tools/population_utils.py):
   - Preferencia por `abs_mask_int` (normalizado) y, cuando existe, se añade `__root_midi` a la clave para distinguir acordes en diferentes octavas reales.
   - Si faltan máscaras, fallback por (`code`,`interval`) normalizados. 
4) Contadores: `Total`, break-down por `n`, conteo de triadas/sevenths “named” (coinciden con intervalos canónicos).

## 5. Transposiciones y expansión por escala

- Transposiciones (si están activas) se aplican a la población combinada antes del dedupe: se generan nuevas filas por cada paso solicitado y se marca `__transposition__` y `__source__`.
- Expansión por escala (generate_scale_population) toma una población base y produce transposiciones que pertenecen a la escala, anotando `__scale_*`.

## 6. Preparación para comparación (Parámetros de comparación)

1) Selección: la comparación toma la población final; si el usuario no selecciona filas en la tabla, se usan todas las disponibles.
2) Inline SQL vs payload JSON:
   - Si la selección tiene IDs válidos y no es excesiva, se compone un SQL inline con `IN/ANY`.
   - Si hay IDs duplicados, familias de inversiones, demasiados IDs o columnas especiales, se exporta la población a JSONL (payload) y el comparador lee ese archivo.
3) `tools/compare_proposals.py`:
   - Carga entradas (desde DB o JSONL), arma histogramas y aplica propuestas/metricas/reducciones.
   - “Proposals”: transformaciones (e.g., Media por clase), “Metrics”: distancia, “Reductions”: MDS/UMAP.
   - “build_scatter_payload” genera datos para scatter plots; se apoya en `visualisations/proposals.py`. Si `chroma` no existe o es NaN, se deriva de `notes`.
4) Salida: carpeta `outputs/gui_runs/YYYYMMDD_HHMMSS/compare_.../report.html` con scatter y resúmenes.

## 7. Instrumentación y logs

- En modo DB, el launcher emite logs por etapa: `[db] base …`, `[db] pop …`, `[db] filtros …`, `[db] concat …`, `[db] transposiciones …`, `[db] total …`.
- También muestra filas y memoria aproximada (MB) por bloque para diagnosticar cuellos de botella.
- Vista rápida (LIMIT) figura explícitamente en el log cuando está activa.

## 8. Buenas prácticas

- Para previsualizar poblaciones grandes, mantener activa la “Vista rápida”.
- Usar `ColumnProfile.MINIMAL` cuando solo se necesita contar/inspeccionar intervalos/IDs.
- Para evitar “voicings abiertos” en combinatorial, usar “Máx. intervalo interno ≤ 7”.
- Mezclas DB + combinatorial: si se exporta a JSONL, asegurarse de que columnas como `chroma` estén presentes (la generación combinatorial ya las incluye).

## 9. Mapa de flujo (resumen paso a paso)

1) Usuario elige modo (DB | Combinatorial) y ajusta parámetros.
2) (Opcional) Activa Vista rápida (LIMIT por fuente).
3) Genera Población Temporal → tabla.
4) (Opcional) Aplica Filtros dinámicos y/o Transposiciones.
5) Añade a Población Final → dedupe.
6) Pestaña “Parámetros de comparación”:
   - Selecciona propuestas/métricas/reducciones.
   - Si la población cumple criterios (IDs válidos, SQL no excesivo) → inline SQL; si no → export JSONL.
   - Ejecuta comparador (`tools/compare_proposals.py`).
7) Se genera el reporte HTML con scatter y métricas.

