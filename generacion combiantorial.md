# Generacion combiantorial para validacion experimental en ChordSpace

## 0. Respuesta corta a tu duda central

Si, tu idea es correcta y es viable en este repo: como el generador combinatorial puede producir cualquier acorde como conjunto ordenado de notas MIDI, puedes reconstruir las poblaciones de validacion de cada experimento traduciendo las reglas del paper/corpus a restricciones de generacion.

La condicion importante es esta:

- Si el paper define estimulos exactos (acordes concretos, mismo voicing/registro), hay que reproducir esos mismos acordes.
- Si el paper solo define una clase de estimulos (por ejemplo "triadas en rango SATB"), basta una heuristica bien documentada y reproducible.

Este documento te deja ese puente formalizado.

---

## 1. Pipeline real implementado en tu repositorio

### 1.1 Representacion base del acorde

En tu marco actual, el objeto base es un `pitch chord` en MIDI absoluto (no pc-set puro):

- El orden es estrictamente creciente (`n1 < n2 < ... < nm`).
- No hay unisonos dentro del mismo acorde.
- La octava/registro se conserva y afecta los features (rugosidad, etc.).

Referencias del repo:

- `metodologia_temporal/ESTRUCTURA_MATEMATICA_DETALLADA.md`
- `metodologia_temporal/evaluacion-experimentos.md`

### 1.2 Generacion combinatorial (codigo actual)

Funcion principal:

- `services/combinatorial_generator.py` -> `generate_combinatorial_chords(alphabet, octave_min, octave_max, cardinalities, structural_mode)`

Flujo:

1. Se construye el universo MIDI desde `alphabet` (PCs 0..11) y rango de octavas `[octave_min, octave_max]`.
2. Se agrega una nota de borde en `octave_max + 1` para `boundary_pc = min(alphabet)` (si cae en 0..127).
3. Para cada cardinalidad `k`, se recorren todas las combinaciones sin repeticion (`itertools.combinations`).
4. Cada acorde se transforma en registro completo con vista real y vista normalizada:
   - Vista real: `notes`, `bass`, `octave`, `interval`, `frequencies`, `notes_abs_json`, `abs_mask_midi`, `__root_midi`.
   - Vista normalizada/anclada: `__norm_interval`, `__norm_notes`, `__norm_code`, `__norm_bass`.
5. Se calcula `__struct_semitones` y `__structure_id` para identificar estructura de voicing.
6. Si `structural_mode=True`, se colapsa por patron estructural `(k, offsets)` y se emite una fila canonica por estructura.

Notas de fidelidad:

- El comportamiento de "nota de borde" en codigo actual es automatico (no toggle explicito).
- En modo estructural, el catalogo queda deduplicado por estructura.

### 1.3 Filtros de subconjuntos (control de poblaciones)

Servicio:

- `services/population_filter.py` -> `filter_dataframe(df, ChordFilters)`

Filtros disponibles (aplican sobre DB y combinatorial):

- Cardinalidad (`n`)
- Span (`span_semitones` min/max)
- Maximo intervalo interno
- Pitch classes incluidas/excluidas con modos:
  - `contains_all`
  - `contains_any`
  - `subset_of`
- Patrones de intervalos:
  - `exact`
  - `subseq`
  - `any_value`
- Codigos (`code`)

Definicion de estructura de filtros:

- `tools/data_access.py` -> `ChordFilters`

### 1.4 Dedupe y mezcla de poblaciones

Utilidad:

- `tools/population_utils.py` -> `dedupe_population(df)`

Regla principal:

- Prioriza llave `abs_mask_int`.
- Si existe `__root_midi`, lo agrega a la llave para no colapsar acordes iguales en PC pero en octavas distintas.
- Fallback: `code + interval`.

Esto es crucial para experimentos con control de registro (Exp 3, Exp E).

### 1.5 Transformaciones utiles para validacion

Disponibles:

- Transposicion de acordes: `synth_tools.py` (`transpose_row`, `make_transpositions_df`)
- Inversiones: `synth_tools.py` (`invert_row`, `make_inversions_df`)
- Expansion por escala: `tools/population_builders.py` (`generate_scale_population`)

Precaucion tecnica:

- Las utilidades de `synth_tools.py` para transposicion/inversion trabajan sobre representaciones ancladas en un rango pequeno por defecto (`0..24`) salvo que habilites opciones para permitir fuera de rango. Para poblaciones combinatoriales MIDI reales (48..72, por ejemplo), usa primero el flujo de `services/combinatorial_generator.py` y luego aplica transformaciones con control explicito de rango.

---

## 2. Formalizacion del "puente" paper -> generador

Para cada experimento/articulo, define una ficha con 8 campos:

1. `encoding_origen`: MIDI absoluto, pc-set, etiquetas de tipo de acorde.
2. `rango`: octavas o limites MIDI.
3. `cardinalidades`: lista de `k`.
4. `reglas_pc`: alfabeto de PCs permitido.
5. `reglas_voicing`: span, max intervalo interno, inclusiones/exclusiones.
6. `sampling`: enumeracion total o muestreo.
7. `etiquetas`: clase perceptual/funcional/estilistica.
8. `metrica_objetivo`: Spearman, Mann-Whitney, KL, silhouette, etc.

Luego la traduccion al repo es directa:

- `generate_combinatorial_chords(...)`
- `filter_dataframe(..., ChordFilters(...))`
- `dedupe_population(...)`
- (opcional) `generate_scale_population(...)`

---

## 3. Recetas de poblacion por experimento de validacion

Esta seccion sigue `metodologia_temporal/evaluacion-experimentos.md`.

### 3.1 Exp 1 - Triadas diatonicas baseline

Objetivo:

- Separar mayor/menor/disminuido en embedding.

Poblacion sugerida:

- `alphabet=[0,2,4,5,7,9,11]`
- `octave_min=3`, `octave_max=4` (con nota de borde se cubre C3..C5 aprox si `min(alphabet)=0`)
- `cardinalities=[3]`
- filtro de patrones para triadas canonicas:
  - mayor: `[4,3]`
  - menor: `[3,4]`
  - disminuida: `[3,3]`
- opcional: incluir inversiones con `make_inversions_df`.

### 3.2 Exp 2 - Segregacion de extremos

Objetivo:

- Separar triadas "normales" vs clusters/poliacordes.

Poblaciones:

- `P_triada`: igual a Exp 1.
- `P_extrema`:
  - `alphabet=list(range(12))`
  - `cardinalities=[4,5,6,7,8]`
  - filtros para casos extremos:
    - `max_internal_interval <= 2` para clusters densos.
    - y/o `span_semitones` muy bajo (compactos) o muy alto (abiertos).

Comparacion:

- silhouette por etiqueta (`triada`, `extrema`) y distancias cruzadas.

### 3.3 Exp 3 - Misma estructura en distinto registro (no d->0)

Objetivo corregido:

- Probar vecindad fuerte entre acordes con mismas PC y diferente registro.

Construccion:

1. Toma una base de triadas (Exp 1).
2. Genera versiones desplazadas por octava (+12, -12 cuando sea valido).
3. Dedupe preservando `__root_midi`.
4. Compara:
   - distancias intra-familia (misma estructura, distinto registro)
   - vs distancias entre familias no relacionadas.

### 3.4 Exp 4 y 5 - Escalabilidad N~1k y N~100k

Objetivo:

- Ver estabilidad topologica al crecer N.

Construccion:

- Universo grande:
  - `alphabet=list(range(12))`
  - rango de 2 a 4 octavas segun presupuesto computacional.
  - cardinalidades mixtas `[3,4,5]`.
- `P_1k`: muestreo estratificado por cardinalidad.
- `P_100k`: muestra grande del mismo generador y mismas reglas.
- No cambies heuristicas entre tamanos; cambia solo N.

### 3.5 Exp 6 - Robustez a jitter

Objetivo:

- Probar estabilidad frente a pequenas perturbaciones de frecuencia.

Construccion:

1. Genera poblacion base combinatorial (por ejemplo triadas y tetradas).
2. Para cada acorde, genera replica con jitter en Hz en la etapa de calculo acustico (no alteres la identidad MIDI original para etiquetado).
3. Evalua estabilidad de vecindad (`trustworthiness`, `continuity`) antes/despues.

### 3.6 Exp C - Validacion perceptual (Bowling + Harrison/Pearce)

Objetivo:

- Alinear puntajes del modelo con juicios humanos.

Poblacion:

- `P_humana_real`: acordes exactamente reportados en datasets perceptuales.
- `P_sintetica_contexto`: acordes del mismo rango/cardinalidad para contextualizar el universo.

Regla operativa:

- Si dataset trae MIDI: usar directo.
- Si trae solo PC/tipo de acorde: aplicar una politica de voicing fija y explicita (misma para todo el dataset).

Metricas:

- Spearman entre score humano y score del modelo (rugosidad o distancia).
- Parcial controlando covariables (ej. cardinalidad, notas comunes).

### 3.7 Exp D - Sustituibilidad funcional en estilo Bach

Objetivo:

- Ver si sustitutos funcionales quedan proximos en el espacio.

Poblaciones:

- `P_Bach_real`: verticalidades extraidas del corpus Bach (MIDI o PC->MIDI con politica fija).
- `P_Bach_like_sintetica`: combinatorial restringida por:
  - rango observado en Bach
  - distribucion de cardinalidades
  - filtros de intervalo/voicing compatibles.

Etiquetado funcional:

- Necesitas funciones armonicas/contexto (I, V, ii6, etc.) desde anotacion o analizador.
- Define pares positivos por contexto `(n-1, n+1)` y compara contra control.

### 3.8 Exp E - Consistencia estilistica (Bach vs ruido vs extremos)

Objetivo:

- Separacion de manifolds A/B/C.

Poblaciones:

- A: `P_Bach_real`
- B: `P_ruido_uniforme` (misma cardinalidad/rango que A, pero muestreo uniforme)
- C: `P_extrema` (clusters/poliacordes con reglas anti-estilo)

Condicion critica de diseno:

- Igualar cardinalidad y rango entre A y B para no introducir sesgo trivial.

Metricas:

- KL/cross-entropy de densidades en embedding + silhouette por grupo.

### 3.9 Mapeo detallado de datasets externos a tu generador

#### A) Harrison & Pearce / hrep / incon

Hecho util para tu puente:

- `hrep` implementa representaciones de acorde y voicing.
- `incon` implementa modelos de consonancia sobre esas representaciones.
- El articulo de Harrison & Pearce describe su pipeline como combinacion de interference, harmonicity y cultural familiarity.

Regla practica de reproduccion en ChordSpace:

1. Extrae acordes MIDI/PC desde dataset o paquete.
2. Si tienes solo PC:
   - Aplica una politica fija de voicing (ejemplo: registro medio).
   - Mantiene maximo de notas por acorde segun protocolo del paper.
3. Convierte cada acorde a `notes_abs_json` y genera columnas compatibles (`chroma`, `interval`, `span_semitones`).
4. Mezcla con una poblacion combinatorial de contexto para ubicar cada estimulo en tu universo.

Detalle de fidelidad que puedes heredar:

- En `hrep`, `voice_chord` se define como "voicing nearest around middle C" y soporta `max_notes` (default 5), lo cual te da una regla concreta para anclar PC-sets en registro sin inventar heuristicas arbitrarias.

#### B) hcorp / Bach

Hecho util para tu puente:

- `hcorp` incluye `bach_chorales_1` y `bach_chorales_1b`.
- `bach_chorales_1b` documenta acordes extraidos automaticamente de `bach_chorales_1`, con extraccion cada quaver.

Regla practica de reproduccion en ChordSpace:

1. Toma `bach_chorales_1b` como fuente de verticalidades.
2. Convierte cada sonoridad a MIDI absoluto o a PC + politica de voicing fija.
3. Construye `P_Bach_real` con esas verticalidades.
4. Estima distribuciones de cardinalidad/span/intervalos de `P_Bach_real`.
5. Genera `P_Bach_like_sintetica` con `generate_combinatorial_chords` + filtros que igualen esas distribuciones.

#### C) JSB Chorales dataset

Hecho util para tu puente:

- JSB repo publica variantes de 1st order y 2nd order, y conserva `original_data`.
- Esto te permite elegir el nivel temporal (mas crudo o ya procesado) segun el experimento.

Regla practica:

1. Para Exp D (funcional), usa una resolucion temporal que preserve contexto de vecinos `n-1` y `n+1`.
2. Para Exp E (manifold estilo), puedes usar verticalidades agregadas y deduplicadas por llave musical.

#### D) Durham Chord Dataset (DCD)

Hecho util para tu puente:

- DCD es un dataset combinatorial de acordes 12-TET con 4755 combinaciones de pitch, mas harmonicity/roughness y embedding.
- Sirve como benchmark externo porque su filosofia de "catalogo sistematico" es cercana a tu enfoque.

Regla practica:

1. Importa acordes DCD como poblacion de control externa.
2. Proyecta DCD con tu pipeline (mismas metricas/reductor).
3. Compara ordenamientos y separaciones contra los valores de roughness/harmonicity del propio DCD.

---

## 4. Pipeline operativo recomendado (paso a paso)

### 4.1 Paso 1 - Define contrato del experimento

Antes de generar nada, escribe:

- rango MIDI objetivo
- cardinalidades
- reglas de voicing
- si requiere estimulos exactos o equivalentes
- metrica estadistica

### 4.2 Paso 2 - Genera universo base

Ejemplo:

```python
from services.combinatorial_generator import generate_combinatorial_chords

df = generate_combinatorial_chords(
    alphabet=[0,2,4,5,7,9,11],
    octave_min=3,
    octave_max=4,
    cardinalities=[3,4],
    structural_mode=False,
)
```

### 4.3 Paso 3 - Filtra por reglas del experimento

```python
from services.population_filter import filter_dataframe
from tools.data_access import ChordFilters

f = ChordFilters(
    cardinalities=[3],
    span_min=4,
    span_max=16,
    max_internal_interval=7,
    include_pitch_classes=[0,2,4,5,7,9,11],
    include_pc_mode="subset_of",
    interval_mode="exact",
    interval_patterns=[[4,3],[3,4],[3,3]],
)
df = filter_dataframe(df, f)
```

### 4.4 Paso 4 - Deduplica y preserva metadatos

```python
from tools.population_utils import dedupe_population

df, dedupe_mode = dedupe_population(df)
```

### 4.5 Paso 5 - Construye poblaciones A/B/C del experimento

- A: poblacion "real" (humanos o Bach)
- B: control uniforme
- C: extremos

Todas con mismas columnas compatibles con pipeline (`notes_abs_json`, `chroma`, `interval`, `span_semitones`, etc.).

### 4.6 Paso 6 - Corre vectorizacion, distancias, embedding y test estadistico

Manteniendo la misma configuracion entre poblaciones para comparabilidad.

---

## 5. Donde la heuristica SI alcanza y donde NO alcanza

### 5.1 Si alcanza

- Cuando el paper define clases de acordes y restricciones generales.
- Cuando quieres validacion de ordenamientos relativos (Spearman, separacion de clusters).
- Cuando el objetivo es comparar regiones del espacio generado por tus reglas.

### 5.2 No alcanza por si sola

- Cuando el paper exige replicas exactas de estimulos originales y timing.
- Cuando necesitas etiquetas funcionales humanas (analisis armonico anotado).
- Cuando hay variables fuera de tu generador simbolico (timbre especifico, dinamica real, contexto temporal detallado).

Conclusion metodologica:

- Tu enfoque "genero todo por combinatoria y aplico heuristicas del paper" es correcto para validacion estructural/perceptual de primer nivel.
- Para validacion estrictamente replicativa, debes inyectar estimulos/corpus originales como subconjuntos dentro del mismo espacio combinatorial.

---

## 6. Checklist reproducible para cada experimento

- Guardar parametros de generacion (`alphabet`, octavas, cardinalidades, `structural_mode`).
- Guardar reglas de filtro (`ChordFilters`) y orden de aplicacion.
- Guardar estrategia de dedupe y llaves usadas.
- Guardar criterio de muestreo (enumeracion total vs sample).
- Guardar version de codigo y fecha.
- Guardar mapeo de datos externos a `notes_abs_json`.
- Guardar estadistica objetivo y prueba usada.

---

## 7. Fuentes

### 7.1 Fuentes internas del repositorio

- `metodologia_temporal/evaluacion-experimentos.md`
- `metodologia_temporal/ESTRUCTURA_MATEMATICA_DETALLADA.md`
- `metodologia_temporal/modelo_computacional_de_generacion_y_tratamiento_de_acordes_v_1.md`
- `docs/FLUJO_DATOS_GUI.md`
- `services/combinatorial_generator.py`
- `services/population_filter.py`
- `tools/data_access.py`
- `tools/population_utils.py`
- `tools/population_builders.py`
- `synth_tools.py`

### 7.2 Fuentes externas (articulos, paquetes y corpus)

- Harrison, P. M. C., & Pearce, M. T. (2020). Simultaneous consonance in music perception and composition. *Psychological Review*.  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7006947/

- Bowling, D. L., Purves, D., & Gill, K. Z. (2018). Vocal similarity predicts the relative attraction of musical chords. *PNAS*.  
  https://pubmed.ncbi.nlm.nih.gov/30030333/

- Paquete `hrep` (representacion de acordes y voicing de sonoridades).  
  https://github.com/pmcharrison/hrep

- Paquete `incon` (modelado de consonancia).  
  https://github.com/pmcharrison/incon

- Paquete `hcorp` (corpora, incluye `bach_chorales_1` y `bach_chorales_1b`).  
  https://github.com/pmcharrison/hcorp

- Documentacion `bach_chorales_1b` en hcorp.  
  https://rdrr.io/github/pmcharrison/hcorp/man/bach_chorales_1b.html

- JSB Chorales dataset (versiones 1st/2nd order y datos originales).  
  https://github.com/czhuang/JSB-Chorales-dataset

- Durham Chord Dataset (DCD), dataset combinatorial de acordes 12-TET para consonancia/representacion.  
  https://pmcharrison.github.io/durham-chord-dataset/
