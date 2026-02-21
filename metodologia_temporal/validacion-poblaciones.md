# Validación experimental de ChordSpace mediante poblaciones combinatoriales

## 1. Introducción y objetivo

El modelo ChordSpace define un espacio geométrico de acordes basado en una representación formal de **pitch chords** (tuplas de números MIDI estrictamente crecientes) y en un conjunto de funciones que calculan propiedades psicoacústicas (por ejemplo, rugosidad) y reducciones de dimensión a 2D. El objetivo de este documento es justificar, con rigor metodológico y referencias, cómo usamos **poblaciones de acordes generadas combinatorialmente** para:

1. Replicar el dominio de estímulos de Bowling et al. (2018) y validar la relación entre geometría y consonancia percibida.
2. Construir poblaciones “barrocas” y de “ruido” compatibles con un corpus de corales de Bach, para validar la separación estilística y la sustituibilidad funcional.

El énfasis está en explicar **por qué** este enfoque es válido y cómo se implementa de manera reproducible dentro del repositorio de ChordSpace.

---

## 2. Universo formal de acordes en ChordSpace

### 2.1 Definición de pitch chord y universo \(U(A,\text{oct\_min},\text{oct\_max},k)\)

En ChordSpace, un acorde se representa como un **pitch chord**:

\[
\mathbf{n} = (n_1,\dots,n_k) \in \mathbb{Z}^k, \quad n_1 < \dots < n_k,
\]

donde cada \(n_i\) es un número de nota MIDI entero, sin notas duplicadas dentro del mismo acorde (no se permiten unísonos internos). La identidad del acorde incluye tanto las **clases de altura** (pitch classes) como el **registro absoluto**; es decir, ChordSpace no colapsa octavas ni aplica invariancia por transposición \cite{harrison2020simultaneous}.

Dado un conjunto de clases de altura \(A \subseteq \{0,\dots,11\}\), un rango de octavas \([\text{oct\_min},\text{oct\_max}]\) y una cardinalidad \(k\), definimos el universo combinatorial de acordes:

\[
U(A,\text{oct\_min},\text{oct\_max},k) = \big\{\mathbf{n} : n_i \in \mathbb{Z},\; \pi(n_i) \in A,\; \text{oct\_min} \leq \text{oct}(n_i) \leq \text{oct\_max},\; n_1 < \dots < n_k\big\},
\]

donde \(\pi\) proyecta a clase de altura módulo 12 y \(\text{oct}(n)\) es la octava de \(n\). Para un conjunto de cardinalidades \(K\), definimos

\[
U(A,\text{oct\_min},\text{oct\_max},K) = \bigcup_{k \in K} U(A,\text{oct\_min},\text{oct\_max},k).
\]

Este universo es precisamente lo que genera el servicio `generate_combinatorial_chords` en el repositorio, bajo distintas configuraciones de `alphabet`, `octave_min`, `octave_max` y `cardinalities`.

### 2.2 Generación y filtrado en el repositorio

El flujo de generación implementado en el código es:

- `generate_combinatorial_chords(alphabet, octave_min, octave_max, cardinalities, structural_mode)`:
  - Construye el universo de notas MIDI permitido a partir de `alphabet` (clases de altura 0–11) y el rango de octavas.
  - Recorre todas las combinaciones sin repetición posibles (`itertools.combinations`).
  - Para cada acorde, calcula múltiples vistas (notas absolutas, clases de altura, intervalos internos, span, etc.).
- `filter_dataframe(df, ChordFilters)`:
  - Aplica filtros sobre cardinalidad, span en semitonos, intervalos internos, pertenencia a subconjuntos de clases de altura, etc.

De este modo, cualquier subconjunto finito del universo \(U(A,\text{oct\_min},\text{oct\_max},K)\) que pueda definirse mediante restricciones sobre clases de altura, registro y cardinalidad puede **generarse explícitamente** en el repositorio.

---

## 3. Validación perceptual: población Bowling (2018)

### 3.1 Dominios de acordes en Bowling et al. (2018)

Bowling et al. (2018) estudian la atracción relativa de acordes en función de su similitud espectral con la voz humana \cite{bowling2018vocal}. Para ello, recogen valoraciones de consonancia para **todas las díadas, tríadas y tétradas cromáticas dentro de una octava**:

> “Ratings of consonance for every possible chromatic dyad, triad, and tetrad within a single octave were obtained from 30 subjects...” \cite{bowling2018vocal}.

En la sección de métodos, especifican además el número de acordes por tipo (12 díadas, 66 tríadas, 220 tétradas) y documentan que los tonos se extraen de la escala cromática en una sola octava, sintetizados como tonos complejos con estructura espectral inspirada en la voz \cite{bowling2018vocal}.

Desde el punto de vista **simbólico**, el dominio de estímulos de Bowling es por tanto:

- Alfabeto de clases de altura: \(A = \{0,\dots,11\}\) (escala cromática completa).
- Una sola octava (12 posiciones discretas).
- Cardinalidades \(k \in \{2,3,4\}\) (díadas, tríadas, tétradas).

La diferencia entre los conteos combinatoriales estándar (por ejemplo, \(\binom{12}{4} = 495\) tétradas distintas de cuatro clases de altura) y los 220 acordes reportados se explica por el hecho de que Bowling construyen acordes como **intervalos sobre un bajo fijo** (conjuntos de 1, 2 o 3 intervalos tomados de los 12 posibles), en lugar de elegir arbitrariamente subconjuntos de cuatro alturas sin referencia a un bajo \cite{bowling2018vocal}.

### 3.2 Traducción al universo de ChordSpace

En ChordSpace, replicamos el dominio de Bowling fijando:

- **Alfabeto**: \(A = \{0,\dots,11\}\).
- **Rango de octavas**: una sola octava de referencia, por ejemplo \(\text{oct\_min} = \text{oct\_max} = 4\), correspondiente aproximadamente a C4–B4 (MIDI 60–71). Esta elección es una **convención metodológica**: el artículo fija una octava cromática pero no depende del número MIDI exacto.
- **Cardinalidades**: \(k \in \{2,3,4\}\).

Con estos parámetros, el universo

\[
U(A,4,4,\{2,3,4\})
\]

contiene simbólicamente todas las combinaciones de 2–4 notas cromáticas dentro de una octava. Dentro de este universo se encuentra, como subconjunto, el conjunto más restringido de acordes que Bowling define combinando intervalos sobre un bajo fijo. Esto significa que:

- Podemos **importar directamente** los estímulos exactos de Bowling (cuando estén disponibles en materiales suplementarios) como pitch chords en este universo.
- O bien podemos generar, mediante reglas combinatoriales, un espacio *igual o más amplio* que el de Bowling, en el que su conjunto de estímulos se embebe de forma natural.

### 3.3 Justificación de validez

Desde la perspectiva de la tesis, lo que se busca no es replicar cada detalle acústico (timbre exacto, afinación fina), sino verificar si la **geometría de distancias** inducida por el modelo (rugosidad, vector de intervalos, etc.) respeta ordenamientos de consonancia similares a los observados en humanos. Para ello basta con que el **dominio de acordes** sea compatible:

- Trabajamos en 12-TET con alturas discretas, como hace el análisis cromático de Bowling \cite{bowling2018vocal}.
- Usamos las mismas clases de acordes (díadas, tríadas, tétradas) dentro de una octava.
- Las pequeñas diferencias de registro absoluto (elegir C4–B4 en lugar de otra octava) no alteran la estructura combinatorial del espacio ni la interpretación de los resultados.

Por tanto, es metodológicamente válido usar el universo \(U(A,4,4,\{2,3,4\})\) como dominio de validación perceptual, siempre que se documenten claramente:

- Qué subconjunto corresponde a los estímulos exactos de Bowling (si se usan).
- Qué parte del espacio adicional se explora mediante poblaciones sintéticas.

---

## 4. Validación barroca y estilística: poblaciones tipo Bach

### 4.1 Corpus de corales de Bach y representación simbólica

Para validar la relevancia estilística de ChordSpace, nos apoyamos en corpora de corales de Bach como los ofrecidos en paquetes como `hcorp` (por ejemplo, `bach_chorales_1b`) o en datasets JSB estandarizados \cite{hcorp,jsb2017}. Estos corpora representan cada coral como una sucesión de **sonoridades verticales** (acordes) a lo largo del tiempo, típicamente en un rango SATB (soprano, alto, tenor, bajo).

Cada verticalidad puede convertirse en un pitch chord de ChordSpace:\
\(\mathbf{n} = (n_1<\dots<n_k)\), donde los \(n_i\) son números MIDI de las voces activas en ese pulso. Esta conversión se hace preservando el registro absoluto, de modo que la consonancia sensorial calculada por nuestro modelo sigue siendo sensible a la altura real del acorde, tal como sugiere la literatura psicoacústica \cite{harrison2020simultaneous}.

### 4.2 Definición de poblaciones barrocas

Proponemos tres tipos de poblaciones para los experimentos de validación barroca:

1. **Población real barroca (P\_BACH\_REAL):**
   - Conjunto de todos los acordes verticales extraídos de un corpus de corales de Bach (por ejemplo, `bach_chorales_1b`).
   - Cada acorde se representa como pitch chord en un rango aproximado C3–C6 (MIDI 48–84).
   - Cardinalidades típicas \(k \in \{3,4\}\), correspondientes a tres o cuatro voces activas.

2. **Superpoblación sintética barroca (P\_BACH\_SYN):**
   - Universo combinatorial \(U(A,\text{oct\_min},\text{oct\_max},K)\) con:
     - \(A = \{0,\dots,11\}\) (todas las clases de altura).
     - \(\text{oct\_min} = 3\), \(\text{oct\_max} = 5\) (aprox. C3–C5), rango coherente con la tesitura de los corales.\
     - \(K = \{3,4\}\).
   - Sobre este universo, se aplican filtros (`ChordFilters`) para aproximar las estadísticas de \(P_\text{BACH\_REAL}\):
     - Distribución de cardinalidades (proporción de tríadas vs. tétradas).
     - Span máximo en semitonos (por ejemplo, acotar a 24 semitonos para restringir a dos octavas, como en un coral típico).
     - Restricciones suaves sobre intervalos internos (evitar clusters cromáticos extremadamente densos que rara vez aparecen en Bach).

3. **Poblaciones de control (P\_NOISE, P\_EXTREMES):**
   - **P\_NOISE:** acordes generados aleatoriamente en el mismo universo \(U(A,\text{oct\_min},\text{oct\_max},K)\), con las mismas cardinalidades y rango que \(P_\text{BACH\_REAL}\), pero sin intentar imitar estadísticas internas. Sirve como “ruido uniforme” para contrastar la estructura estilística.\
   - **P\_EXTREMES:** acordes “extremos” (clusters cromáticos, poliacordes y estructuras muy disonantes) generados mediante filtros que fuerzan spans muy pequeños (clusters) o muy grandes y acumulación de segundas menores. Estos acordes funcionan como referencia de alta complejidad/disonancia, análoga a los acordes extremos del Exp 2.

### 4.3 Justificación de validez estilística

El uso de \(P_\text{BACH\_REAL}\) y \(P_\text{BACH\_SYN}\) se justifica así:

- \(P_\text{BACH\_REAL}\) contiene **los acordes efectivamente empleados** en un corpus canónico barroco, por lo que cualquier propiedad geométrica o psicoacústica observada en esta población es directamente relevante para la música de Bach \cite{hcorp}.
- \(P_\text{BACH\_SYN}\) extiende este conjunto a todas las configuraciones de acorde **combinatorialmente posibles** que respetan las mismas restricciones globales (rango, cardinalidad, ciertas limitaciones interválicas). Esto permite preguntar si las propiedades observadas en \(P_\text{BACH\_REAL}\) son idiosincrásicas de los corales o reflejan propiedades más generales del espacio de acordes “al estilo Bach”.
- Las poblaciones de control \(P_\text{NOISE}\) y \(P_\text{EXTREMES}\) permiten contrastar si el embedding 2D distingue claramente entre el “manifold barroco” y regiones de ruido o complejidad extrema, lo cual es coherente con enfoques recientes de análisis estilístico y separación de manifolds armónicos \cite{harrison2020simultaneous}.

En este contexto, el generador combinatorial no pretende “inventar” acordes de Bach, sino construir un **supraespacio** estructuralmente compatible en el que los acordes reales del corpus se embeben junto con acordes hipotéticos del mismo tipo.

---

## 5. Validez epistemológica de la aproximación

La validez de esta estrategia descansa en tres pilares:

1. **Compatibilidad formal de dominios:**
   - Bowling et al. trabajan con acordes cromáticos discretos en una octava; este dominio es un subconjunto bien definido de \(U(A,\text{oct\_min},\text{oct\_max},k)\) cuando \(A = \{0,\dots,11\}\) y \(\text{oct\_min} = \text{oct\_max}\) \cite{bowling2018vocal}.
   - Los datasets reanalizados por Harrison & Pearce y los corpora barrocos trabajan con acordes simultáneos en 12-TET, que son igualmente representables como pitch chords en el universo de ChordSpace \cite{harrison2020simultaneous,hcorp}.

2. **Claridad en lo que es “estímulo exacto” vs. “población sintética”:**
   - Cuando se dispone de las listas de acordes exactos (Bowling, corpora de Bach), se utilizan directamente como subconjuntos finitos de \(U\).
   - Cuando solo se especifica una **clase de estímulos** (por ejemplo, “todas las dyads/triads/tetrads cromáticas en una octava”), se define un universo combinatorial equivalente y se documenta la traducción con parámetros de generación y filtros.

3. **Naturaleza de las preguntas de validación:**
   - Las métricas que se utilizan (correlación de Spearman entre rugosidad y ratings de consonancia, trustworthiness, continuity, silhouette, separación de manifolds) dependen de ordenamientos y estructuras globales más que de detalles microacústicos (timbre exacto, pequeñas diferencias de afinación). Esto hace razonable trabajar con representaciones simbólicas en 12-TET y timbres sintéticos estándar, como recomiendan trabajos recientes de modelado de consonancia simultánea \cite{harrison2020simultaneous}.

Siempre que estas decisiones se documenten explícitamente en la tesis (parámetros de \(U\), filtros aplicados, diferencias con respecto a los estudios originales), el lector puede evaluar con claridad el alcance y las limitaciones de los experimentos de validación.

---

## 6. Prompt sugerido para el asistente del repositorio

A continuación se propone un prompt reutilizable para el asistente del repositorio (por ejemplo, un modelo con acceso al código) que automatiza la generación de las poblaciones descritas.

```text
Vas a ayudarme a generar poblaciones de acordes en el repositorio ChordSpace usando SOLO las funciones ya existentes.

Contexto de alto nivel:
- Un acorde es un pitch chord: tupla estrictamente creciente de notas MIDI sin duplicados.
- El universo combinatorial se genera con generate_combinatorial_chords(alphabet, octave_min, octave_max, cardinalities, structural_mode).
- El filtrado se hace con filter_dataframe(df, ChordFilters(...)).

Tareas:
1) Te daré una descripción breve de un experimento externo (por ejemplo, “Bowling 2018: todas las díadas, tríadas y tétradas cromáticas en una octava” o “corales de Bach en rango SATB, cardinalidades 3–4”).

2) Con esa descripción, debes devolverme:
   a) Una especificación explícita de parámetros para llamar a generate_combinatorial_chords:
      - alphabet (lista de PCs 0–11),
      - octave_min y octave_max (enteros),
      - cardinalities (lista de k),
      - structural_mode (True/False).
   b) Una configuración concreta de ChordFilters que imite las restricciones adicionales:
      - span_min/span_max,
      - max_internal_interval,
      - include/exclude pitch classes,
      - patrones de intervalo si aplica.
   c) Un bloque de código Python que:
      - llama a generate_combinatorial_chords con esos parámetros,
      - aplica filter_dataframe,
      - deja el resultado en un DataFrame llamado P_<NOMBRE_EXPERIMENTO>.

3) No inventes funciones nuevas ni cambies las firmas.
   Usa solo:
   - services/combinatorial_generator.generate_combinatorial_chords
   - services.population_filter.filter_dataframe
   - tools.data_access.ChordFilters
   - y utilidades ya presentes en el repo.

4) Documenta en comentarios del código cualquier decisión metodológica (por ejemplo, elección de octava de referencia o suavización de filtros), distinguiendo claramente entre:
   - condiciones que vienen del paper/corpus,
   - y decisiones propias de este proyecto.

Ejemplo de descripción que te daré:
- EXPERIMENTO: Bowling2018
  DOMINIO: todas las díadas, tríadas y tétradas cromáticas dentro de una octava.
  OBJETIVO: crear P_BOWLING con todas las combinaciones relevantes según el dominio.

Cuando recibas una descripción así, responde SOLO con:
- comentarios breves explicando tu criterio,
- el dict de parámetros para generate_combinatorial_chords,
- la construcción de ChordFilters,
- y el bloque de código para construir el DataFrame.
```

Este prompt garantiza que el asistente del repositorio trabaje siempre dentro de los límites de las funciones existentes y de las definiciones formales del universo \(U(A,\text{oct\_min},\text{oct\_max},K)\), manteniendo la trazabilidad entre dominios experimentales externos y poblaciones generadas en ChordSpace.
