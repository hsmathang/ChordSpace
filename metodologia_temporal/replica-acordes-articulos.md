# Nuestro sistema replica acordes de otros artículos para la experimentación

## 1. Universo combinatorial de acordes en ChordSpace

En nuestro modelo, todo acorde se representa como un **pitch chord**: una tupla estrictamente creciente de números MIDI \((n_1 < \dots < n_k)\) sin notas duplicadas (sin unísonos), definida sobre un rango de octavas fijo y un alfabeto de 12 clases de altura (0–11). Este universo se puede formalizar como un conjunto \(U(A, \text{oct\_min}, \text{oct\_max}, k)\), donde \(A \subseteq \{0,\dots,11\}\) es el alfabeto de pitch classes y \(k\) es la cardinalidad del acorde.

Todo acorde 12-TET con alturas discretas, sin microtonos y dentro de ese rango de octavas es, por definición, un elemento de algún \(U(A, \text{oct\_min}, \text{oct\_max}, k)\). Por eso, cualquier estímulo usado en experimentos que trabajen en 12-TET (Harrison & Pearce, Bowling, JSB, etc.) puede verse como un subconjunto de este universo combinatorial, una vez se fijan alfabeto, rango y cardinalidad.\cite{harrison2020simultaneous,bowling2018vocal,hcorp,hrep}

---

## 2. Cuándo coinciden exactamente las poblaciones de acordes

Decimos que las poblaciones “coinciden” en sentido fuerte cuando el artículo o corpus:

1. Trabaja en **temperamento igual de 12 notas** (12-TET), sin microtonos.
2. Especifica **rango de registro** (por ejemplo, un rango vocal SATB o un intervalo MIDI explícito).
3. Fija **cardinalidad** (número de notas por acorde) o un intervalo pequeño de cardinalidades.
4. No introduce restricciones adicionales que contradigan nuestra definición (por ejemplo, afinación no igual, glissandi continuos, etc.).

Si ajustamos nuestro generador combinatorial con el mismo alfabeto \(A\), el mismo rango de octavas y las mismas cardinalidades, el conjunto de acordes “permitidos” por el experimento es exactamente un subconjunto de nuestro \(U(A, \text{oct\_min}, \text{oct\_max}, k)\). En estos casos, los estímulos de los artículos son literalmente instancias de nuestro espacio y pueden tratarse como tales sin introducir ninguna heurística adicional.\cite{harrison2020simultaneous,hcorp,jsb2017}

Ejemplos típicos de esta situación son:

- Acordes extraídos de corales de Bach en formato MIDI (corpora JSB, `bach_chorales_1b` en `hcorp`).\cite{hcorp,jsb2017}
- Voicings producidos por `hrep` como `voice_chord` en un rango vocal alrededor de do central.\cite{hrep}
- Catálogos combinatoriales como el Durham Chord Dataset, construidos enumerando sistemáticamente acordes 12-TET bajo ciertas reglas.\cite{durhamChordDataset}

---

## 3. Cuándo las poblaciones son solo compatibles pero no idénticas

Hay casos en los que nuestro universo de acordes es **compatible** con el de los artículos, pero las poblaciones que generamos por heurística no replican uno a uno los estímulos originales:

1. **Doblajes de voces (unísonos):**
   En corales de Bach o en música coral real, varias voces pueden cantar la misma nota al unísono. Nuestro modelo, sin embargo, representa acordes como conjuntos sin repetición; dos voicings que difieren solo por un doblaje se colapsan al mismo pitch chord. Esto es aceptable cuando el foco está en la disonancia sensorial vertical (rugosidad), donde el efecto de los doblajes es menor que el de cambiar la estructura de alturas.\cite{harrison2020simultaneous}

2. **Muestreo vs. enumeración total:**
   Algunos estudios toman una muestra finita de acordes bajo ciertas reglas (por ejemplo, 40–60 acordes elegidos aleatoriamente dentro de una clase de estímulos). Nuestro generador puede enumerar todo el espacio posible o muestrear de nuevo con la misma distribución de cardinalidad y rango, pero las muestras concretas no coinciden exactamente salvo que carguemos los datos originales.\cite{durhamChordDataset}

3. **Detalles de timbre y dinámica:**
   Experimentos como los de Bowling et al. modelan con detalle el espectro vocal y timbres concretos. Nuestro pipeline usa un timbre sintético fijo para el cálculo de rugosidad (modelo de Sethares), lo cual mantiene la estructura relativa de disonancia pero no reproduce todas las características acústicas finas de los estímulos originales.\cite{bowling2018vocal}

En estos contextos, lo que ofrecemos son **poblaciones sintéticas del mismo tipo** que los estímulos experimentales: respetan las mismas reglas estructurales (tipo de acorde, rango, cardinalidad), pero no pretenden ser copias exactas de cada estímulo individual.

---

## 4. Uso de heurísticas basadas en los artículos

Para la mayoría de las validaciones que nos interesan (ordenamientos relativos de consonancia, separación de estilos, preservación de la geometría), es suficiente con reproducir cuidadosamente las **restricciones estructurales** descritas en cada artículo o corpus y generar acordes sintéticos que las cumplan:

- Mismo tipo de triadas o tetradas (mayor, menor, disminuida, etc.).
- Mismos rangos aproximados de registro (por ejemplo, SATB o un intervalo MIDI concreto).
- Mismas cardinalidades y spans típicos.

En este escenario, las heurísticas “del artículo” se implementan como parámetros del generador combinatorial (alfabeto, octavas, cardinalidad) y como filtros de población (span máximo, máximo intervalo interno, inclusión o exclusión de ciertos patrones interválicos).\cite{hrep, hcorp, durhamChordDataset}

Este diseño es metodológicamente sólido cuando:

- Las métricas de validación se basan en **ordenamientos** (correlación de Spearman entre rugosidad y juicios humanos, por ejemplo).
- Se comparan **estructuras globales** (silhouette score por clase de acorde, separación de manifolds Bach vs. ruido, métricas de trustworthiness/continuity en embeddings).\cite{harrison2020simultaneous, durhamChordDataset}

---

## 5. Casos en los que hay que usar los estímulos originales

Las heurísticas combinatoriales no bastan cuando el objetivo es una **replicación estricta** de resultados publicados. En particular:

- Si queremos demostrar que nuestro modelo reproduce **exactamente** los coeficientes de correlación o las curvas reportadas por Harrison & Pearce en su conjunto de estímulos, necesitamos usar esos mismos acordes y no un muestreo nuevo.\cite{harrison2020simultaneous}
- Si el efecto que se estudia depende de la **secuencia temporal exacta** (orden de acordes, contexto a largo plazo, timing, dinámica), entonces debemos trabajar directamente con el corpus anotado (por ejemplo, corales de Bach con análisis funcional) y tratar cada verticalidad original como un pitch chord dentro de nuestro universo, sin reemplazarla por acordes sintéticos generados al azar.\cite{hcorp,jsb2017}

En estos casos, el papel del generador combinatorial es complementario: sirve para explorar el espacio completo de acordes compatibles con el estilo o con las reglas experimentales, pero la comparación numérica principal debe hacerse sobre el subconjunto de estímulos originales.

---

## 6. Conclusión metodológica

Nuestro sistema de generación combinatorial es lo bastante general como para **contener** como casos particulares las poblaciones de acordes usadas en gran parte de la literatura reciente sobre consonancia simultánea y análisis armónico (Harrison & Pearce, Bowling, corpus de Bach, catálogos combinatoriales tipo Durham).\cite{harrison2020simultaneous,bowling2018vocal,hcorp,jsb2017,durhamChordDataset}

La estrategia correcta es siempre distinguir, de forma transparente, entre:

- Experimentos donde los acordes de validación son **estímulos exactos** importados de otros trabajos (subconjuntos explícitos de \(U\)).
- Experimentos donde usamos **poblaciones sintéticas del mismo tipo**, generadas por heurísticas que respetan las restricciones estructurales de los artículos.

Documentando claramente esa diferencia y las decisiones de rango, cardinalidad y voicing, podemos justificar con rigor que nuestro espacio de acordes replica, en el sentido correcto, los dominios armónicos de los trabajos de referencia.
