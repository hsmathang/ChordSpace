# Evaluación general de los experimentos propuestos

En conjunto, la batería de experimentos es coherente con tu modelo y cubre bien tres cosas: (1) que el mapa 2D respeta la geometría de distancias, (2) que la geometría tiene sentido psicoacústico, y (3) que no es un artefacto trivial respecto al estilo barroco. Los experimentos A, B, C y E están muy bien planteados; el que más ajustes conceptuales necesita es el Exp 3 (invariancia de octava) y, en menor medida, concretar mejor cómo se medirá el Exp D.

---

## 1. Contexto rápido del modelo

Tu modelo toma acordes discretizados en MIDI, calcula una medida de disonancia/rugosidad basada en Sethares y otras características (vector de clases de intervalo, etc.), y luego los embebe en 2D con un método de reducción de dimensión (MDS, UMAP, etc.). La calidad del mapa se evalúa con métricas topológicas (trustworthiness, continuity, stress, etc.) y métricas musicales/perceptuales.

En este contexto, “validar el modelo” significa al menos tres cosas:

- Topológicamente: el mapa 2D **no deforma demasiado** las distancias del espacio original (trustworthiness, continuity, Shepard, stress).
- Psicoacústicamente: la geometría está alineada con datos humanos de consonancia o similitud de intervalos.
- Estilísticamente: acordes de Bach, ruido y acordes “extremos” ocupan regiones diferenciadas, y acordes funcionalmente equivalentes quedan cerca.

---

## 2. Serie A (Exp 1–6): validación musical incremental

### Exp 1 – Tríadas diatónicas como línea base

**Qué hace:**  
- Usas 21 tríadas diatónicas en Do mayor (incluyendo inversiones) como conjunto muy controlado.  
- Quieres ver si el mapa 2D diferencia claramente acordes **mayores, menores y disminuidos** sin haberle dicho al modelo qué etiqueta tiene cada acorde.

**Qué prueba:**

- Si la medida de disonancia + vector de intervalos contienen información suficiente para que acordes mayor, menor y disminuido se separen en el mapa.  
- Estadísticamente, puedes medirlo con algo simple y muy interpretable:
  - Entrenar un clasificador muy sencillo (p. ej. k-NN) sobre las coordenadas 2D y ver si acierta la “cualidad” del acorde (mayor/menor/disminuido) con alta exactitud.  
  - Calcular **Silhouette score** por clase de acorde: mide qué tan cerca está cada acorde de los de su grupo y qué tan lejos de otros grupos (valores cercanos a 1 son buenos).

**Valoración:**  
Es un excelente “smoke test”: con pocos acordes, controlados, y una hipótesis clara (“el mapa debe separar cualidades armónicas básicas”).

---

### Exp 2 – Segregación de acordes extremos

**Qué hace:**  
- Añades a la colección anterior acordes muy densos: clusters cromáticos, poliacordes, etc.  
- Hipótesis: estos acordes de “alta complejidad/disonancia” deben caer en una región separada del manifold, no mezclados con las tríadas.

**Qué prueba (musicalmente):**

- Que la **rugosidad sensorial y la estructura de intervalos** generan una especie de “eje de complejidad armónica” en el espacio.  
- Que el mapa no sufre el típico **crowding problem**: si mezcla clusters y tríadas en el mismo vecindario, entonces no está capturando bien la complejidad/disonancia.

**Cómo medirlo de forma sencilla:**

- Comparar la distribución de distancias internas:
  - Distancias entre tríadas vs. distancias entre clusters vs. distancias cruzadas (tríada–cluster).  
- Medir Silhouette score usando la etiqueta “tipo de acorde” (triada vs. extremo). Un buen mapa debe dar silhouette alto para esa partición.

**Valoración:**  
Muy buen experimento para ver si la complejidad/disonancia es una dimensión latente del espacio.

---

### Exp 3 – “Invariancia de octava” (tensión con tu modelo)

**Definición del experimento en tu texto:**  
- Datos: tríadas con duplicaciones de octava, como C3–E3–G3 vs C4–E4–G4.  
- Hipótesis: la distancia entre esas versiones debe tender a cero (d → 0).  
- Razón musical: comprobar que el sistema “aprende clases de altura” (pitch classes) y no solo frecuencias absolutas.

**Problema conceptual:**

1. En tu capítulo 3 defines el objeto básico como **pitch chord**, no como pc-set: mantienes el registro absoluto y **rechazas explícitamente** la invariancia por octava y por transposición.  
2. Desde la psicoacústica, se sabe que la consonancia sensorial **sí depende del registro**: Eerola y Lahdelma muestran que la consonancia varía de forma cúbica con la altura del acorde, con óptimo alrededor de C4–C5 y caída en graves y agudos.

Eso significa que:

- No es coherente esperar d ≈ 0 entre C3–E3–G3 y C4–E4–G4 si tu modelo se basa en Sethares y en pitch chords: la rugosidad cambia con el registro.  
- Lo que sí es razonable esperar es **“distancias pequeñas pero no cero”**: acordes con la misma estructura intervalar deberían ser vecinos cercanos, pero no idénticos.

**Cómo reformularlo:**

- Cambiar la hipótesis a algo como:  
  “Las versiones en diferentes octavas de la misma tríada quedan **más cerca entre sí** que acordes de distinta cualidad o distinta estructura intervalar.”  
- Estadísticamente:  
  - Comparar la distribución de distancias entre “mismas pitch classes, distinto registro” con la distribución de distancias entre acordes no relacionados.  
  - Un simple test de rangos (p. ej. Mann–Whitney) puede mostrar que las primeras distancias son significativamente menores.

**Conclusión:**  
La idea de probar “invariancia de octava” es buena, pero la hipótesis d → 0 choca con las decisiones de identidad de tu propio modelo y con la literatura psicoacústica. Mejor hablar de **vecindad fuerte** (distancia pequeña) en lugar de identidad geométrica.

---

### Exp 4 y 5 – Escalabilidad (N ≈ 1000 vs N = 100k)

**Qué hacen:**  
- Muestras de tamaño medio (~1000 acordes) y muy grande (~100k).  
- Hipótesis: estructuras globales (círculo de quintas, agrupaciones tonales) no se destruyen al aumentar N.

**Qué prueba (estadísticamente):**

- Cómo se comportan **trustworthiness T(k)** y **continuity C(k)** al crecer N:  
  - T(k): qué proporción de los vecinos cercanos en el mapa 2D **eran realmente vecinos** en el espacio original.  
  - C(k): qué proporción de los vecinos cercanos en el espacio original siguen siéndolo en el mapa (que no “rompas” vecindarios).

**Diseño razonable:**

- Fijar un conjunto de parámetros del reductor (p. ej. UMAP con un k de vecinos fijo) y comparar los promedios de T(k) y C(k) entre N=1000 y N=100k.  
- Complementar con diagramas de Shepard (dispersión entre distancias originales y embebidas) usando un muestreo de pares para no explotar la complejidad cuadrática.

**Valoración:**  
Es un buen test de “big data musical”: pregunta si tu mapa sigue siendo interpretable cuando llenas densamente el espacio de acordes.

---

### Exp 6 – Resistencia al ruido (jitter)

**Qué hace:**  
- Tomas acordes y les aplicas pequeñas variaciones de frecuencia (“jitter”), que conceptualmente corresponden a ligeros desafinamientos o variaciones microtonales.  
- Hipótesis: la estructura topológica no debe cambiar drásticamente; T(k) debería permanecer alta o al menos relativamente estable.

**Qué prueba:**

- Estabilidad del modelo frente a pequeñas perturbaciones del input, algo esencial si luego quieres aplicarlo a datos reales con ruido de interpretación.  
- En términos estadísticos, puedes comparar:  
  - T(k) y C(k) antes y después de aplicar jitter.  
  - La distribución de distancias entre cada acorde original y su versión con jitter.

**Valoración:**  
Muy buena idea; refuerza que tus embeddings no son frágiles ni dependen de un redondeo excesivamente idealizado.

---

## 3. Experimento B – Validación topológica rigurosa

**Qué haces:**  
- Usas **trustworthiness** y **continuity** para contrastar la hipótesis nula “el mapa es básicamente aleatorio”.  
- Hipótesis nula: T(k) y C(k) no difieren significativamente de una proyección aleatoria (valores alrededor de 0.5 para k moderado).

**Conceptos estadísticos en sencillo:**

- Imagina que cada acorde tiene una lista de “vecinos más cercanos” en el espacio original y otra lista en el mapa 2D.  
- **Trustworthiness T(k)**:  
  - Mira los k vecinos más cercanos en el mapa.  
  - Penaliza los puntos que aparecen como vecinos en el mapa pero **no lo eran** en el espacio original (“falsos amigos”).  
  - Un valor cercano a 1 significa que el mapa casi no inventa vecinos falsos.  
- **Continuity C(k)**:  
  - Mira los k vecinos más cercanos en el espacio original.  
  - Penaliza los que dejan de ser vecinos en el mapa (“roturas”: amigos de verdad que ya no lo parecen).

Venna y Kaski muestran precisamente este trade-off y proponen local MDS como método para controlar el compromiso entre T y C.

**Cómo hacerlo verdaderamente “rigurosamente” estadístico:**

- Generar muchas proyecciones nulas (por ejemplo, embarajar las coordenadas o usar embeddings aleatorios) y calcular T(k) y C(k) para cada una.  
- Eso te da una distribución de referencia; puedes ver si el T(k) y C(k) de tu modelo están muy por encima (p-valores pequeños).

**Valoración:**  
El planteamiento es correcto y la referencia Venna & Kaski es la adecuada para estas métricas y su interpretación.

---

## 4. Experimento C – Validación perceptual con Bowling (2018) y Harrison & Pearce (2020)

**Objetivo:**  
Ver si distancias en tu espacio geométrico se alinean con cómo las personas perciben **similitud de intervalos** y **consonancia simultánea de acordes**.

### Dataset 1 – Bowling et al. (2018): intervalos vocales

- Bowling y colaboradores han trabajado en cómo los humanos perciben la similitud y la “naturalidad” de intervalos vocales, y Harrison & Pearce citan y discuten estos trabajos en su revisión sobre consonancia.  
- Típicamente, estos estudios recogen juicios humanos sobre qué intervalos se parecen o resultan más agradables/estables.

**Cómo encaja con tu modelo:**

- Para cada intervalo del dataset, calculas la característica que tu modelo usaría (rugosidad, estructura de intervalos, etc.).  
- Construyes una matriz de “distancias de modelo” y la comparas con la matriz de distancias perceptuales humanas.  
- Usas **correlación de Spearman ρ** entre las entradas de ambas matrices:  
  - Spearman no mira los valores crudos, sino el **orden**: si un intervalo A es percibido como más similar a B que a C, quieres que el modelo también lo refleje.

### Dataset 2 – Harrison & Pearce (2020): consonancia simultánea

Harrison & Pearce (2020, *Psychological Review*) recopilan y reanalizan datos de consonancia simultánea de más de 500 participantes y proponen un modelo que combina **interferencia (rugosidad)**, **periodicidad/armonicidad** y **familiaridad cultural**.

**Tu uso propuesto:**

- Para cada acorde experimental en sus datasets, calculas:  
  - Distancia en tu espacio o rugosidad del acorde según tu modelo.  
  - Consonancia media reportada por participantes.  
- Vuelves a usar **Spearman ρ** entre “predicción del modelo” y “juicios humanos”.  
- Además, propones **correlación parcial**, controlando por el **número de notas comunes** entre acordes.  
  - Intuición: acordes que comparten muchas notas tienden a sonar más parecidos; quitar ese efecto te deja ver lo que aporta la rugosidad por sí sola.

**Conceptos estadísticos explicados:**

- **Correlación de Spearman:**  
  - Ordenas las observaciones de menor a mayor en cada variable.  
  - Calculas la correlación lineal entre esos rangos.  
  - Es robusta a no-linealidades suaves y outliers; ideal cuando no sabes si la relación es exactamente lineal.  
- **Correlación parcial:**  
  - Es como preguntar: “si fijo el ‘número de notas comunes’ constante, ¿siguen estando asociadas la distancia del modelo y la consonancia humana?”.  
  - Técnicamente, es la correlación entre los residuos de dos regresiones (cada variable ajustada respecto a la variable de control).

**Valoración y referencias:**  
Este experimento es muy sólido conceptualmente: usas datasets independientes y muy respetados, y las métricas elegidas (Spearman, correlación parcial) son estándar en psicología de la música.

---

## 5. Experimento D – Sustituibilidad funcional en estilo Bach

**Qué haces:**  
- Corpus: corales de Bach (JSB Chorales), alrededor de 370–400 piezas, disponible en múltiples ediciones y datasets.  
- Defines que dos acordes son **sustitutos funcionales** si comparten el mismo contexto armónico (mismos vecinos n−1 y n+1).  
- Hipótesis: pares como IV y ii6 en el mismo contexto deberían ser vecinos cercanos en tu espacio geométrico.

**Musicalmente:**

- En armonía barroca, acordes de distinta “superficie” (distintas alturas) pueden cumplir funciones parecidas si su contexto tonal y su conducción de voces son equivalentes.  
- Tu modelo solo ve la **fotografía vertical instantánea** (rugosidad, estructura intervalar), no la línea de tiempo ni la conducción de voces; por eso es muy interesante ver hasta qué punto **la función emerge** de las propiedades sensoriales.

**Cómo reforzar el diseño estadísticamente:**

- Para cada par de acordes candidatos a “sustitutos”, calculas su distancia en el espacio.  
- Construyes dos conjuntos de distancias:  
  - Distancias entre acordes marcados como funcionalmente equivalentes.  
  - Distancias entre acordes no equivalentes (muestra de control).  
- Puedes comparar estas distribuciones mediante:  
  - Un test no paramétrico (Mann–Whitney) para ver si las distancias de los pares funcionales son significativamente menores.  
  - Un AUC/ROC donde la “clase positiva” son pares funcionales y el score es “distancia negativa” (más pequeño, más probable sustituto).

**Valoración:**  
Método conceptualmente fuerte, pero necesitas especificar mejor las métricas y el procedimiento estadístico (distribuciones de distancias, tests) para que sea realmente convincente.

---

## 6. Experimento E – Consistencia estilística (Bach vs ruido)

**Qué haces:**  
- Proyectas tres tipos de acordes:  
  - A: acordes que aparecen en corales de Bach.  
  - B: “ruido uniforme” (acordes generados al azar).  
  - C: acordes extremos (clusters, poliacordes).  
- Quieres cuantificar qué tan separados están estos “manifolds” usando **divergencia de Kullback–Leibler**, **cross-entropy** y **Silhouette score**.

**Conceptos estadísticos/musicales simplificados:**

- Piensa el mapa 2D como un “plano” donde caen puntos (acordes). Cada grupo (Bach, ruido, extremos) genera una **distribución de densidad** en ese plano.  
- **Kullback–Leibler (KL) divergence:**  
  - Mide cuánto se parece una distribución a otra.  
  - Si la densidad de Bach y la de ruido fueran idénticas, la KL sería 0 (el modelo no distinguiría estilo).  
  - Cuanto mayor sea la KL, más “sorprendente” es ver acordes de Bach si asumieras el modelo de ruido, lo que indica buena separación de estilo.  
- **Cross-entropy:**  
  - Similar idea; mide el promedio de “sorpresa” cuando usas un modelo de probabilidad para describir datos de otra distribución.  
- **Silhouette score (por grupo):**  
  - Para cada acorde, mira qué tan cerca está del centro de su propio grupo comparado con otros grupos.  
  - Valores cercanos a 1 indican que el punto está bien metido en su clúster; valores negativos indican que estaría mejor en otro clúster.

**Valoración:**  
Este experimento es muy potente para vender la idea de un “manifold barroco” frente al ruido; combina bien con el Exp D (función interna del estilo Bach). La elección de KL, cross-entropy y silhouette es conceptualmente adecuada.

---

## 7. Resumen crítico y sugerencias

- **Lo más fuerte de tu diseño:**  
  - Combina validación **matemática/topológica** (Exp B) con validación **psicoacústica** (Exp C) y **musical/estilística** (Exp A, D, E).  
  - Usa métricas estándar (trustworthiness, continuity, Spearman, KL, silhouette), bien alineadas con la literatura.  

- **Ajustes recomendados:**  
  - Reformular el Exp 3 para que sea coherente con el concepto de **pitch chord** y la dependencia del registro: esperar **distancias pequeñas**, no 0.  
  - Especificar, para los Exp D y E, exactamente qué tests estadísticos usarás (distribuciones de distancias, tests no paramétricos, AUC, etc.).  
  - Al escribir, explicitar de manera muy sencilla:  
    - Qué es T(k) y C(k) (vecinos falsos vs roturas de vecindario).  
    - Qué es Spearman (correlación de rangos) y correlación parcial (quitar el efecto de un factor de confusión).  
    - Qué miden KL y silhouette (separación de distribuciones y de clústeres).
