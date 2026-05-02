# Guion maestro de defensa de tesis

## Modelo computacional para la exploración de acordes en la composición musical

**Autor:** Hernán Santiago Angarita García  
**Programa:** Maestría en Matemática Aplicada — Universidad Nacional de Colombia  
**Director:** Andrés Torres  
**Codirector:** Francisco Gómez  
**Versión de trabajo:** integración de maqueta, idea conversada, tesis, artículo y precisiones posteriores

---

## Propósito de este documento

Este documento busca funcionar como **guion completo de defensa**. No es solamente un libreto oral, ni solamente un marco teórico ampliado. La idea es que sirva a la vez como:

1. **mapa narrativo** de la exposición;
2. **reserva de contenido** para extraer diapositivas;
3. **banco de definiciones, ejemplos y transiciones** para responder preguntas del jurado;
4. **documento integrador** entre la tesis escrita, el artículo corto y la idea conversada del guion.

La intención central es que la defensa no se sienta como un resumen burocrático del manuscrito, sino como una historia intelectual clara: **por qué importa este problema, qué faltaba en el estado del arte, qué propuse exactamente, cómo lo evalué, qué encontré, qué significa y qué queda abierto**.

---

# Título sugerido para la defensa

## Modelo computacional para la exploración perceptual de acordes en la composición musical

> Variante oral posible: **Explorar acordes por su huella perceptual: un modelo computacional basado en rugosidad**.

La tesis conserva su título institucional. La charla puede darse el lujo de hacer más explícito el corazón del problema: la exploración perceptual de acordes aislados.

---

# 1. Apertura y gancho inicial

## 1.1 Idea rectora de la apertura

La apertura debe plantear desde el comienzo que esta investigación no trata solo de teoría musical abstracta. Trata de una pregunta más amplia y más humana: **cómo explorar el espacio de las sonoridades posibles cuando lo que queremos preservar no es solo la estructura del acorde, sino algo de lo que el oído realmente percibe**.

La primera tarea de la defensa es construir relevancia. Para ello conviene abrir en dos tiempos: primero con un **gancho contemporáneo**, luego con una **formulación precisa del problema**.

## 1.2 Posibles ganchos de entrada

Se puede escoger uno solo o combinar dos muy brevemente.

### Gancho A: actualidad tecnológica

Hoy la industria de la música, la generación con IA, los sistemas de recomendación y hasta el procesamiento perceptual de audio dependen cada vez más de representaciones computacionales del sonido y de la percepción. Sin embargo, cuando se trata de armonía, muchas representaciones siguen respondiendo mejor a categorías teóricas heredadas que a la textura sonora real del acorde.

### Gancho B: problema intelectual

Durante décadas la teoría musical ha sabido clasificar acordes, relacionarlos, agruparlos, estudiarlos como objetos algebraicos o geométricos. Pero una pregunta sigue siendo incómoda: **si dos acordes suenan parecidos, dónde está codificada esa cercanía en el modelo**.

### Gancho C: formulación breve y poderosa

> Sabemos nombrar muchos acordes. Sabemos clasificarlos. Sabemos incluso construir geometrías elegantes para relacionarlos. Pero todavía es difícil explorar el espacio de los acordes desde una pregunta muy concreta: **qué tan cerca suenan entre sí dos acordes aislados**.

## 1.3 Transición a la motivación

A partir de ahí conviene decir que esta tesis se ubica exactamente en esa grieta: entre la clasificación estructural del acorde y su identidad perceptual.

---

# 2. Motivación, relevancia y justificación

## 2.1 La música y el problema de la simultaneidad

La música puede pensarse, de forma elemental, en tres niveles: melodía, ritmo y armonía. Aquí nos interesa la armonía, y en particular los **acordes**, entendidos como conjuntos de notas que suenan simultáneamente.

Pero en ese punto aparece una dificultad decisiva. Un acorde puede describirse de muchas maneras:

- como nombre armónico;
- como conjunto de clases de altura;
- como voicing particular;
- como evento MIDI;
- como colección de frecuencias;
- como fenómeno perceptual.

No todas esas descripciones conservan la misma información. Algunas sirven muy bien para clasificar; otras sirven mejor para sintetizar o para transcribir; otras son más sensibles a lo que el oído experimenta.

## 2.2 Qué falta en muchas representaciones existentes

Buena parte de la tradición teórica occidental ha privilegiado categorías funcionales, equivalencias algebraicas o relaciones de conducción de voces. Eso ha sido extremadamente fecundo. Pero cuando queremos **explorar sonoridades posibles**, y no solamente reconocer estructuras ya consagradas, se vuelve importante preguntarse si esas representaciones conservan suficiente información perceptual.

Aquí aparece la intuición central del trabajo: si una representación identifica como equivalentes objetos que el oído no percibe como equivalentes, entonces esa representación puede ser excelente para cierta pregunta teórica, pero insuficiente para una pregunta perceptual y exploratoria.

## 2.3 Justificación artística y computacional

Esta investigación se justifica en dos frentes.

### Justificación musical

Un compositor no solo trabaja con etiquetas; trabaja con texturas, tensiones, densidades, colores armónicos, proximidades sonoras. Si queremos ampliar el repertorio de sonoridades explorables, necesitamos un espacio donde la cercanía entre acordes no dependa únicamente de la tradición funcional o de equivalencias abstractas.

### Justificación computacional

Desde el punto de vista computacional, un algoritmo puede generar millones de acordes. Lo difícil no es generarlos: lo difícil es **organizarlos** de una manera útil. Se necesita una representación que permita comparar, medir distancias, visualizar regiones y sugerir vecindades con sentido musical.

## 2.4 Justificación epistemológica

La tesis también responde a una pregunta epistemológica: ¿qué pasa si intentamos construir un espacio de acordes no desde la gramática histórica de la tonalidad, sino desde un descriptor psicoacústico explícito, sensible a la distribución interna de la rugosidad?

---

# 3. Planteamiento del problema

## 3.1 Formulación del problema

El análisis armónico tradicional y muchos enfoques computacionales describen, clasifican o aprenden repertorios ya estabilizados. Pero eso no resuelve automáticamente el problema de **explorar** nuevas configuraciones armónicas desde una noción perceptual de similitud.

El punto de partida de esta tesis es el siguiente:

> muchas sonoridades posibles quedan por fuera del vocabulario habitual no porque sean incomputables, sino porque no contamos con una representación que permita ubicarlas y compararlas de una manera perceptualmente informada.

## 3.2 Pregunta central

La pregunta central puede expresarse así:

> ¿Es posible construir un espacio de representación computacional para acordes aislados, basado en su huella psicoacústica de rugosidad, que permita ubicar y descubrir geométricamente acordes sustitutos con sonoridades afines?

## 3.3 Hipótesis de trabajo

La hipótesis que guía el trabajo es que, dada una población grande de acordes sin contexto tonal o progresivo fijo, existen configuraciones no exploradas que pueden funcionar como sustitutos de otras conocidas si se comparan mediante una representación que preserve la distribución perceptual de su rugosidad.

Dicho de forma más conceptual:

> si preservamos mejor la estructura interválica y la forma en que esa estructura produce fricción espectral, podemos inducir un espacio donde las cercanías tengan significado sonoro y no solo taxonómico.

---

# 4. Marco conceptual y teórico

Este bloque no debe sonar como una enciclopedia de nombres propios. Debe construir una cadena lógica: cómo se ha pensado la consonancia, cómo se ha formalizado el acorde, qué modelos existen y por qué todavía queda un vacío.

## 4.1 Qué es una nota, qué es un acorde y cómo se representan

Antes de entrar a la parte técnica, conviene fijar algunas definiciones básicas.

### Nota

Una nota puede entenderse como un evento sonoro con cierta frecuencia fundamental, aunque los sonidos musicales reales suelen ser tonos complejos y no sinusoides puras.

### Acorde

En esta tesis, un acorde se entiende como **un conjunto finito de notas que suenan simultáneamente**. El énfasis está puesto en el acorde como **objeto vertical aislado**, no como parte de una progresión.

### Representaciones posibles

Un mismo acorde puede representarse de varias maneras:

- **partitura** o escritura musical;
- **MIDI**, si queremos preservar alturas absolutas y registro;
- **señal de audio**, si queremos estudiar el fenómeno acústico completo;
- **pitch-class set**, si queremos una representación algebraica abstracta;
- **vector de clases de altura o de intervalos**, si queremos una codificación estructural;
- **descriptor psicoacústico**, si queremos modelar una dimensión de la percepción.

## 4.2 Helmholtz: la disonancia como fenómeno físico-fisiológico

Un primer gran momento histórico aparece con Helmholtz. Su importancia no está solamente en su prestigio como científico, sino en el cambio de pregunta que introduce. En lugar de contentarse con una explicación puramente numérica o metafísica de la consonancia, intenta explicar la disonancia desde la interacción física de los sonidos y desde la fisiología del oído.

Su gran intuición es que la aspereza o rugosidad sensorial puede originarse en los batimientos rápidos producidos por parciales cercanos. En otras palabras, la disonancia deja de ser un asunto meramente cultural o normativo y se convierte también en un problema de interferencia acústica.

### Idea útil para la charla

Helmholtz abre el camino para pensar que la armonía no solo se puede clasificar: también se puede **medir** en alguna dimensión perceptual.

## 4.3 Plomp y Levelt: banda crítica y rugosidad

Plomp y Levelt refinan ese programa al mostrar experimentalmente que la máxima disonancia sensorial no depende de una diferencia absoluta de frecuencias constante, sino de la separación relativa respecto de la **banda crítica**. Allí entra la membrana basilar, la tonotopía coclear y la idea de que el sistema auditivo no resuelve igual dos componentes cercanos en todas las regiones del espectro.

Este punto es decisivo para la tesis por dos razones:

1. la rugosidad tiene una **base fisiológica** y no puramente cultural;
2. la rugosidad depende del **registro absoluto**, no solo de clases de altura abstractas.

## 4.4 Sethares: consonancia como función del par intervalo–timbre

Sethares da el paso computacional crucial. A partir de la curva de Plomp–Levelt, formaliza un modelo para calcular la rugosidad entre tonos complejos. Eso permite pasar de una intuición fisiológica general a una herramienta concreta de cálculo.

Además, Sethares insiste en una idea de enorme peso conceptual: la consonancia no es una propiedad pura del intervalo abstracto, sino del par **intervalo–timbre**. Ese punto es muy valioso para tu defensa porque muestra que una representación sensible al comportamiento espectral no es una extravagancia metodológica, sino una continuidad rigurosa de la tradición psicoacústica.

## 4.5 Forte y la teoría de pitch-class sets

Allen Forte resolvió un problema distinto y lo resolvió brillantemente: cómo clasificar estructuras armónicas en un catálogo finito, especialmente útil para repertorios atonales. La reducción módulo 12, las equivalencias por transposición e inversión, la forma normal y la forma prima son instrumentos extremadamente poderosos para taxonomía estructural.

Pero la tesis necesita señalar con claridad qué se pierde allí desde el punto de vista perceptual.

### Lo que se gana con Forte

- catálogo finito y navegable;
- lenguaje taxonómico fuerte;
- formalización algebraica precisa;
- invariancias muy útiles para ciertas preguntas analíticas.

### Lo que se pierde para esta tesis

- registro absoluto;
- diferencia entre voicings e inversiones perceptualmente no equivalentes;
- densidad y disposición vertical concreta;
- comportamiento espectral;
- rugosidad auditiva.

La defensa debe ser justa aquí: no se trata de refutar a Forte, sino de decir que responde otra pregunta. Tu trabajo aparece donde esa pregunta deja de ser suficiente.

## 4.6 Tymoczko y las geometrías de acordes

Los trabajos de Tymoczko y otros autores sobre orbifolds, voice-leading y espacios geométricos son fundamentales para mostrar que la armonía puede pensarse como un espacio navegable. Esa tradición es muy importante para la tesis porque legitima la idea misma de construir vecindades, regiones y trayectorias.

Sin embargo, el criterio de cercanía en esos modelos suele depender del costo de movimiento entre voces, de equivalencias por octava o de sintaxis armónica. El centro de gravedad de tu tesis es distinto: no parte de la progresión, sino del **acorde aislado** y de su huella perceptual.

## 4.7 Otros enfoques relevantes

También conviene mencionar brevemente:

- **Cambouropoulos** y esquemas de representación armónica;
- **Himpel** y la geometría de la percepción musical;
- enfoques de **machine learning** y embeddings aprendidos de corpus;
- trabajos sobre **armonicidad** y **periodicidad**, especialmente Harrison y Pearce;
- el experimento de **McDermott con los Tsimané**, que ayuda a separar lo fisiológico de lo cultural.

## 4.8 El punto de McDermott y los Tsimané

El caso Tsimané es muy útil narrativamente porque permite decir algo fino: la preferencia estética por ciertas consonancias no es universal en el sentido cultural, pero la rugosidad como fenómeno sensorial sí apunta a un sustrato fisiológico más estable. Eso ayuda a justificar por qué un modelo basado en rugosidad no pretende modelar toda la cultura musical, pero sí una dimensión real del fenómeno auditivo.

## 4.9 Vacío que la tesis intenta llenar

Con todo esto, el vacío queda formulado así:

- los modelos algebraicos clasifican muy bien, pero comprimen demasiado;
- los modelos geométricos relacionales capturan transiciones, pero no necesariamente el acorde aislado como objeto perceptual;
- los modelos de IA aprenden estilos y sesgos históricos, pero no siempre ofrecen interpretabilidad perceptual explícita;
- la rugosidad suele reducirse a un único escalar, perdiendo su distribución interna.

La tesis se ubica exactamente allí: propone **no tirar la rugosidad a un solo número**, sino conservar su distribución por clases de intervalo.

---

# 5. Objetivos

## 5.1 Objetivo general

Desarrollar un modelo computacional para la construcción y exploración de un espacio de representación y sustitución armónica.

## 5.2 Objetivos específicos

1. **Modelar matemáticamente** un conjunto de reglas y características pertinentes para la generación y caracterización de acordes dentro del espacio de representación definido.
2. **Construir una representación intervalar y perceptual** que preserve mejor la estructura interna del acorde y la distribución de su rugosidad.
3. **Implementar técnicas de reducción de dimensionalidad** que permitan visualizar e interpretar el espacio inducido.
4. **Evaluar cuantitativamente** la calidad de la representación, tanto en términos perceptuales como geométricos.
5. **Explorar una noción de sustitución armónica** basada en proximidad métrica dentro del espacio construido.

## 5.3 Cómo deben sonar estos objetivos en la defensa

En la charla no conviene leer los objetivos como lista administrativa. Conviene hacerlos sonar como una secuencia lógica:

> preservar el acorde, describir su rugosidad, construir un espacio, ponerlo a prueba y mostrar que ese espacio puede usarse para explorar sustituciones.

---

# 6. Metodología

Aquí comienza realmente el aporte propio. Este bloque debe sentirse como el corazón de la tesis.

## 6.1 Decisión metodológica fundamental: el acorde como objeto aislado

La primera decisión fuerte del trabajo fue tratar el acorde como **evento sonoro vertical aislado**. Esto significa que la tesis no modela directamente progresiones, función tonal ni conducción de voces. El interés está puesto en el acorde mismo: en su organización interválica, en su registro y en su comportamiento psicoacústico.

Esta decisión delimita el alcance del modelo y al mismo tiempo lo hace más claro.

## 6.2 Sistema de referencia: MIDI, frecuencia y temperamento igual

El universo de trabajo se construye sobre notas MIDI, lo que permite preservar altura absoluta y pasar de manera directa a frecuencias físicas mediante la fórmula estándar con referencia A4 = 440 Hz. Esto es importante porque la rugosidad depende de frecuencia real y no solamente de clase de altura.

Aquí debe explicarse de forma sencilla que escoger el sistema temperado de 12 tonos no significa afirmar que toda la música se reduce a él, sino fijar un marco concreto y operativo para la exploración.

## 6.3 Definición formal del acorde

El acorde se modela como una tupla ordenada y estrictamente creciente de notas MIDI:

\[
\mathbf{n}=(n_1,n_2,\dots,n_m), \qquad n_1<n_2<\cdots<n_m.
\]

La ordenación fija la disposición vertical del acorde y la condición estricta excluye unísonos exactos. Esto preserva la estructura real del objeto sonoro en lugar de colapsarlo mediante permutaciones o equivalencias demasiado fuertes.

## 6.4 Exploración combinatoria del universo de acordes

Una parte importante del aporte práctico de la tesis es que el repositorio no parte de un catálogo fijo de acordes históricos. Permite generar poblaciones de acordes combinatoriamente, controlando:

- alfabetos disponibles;
- cardinalidades permitidas;
- rangos u octavas;
- anclajes;
- filtros estructurales;
- restricciones de distancia.

Aquí conviene detenerse y decir que explorar no significa simplemente enumerar. Significa **definir un universo manejable y justificable**.

### Ejemplo narrativo útil

Si permitimos un alfabeto mayor, una sola octava, una cardinalidad fija y un anclaje en Do, obtenemos una región muy controlada del espacio. Si aumentamos octavas, cardinalidades o alfabetos, el número de objetos crece rápidamente. Esto ayuda a mostrar por qué la organización del espacio es un problema real y no una formalidad.

## 6.5 Anclaje en Do y control experimental

En varios experimentos se restringen los acordes a una nota grave fija, por ejemplo C3. Esto no empobrece el universo; más bien elimina el efecto trivial de la transposición global y deja ver con mayor claridad la estructura interna del acorde y la influencia del registro.

Una buena frase para la defensa sería:

> anclar el acorde no significa perder generalidad musical, sino ganar control experimental.

## 6.6 Descriptor de estructura interválica

El primer componente de la representación es estructural. Para cada acorde se construye un vector que cuenta cuántas veces aparece cada clase de intervalo entre pares de notas.

La tesis adopta **12 bins** y no 6. Esa decisión es crucial porque evita colapsar intervalos complementarios que pueden ocupar posiciones perceptualmente diferentes dentro del acorde.

\[
ic_k(\mathbf{n})=\#\{(i,j): i<j,\ (n_j-n_i)\equiv k\pmod{12}\}, \quad k=1,\dots,12.
\]

La idea de fondo es esta: antes de ponderar perceptualmente el acorde, hay que preservar su distribución interna de distancias.

## 6.7 Modelo psicoacústico de rugosidad

Sobre esa estructura interválica se incorpora el componente perceptual. Se adopta el modelo de rugosidad derivado de Plomp–Levelt y formalizado por Sethares.

La disonancia entre dos parciales de frecuencias cercanas se modela mediante una función del tipo:

\[
d(f_a,f_b,A_a,A_b)=A_aA_b\left(e^{-b_1 s \Delta f}-e^{-b_2 s \Delta f}\right),
\]

donde \(\Delta f=|f_a-f_b|\), y \(s\) ajusta la separación por la banda crítica local.

La tesis usa los parámetros:

- \(b_1=3.5\)
- \(b_2=5.75\)
- \(s_1=0.021\)
- \(s_2=19\)

Cada nota se modela como tono complejo armónico con:

- \(H=6\) armónicos;
- decaimiento exponencial \(\delta=0.88\).

La rugosidad entre dos notas se calcula sumando las interacciones entre sus parciales.

## 6.8 El paso central: perfil de rugosidad en \(\mathbb{R}^{12}\)

Este es el núcleo original del modelo. En lugar de reducir todo a una única rugosidad total, se define un vector bruto \(\Phi_{raw}\in\mathbb{R}^{12}\) cuya componente \(k\) acumula la rugosidad de todos los pares de notas separados por \(k\) semitonos módulo 12:

\[
\Phi_{raw,k}(\mathbf{n})=
\sum_{1\le i<j\le m \atop (n_j-n_i)\equiv k\ (\mathrm{mod}\ 12)}
R(f(n_i),f(n_j)).
\]

Este vector preserva **dónde** está la rugosidad, no solamente cuánta hay. Esa es la diferencia decisiva frente al baseline escalar.

### Comentario importante sobre el bin 12

Los bins 1 a 11 corresponden a intervalos cromáticos de 1 a 11 semitonos. El bin 12 almacena la clase residuo 0 módulo 12, es decir, octavas y sus múltiplos. La justificación es psicoacústica: pares separados por octavas exactas presentan alineación espectral y rugosidad despreciable bajo el modelo Plomp–Levelt/Sethares.

## 6.9 Escalar baseline y normalización

El baseline escalar tradicional se define como:

\[
R_{total}=\|\Phi_{raw}\|_1.
\]

Ese número conserva la magnitud total de la rugosidad, pero destruye la distribución interna por bins.

Para comparar acordes de distintas cardinalidades, la tesis explora varias normalizaciones. La configuración más importante para la geometría fue la normalización por clase con exponente \(\alpha=0.75\):

\[
\Phi_{0.75,k}(\mathbf{n})=
\frac{\Phi_{raw,k}(\mathbf{n})}{\max\{m_k(\mathbf{n}),1\}^{0.75}}.
\]

La idea de esta ponderación es reducir el sesgo por cardinalidad sin borrar del todo la información de magnitud.

## 6.10 Métricas de distancia

Sobre los perfiles normalizados se construyen métricas de similitud. La tesis usa principalmente:

- **distancia euclidiana**, cuando interesa preservar forma y magnitud;
- **similitud coseno**, en experimentos complementarios donde interesa la distribución relativa.

La configuración central del trabajo es **distancia euclidiana** sobre el perfil normalizado principal.

## 6.11 Validación perceptual con Bowling y justificación de Ridge

Aquí entra uno de los bloques metodológicos más importantes y que sí debe aparecer explícitamente en el guion.

Se utilizó la base de Bowling, Purves y Gill, compuesta por **298 acordes** —díadas, tríadas y tétradas— con calificaciones humanas de consonancia. Para cada estímulo se calcularon dos representaciones:

1. el escalar de rugosidad total \(R_{total}\);
2. el perfil completo \(\Phi_{raw}\in\mathbb{R}^{12}\).

La tarea se formuló como problema supervisado de predicción de consonancia percibida. Se compararon tres modelos:

- regresión lineal sobre \(R_{total}\);
- control polinómico de grado 3 sobre el mismo escalar;
- regresión **Ridge** sobre el vector 12D.

### ¿Por qué Ridge?

Porque los bins cromáticos adyacentes comparten interacciones parciales y, por tanto, están correlacionados. La penalización \(L_2\) permite estabilizar coeficientes sin eliminar bins completos. Esto es metodológicamente importante porque evita una lectura ingenua del vector como si fueran variables totalmente independientes.

### Hiperparámetro

El parámetro de regularización se fijó **a priori** en \(\alpha=1.0\). Además, una búsqueda anidada sobre \(\{0.1,1,10,100\}\) confirmó que el desempeño era estable en ese rango. Así se evita ambigüedad sobre posible leakage en la selección del hiperparámetro.

### Validación

Todos los resultados se reportaron con validación cruzada aleatorizada de **cinco folds**, usando predicciones estrictamente fuera de muestra.

## 6.12 Reducción dimensional y visualización

Para explorar la geometría del espacio inducido por la matriz de distancias, se emplearon principalmente dos métodos:

- **MDS**, por su capacidad para preservar estructura global de distancias;
- **UMAP**, por su utilidad exploratoria en la preservación de vecindades locales.

La tesis privilegia MDS como referencia analítica central. UMAP aparece más como exploración complementaria.

## 6.13 Cómo se evalúa si la proyección es buena

La metodología debe explicar de forma natural cómo se valora la calidad del embedding, sin introducir una diapositiva forzada tipo “por qué esto sí funciona”. Los criterios usados fueron:

- **Stress** de Kruskal;
- **Trustworthiness**, para estructura local;
- **correlación de Spearman** entre distancias originales y proyectadas.

La idea importante es que estas métricas no legitiman por sí solas el modelo, pero sí controlan cuánto se deforma el espacio cuando se lo mira en dos dimensiones.

## 6.14 Diseño experimental global

La metodología culmina con una batería de experimentos complementarios.

### Experimento 1: validación perceptual

Base de Bowling, comparación 1D vs 12D.

### Experimento 2: conjunto canónico controlado

Díadas y tríadas básicas con inversiones, ancladas en C3, para verificar si la representación organiza coherentemente familias conocidas.

### Experimento 3: robustez frente a transposición y registro

Las cuatro cualidades triádicas básicas, en 12 raíces cromáticas y tres posiciones de registro.

### Experimento 4: espacio extendido

Conjunto de acordes de cardinalidad 2–5 con séptimas, suspensiones y extensiones.

### Experimento 5: contraste con masa aleatoria

Integración de miles de estructuras aleatorias para observar si el repertorio canónico ocupa una región especial del espacio.

### Experimento 6: sustitución armónica por proximidad métrica

Selección de un acorde consulta y exploración de vecinos cercanos y lejanos.

## 6.15 Idea de cierre para la metodología

> En resumen, la metodología construye un camino completo: generar acordes, preservarlos como objetos, describir su estructura, calcular su rugosidad, distribuir esa rugosidad en doce dimensiones, definir distancias, proyectar el espacio, evaluarlo y finalmente usarlo para explorar vecindades armónicas.

---

# 7. Resultados

## 7.1 Resultado principal: el perfil 12D mejora la predicción perceptual

El resultado más importante de la tesis es que el modelo 12D predice los juicios humanos de consonancia con mucha mayor fidelidad que el baseline escalar.

En la versión afinada del artículo, los resultados fuera de muestra fueron:

- **modelo escalar**: \(r=0.468\), \(R^2=0.218\)
- **modelo 12D con Ridge**: \(r=0.767\), \(R^2=0.588\)
- **control polinómico de grado 3**: \(r=0.456\), \(R^2=0.204\)

El control polinómico es importante porque ayuda a sostener que la mejora no proviene simplemente de introducir no linealidad, sino de conservar la distribución interna de la rugosidad.

## 7.2 Resultado por tamaño del acorde

Una precisión metodológica muy útil para defender este resultado es que el desempeño mejora con el tamaño del acorde:

- **tríadas**: \(R^2=0.574\) frente a \(0.373\)
- **tétradas**: \(R^2=0.670\) frente a \(0.482\)

Esto sugiere que el perfil 12D se vuelve especialmente informativo cuando la estructura interna del acorde es más rica.

## 7.3 Geometría del conjunto canónico

Cuando se proyecta mediante MDS el conjunto canónico de 24 acordes, aparece una organización musicalmente plausible:

- las díadas siguen un gradiente de rugosidad desde la octava hasta la segunda menor;
- las tríadas mayores y menores quedan cercanas;
- las disminuidas se desplazan a zonas de mayor rugosidad;
- la aumentada ocupa una región separada;
- las inversiones quedan próximas entre sí, pero no colapsadas.

Esto satisface una condición importante del diseño: preservar cercanía familiar sin borrar diferencias internas relevantes.

## 7.4 Robustez frente a transposición y registro

En el experimento de 144 tríadas distribuidas en varias raíces y registros, la representación mantiene compactas las nubes correspondientes a una misma cualidad armónica. Al mismo tiempo, el color asociado a la rugosidad total cambia sistemáticamente con el registro, tal como lo predice la dependencia frecuencial de la banda crítica.

Esto muestra que el modelo distingue bien dos cosas:

1. la identidad estructural de la familia del acorde;
2. el efecto perceptual del registro.

## 7.5 Espacio armónico ampliado

En el conjunto extendido de cardinalidad 2–5, la geometría sigue siendo interpretable. Las familias canónicas permanecen en regiones coherentes, mientras otras configuraciones más densas o complejas se dispersan más.

Lo importante aquí es que la organización no depende únicamente del número de notas. Acordes con igual cardinalidad pueden ubicarse en regiones distintas según la distribución específica de su rugosidad entre clases interválicas.

## 7.6 Masa aleatoria y subespacio histórico

Cuando se integran 5.000 estructuras aleatorias junto con los acordes de referencia, el repertorio armónico occidental aparece concentrado en una subregión relativamente compacta del espacio, asociada sobre todo a rugosidades bajas y medias.

Este hallazgo es conceptualmente muy sugerente: la práctica armónica histórica no se dispersa arbitrariamente por el universo combinatorio. Más bien habita una región especial del espacio inducido por el descriptor.

## 7.7 Métricas de calidad del embedding

Para el conjunto extendido y para el contraste masivo, la tesis reporta métricas como trustworthiness, Spearman y stress. El mensaje oral no necesita saturarse con números, pero sí conviene decir que:

- la preservación local es alta;
- el orden relativo global de distancias se conserva bien;
- la distorsión global del embedding se mantiene moderada.

Una forma elegante de decirlo sería:

> el mapa no sustituye al espacio original, pero lo deforma lo suficientemente poco como para ser una herramienta confiable de exploración.

## 7.8 Sustitución armónica por proximidad métrica

Con el acorde **sus4** como consulta, la exploración de vecinos cercanos devuelve acordes de la familia de las suspensiones y configuraciones texturalmente afines. Los vecinos lejanos incluyen acordes disminuidos, aumentados y tríadas con concentración de intervalos mucho más tensionales.

Esto muestra que la métrica en \(\mathbb{R}^{12}\) puede funcionar como un continuo de similitud textural, útil para sugerir sustituciones sin apelar primero a reglas tonales o a conducción de voces.

---

# 8. Discusión

## 8.1 Qué se ha mostrado realmente

El punto central no es que la rugosidad explique toda la consonancia, ni que este modelo derrote a toda la teoría armónica previa. Lo que se ha mostrado es algo más preciso:

> si ya vamos a trabajar con rugosidad, colapsarla a un solo escalar desperdicia información perceptualmente relevante.

La tesis demuestra que preservar la distribución de esa rugosidad por clases de intervalo mejora la representación del acorde aislado.

## 8.2 Discusión frente a Forte

Frente a la teoría de conjuntos de clases de altura, el trabajo muestra que una representación basada en equivalencias fuertes puede ser excelente para taxonomía estructural, pero insuficiente cuando la pregunta es perceptual.

No se trata de decir que Forte está mal, sino de subrayar que responde otra pregunta. La tesis responde una distinta: cómo conservar información que el oído sí percibe cuando las clases abstractas ya no bastan.

## 8.3 Discusión frente a Tymoczko y las geometrías de voice-leading

Los modelos geométricos clásicos organizan acordes en función del movimiento entre voces y de relaciones transformacionales. El espacio propuesto aquí se apoya en otra noción de cercanía: similitud textural inducida por el perfil de rugosidad.

Esto no los vuelve incompatibles. De hecho, una lectura productiva es pensar que tu trabajo puede complementar esas geometrías con un eje perceptual adicional.

## 8.4 Discusión frente a machine learning y embeddings aprendidos

Los embeddings aprendidos de corpus pueden capturar regularidades históricas o estilísticas, pero suelen heredar sesgos del repertorio y no siempre ofrecen interpretabilidad perceptual explícita. El espacio propuesto en esta tesis, en cambio, es interpretable desde su definición y no depende de entrenamiento sobre un corpus histórico.

## 8.5 Discusión frente a Harrison y Pearce

Aquí es importante ser cuidadoso. Harrison y Pearce muestran que roughness, harmonicity y periodicity contribuyen de forma independiente a la predicción de consonancia, y que combinadas pueden alcanzar desempeños más altos que roughness por sí sola.

La tesis debe reconocer eso explícitamente. El aporte no consiste en afirmar que la rugosidad basta, sino en mostrar que **una rugosidad distribuida en 12 bins ya mejora claramente frente al roughness escalar**.

## 8.6 Límites del modelo

La discusión debe declarar con honestidad varias limitaciones:

1. el modelo está restringido a **acordes aislados**;
2. no modela directamente progresiones ni voice-leading;
3. usa un espectro armónico idealizado;
4. no incorpora aún harmonicidad ni periodicidad como ejes explícitos;
5. el dataset de Bowling es relativamente modesto y culturalmente situado.

Lejos de debilitar la tesis, decir esto la fortalece, porque muestra que el alcance está controlado y bien delimitado.

## 8.7 Cómo defender el uso de MDS

Este punto hay que decirlo casi con las palabras del artículo revisado:

> la geometría bidimensional es un **consistency check**, no una validación independiente.

Eso protege la defensa frente a críticas sobre stress o deformaciones inevitables del embedding.

---

# 9. Trabajos futuros

## 9.1 Integrar harmonicidad y periodicidad

La línea más inmediata es extender el modelo hacia descriptores perceptuales más ricos, especialmente harmonicidad y periodicidad, para dialogar de forma más directa con el estado del arte contemporáneo.

## 9.2 Usar espectros tímbricos reales

Otra extensión natural es reemplazar el tono armónico idealizado por espectros obtenidos de instrumentos reales, o incluso estudiar cómo cambia la topología del espacio según el timbre.

## 9.3 Pasar de acordes aislados a progresiones

El trabajo actual se concentra en el acorde vertical. Una continuación muy natural sería estudiar perfiles de rugosidad sobre trayectorias armónicas y vincular el espacio propuesto con modelos de tensión temporal y conducción de voces.

## 9.4 Desarrollo de herramientas compositivas interactivas

El repositorio y la lógica del modelo permiten pensar en herramientas interactivas donde el compositor navegue vecindades, explore sustituciones, compare perfiles y descubra regiones del espacio armónico no necesariamente explotadas por la tradición.

## 9.5 Validación perceptual ampliada

También queda abierto diseñar nuevos experimentos perceptuales, quizá con más acordes, más timbres, más contextos culturales y tareas experimentales diferentes a la escala de consonancia de Bowling.

---

# 10. Conclusiones

## 10.1 Conclusión principal

La tesis propone una representación computacional de acordes basada en un **perfil de rugosidad distributivo en 12 dimensiones**. Esa representación preserva mejor la estructura perceptual del acorde aislado que la reducción tradicional a un escalar de rugosidad total.

## 10.2 Conclusión empírica

El perfil 12D mejora sustancialmente la predicción de juicios humanos de consonancia frente al baseline escalar y organiza de manera interpretable conjuntos canónicos, espacios ampliados y contrastes masivos con estructuras aleatorias.

## 10.3 Conclusión geométrica

El espacio métrico inducido no solo sirve para describir, sino también para **explorar**. Permite visualizar vecindades, distinguir inversiones, separar regiones de baja y alta rugosidad y proponer una noción de sustitución armónica por proximidad textural.

## 10.4 Conclusión conceptual

El aporte central del trabajo puede formularse así:

> la rugosidad no debe pensarse únicamente como una magnitud total, sino como una firma interna distribuida que puede servir de base cuantitativa para estudiar el acorde aislado como objeto perceptual.

## 10.5 Conclusión general

En conjunto, los resultados sugieren que la firma distributiva de rugosidad constituye una base cuantitativa coherente para el análisis sistemático y la exploración algorítmica de espacios armónicos multidimensionales.

---

# 11. Cierre oral sugerido

Una forma sobria y efectiva de cerrar la defensa sería:

> En esta tesis no intenté reemplazar la teoría armónica existente, sino responder una pregunta que ella sola no resolvía del todo: cómo representar y explorar acordes preservando algo de su identidad perceptual. La propuesta fue construir un perfil de rugosidad distribuido, inducir un espacio métrico a partir de él y mostrar que ese espacio tiene valor explicativo, geométrico y compositivo. Si el trabajo logra algo, es abrir una vía para pensar el acorde aislado no solo como categoría teórica, sino también como objeto perceptual navegable.

---

# 12. Agradecimientos

Aquí conviene cerrar de manera humana, breve y digna.

## Versión sobria

Quiero agradecer a mis directores, Andrés Torres y Francisco Gómez, por su guía y paciencia en este proceso. A mi familia y a las personas que me sostuvieron durante el desarrollo de este trabajo. Y a la Universidad Nacional de Colombia, por el espacio académico donde esta investigación pudo tomar forma.

## Versión un poco más personal

Quiero agradecer a mis directores, Andrés Torres y Francisco Gómez, por acompañar esta investigación con generosidad y exigencia. A mi familia, a mis amigos y a quienes me sostuvieron en los momentos difíciles del proceso. Y a la Universidad Nacional de Colombia, por haber hecho posible este recorrido.

---

# 13. Apéndice útil para la defensa oral

## 13.1 Preguntas difíciles previsibles

### ¿Por qué usar rugosidad y no harmonicidad?

Porque el objetivo del trabajo no era modelar toda la consonancia, sino construir y evaluar un descriptor de rugosidad más informativo que el roughness escalar tradicional. La comparación con harmonicity y periodicity queda abierta como continuación necesaria.

### ¿Por qué Ridge?

Porque las componentes del vector 12D están correlacionadas; Ridge estabiliza la estimación sin destruir bins relevantes. Además, el hiperparámetro se fijó a priori y se verificó con nested CV.

### ¿Por qué MDS si el stress no es cero?

Porque el MDS se usa aquí como herramienta exploratoria de consistencia geométrica, no como validación independiente. La interpretación se apoya también en trustworthiness y correlación de Spearman.

### ¿Por qué estudiar acordes aislados si la música ocurre en el tiempo?

Porque el problema del acorde aislado ya es suficientemente rico y no estaba resuelto de la manera que esta tesis propone. Incorporar temporalidad sería una continuación natural, no una corrección del enfoque.

### ¿Por qué 12 dimensiones y no 6?

Porque distinguir intervalos complementarios ayuda a preservar direccionalidad y distribución interna de la rugosidad. El objetivo era evitar colapsos prematuros de información perceptualmente relevante.

## 13.2 Referencias que conviene poder nombrar oralmente

- Helmholtz — *On the Sensations of Tone*.
- Plomp & Levelt (1965) — banda crítica y consonancia tonal.
- Sethares (1993, 2005) — rugosidad computacional y relación timbre–escala.
- Forte (1973) — *The Structure of Atonal Music*.
- Tymoczko (2006, 2011) — geometrías de acordes y voice-leading.
- Himpel (2022) — geometría de la percepción musical.
- Bowling, Purves & Gill (2018) — dataset de consonancia de acordes.
- McDermott et al. (2016) — Tsimané y variación cultural en percepción.
- Harrison & Pearce (2020) — roughness + harmonicity + periodicity.

---

# 14. Nota final de uso

Este documento debe leerse como **guion maestro**. No todo debe decirse en la defensa. Su función es permitir seleccionar, condensar y diseñar diapositivas sin perder el espesor conceptual de la investigación. Las diapositivas finales deben tener **poco texto, títulos sobrios, ejemplos bien escogidos, diagramas claros y una narrativa continua**.

El objetivo no es contar todo. El objetivo es que, al final de la defensa, el jurado entienda con claridad:

1. cuál era el vacío;
2. qué propuso exactamente la tesis;
3. por qué la propuesta es razonable;
4. cómo se puso a prueba;
5. qué encontró;
6. qué significa ese hallazgo dentro de la literatura.

