# Capítulo 3: Metodología

\label{chap:metodologia}

Este capítulo formaliza el marco matemático y computacional que construimos para analizar y visualizar el espacio de acordes. La idea central detrás de esta investigación es sencilla pero poderosa: esa **similitud sonora** de la que tanto hablamos cualitativamente en teoría musical, en realidad puede medirse de forma objetiva usando propiedades psicoacústicas. Específicamente, argumentamos que la **rugosidad sensorial**—la interferencia física de parciales dentro de nuestras bandas críticas—es un indicador mucho más básico y universal de la similitud textural que las clásicas reglas armónicas de un estilo particular. Tomar este camino no fue un capricho técnico; es, de hecho, la postura teórica que nos permite poner a prueba nuestra hipótesis de sustitución.

Para saber si esto funciona, organizamos la metodología alrededor de tres hipótesis clave. Primero, la **Hipótesis de Sustitución (H1)** plantea que sí existe una forma de medir esta similitud sonora (basada en la rugosidad) que nos arroje acordes candidatos para reemplazar a otros, manteniendo una coherencia que el oído percibe claramente. Segundo, la **Hipótesis de Variedad (H2)** sugiere que si imaginamos el espacio psicoacústico de los acordes, notaríamos que no es un bloque sólido y uniforme; más bien, tiene una estructura interna natural de menor dimensión (una Variedad o \textit{Manifold}). Por último, la **Hipótesis de Validez Ecológica (H3)** afirma que la música que escuchamos en el mundo real (nuestro corpus) no flota al azar en este espacio. Por el contrario, se agrupa en regiones muy puntuales con niveles de rugosidad característicos. Esto confirmaría que nuestro modelo no es solo matemática, sino un mapa bastante fiel de lo que los humanos hacemos en la práctica musical.

A partir de aquí, el diseño metodológico va bajando desde estas ideas generales hacia la matemática pura, luego al modelo psicoacústico, la definición del espacio de medición y, para cerrar, la prueba de fuego: contrastar poblaciones de acordes generadas por computadora con música real.

## 3.1 Conceptualización del Problema y Supuestos Fundamentales
\label{sec:conceptualizacion}

Cualquiera que intente visualizar relaciones armónicas choca con un muro casi de inmediato: la música existe en muchas dimensiones, pero nuestro cerebro (y nuestras pantallas) manejan bien solo dos o tres. Esta sección plantea las bases teóricas y las reglas de juego que definen hasta dónde llega nuestro modelo.

### 3.1.1 Justificación de la Reducción Dimensional
\label{subsec:justificacion_reduccion}

Si tomamos el espacio de acordes en su forma más pura y cruda, nos enfrentamos a un monstruo de alta dimensión (por ejemplo, $\mathbb{R}^{12}$). Frente a esto, nos apoyamos en la **Hipótesis de la Variedad (Manifold Hypothesis)**. Básicamente, creemos que los acordes que realmente usamos y tienen sentido musical no están regados uniformemente en ese hiperespacio. Viven atrapados en una estructura de muchas menos dimensiones, moldeada por la física del sonido y las reglas teóricas. Como ya demostró \citeauthor{tymoczkoGeometryMusicalChords2006} \citep{tymoczkoGeometryMusicalChords2006}, la armonía eficiente suele moverse a lo largo de ``orbifolds'' geométricos compactos.

Reducir dimensiones en este trabajo no es solo un truco visual para que el gráfico quede bonito en el PDF; es una búsqueda activa de esa estructura oculta. Además, tenemos que lidiar con la famosa **Maldición de la Dimensionalidad**. Cuando trabajas en espacios tan grandes, las matemáticas tradicionales (como la distancia euclidiana) pierden su capacidad de diferenciar cosas. Así que, al proyectar nuestros datos a $\mathbb{R}^2$ o $\mathbb{R}^3$, lo que intentamos es rescatar la verdadera forma topológica del sistema, acercando lo que calcula el algoritmo a los mapas mentales que los humanos usamos al escuchar, tal como sugirieron \cite{lerdahlTonalPitchSpace1988} y \cite{krumhanslCognitiveFoundationsMusical1990}.

### 3.1.2 Supuestos del Modelo Psicoacústico
\label{subsec:supuestos_psicoacusticos}

Para que este problema monstruoso se vuelva algo que podamos calcular, tuvimos que establecer algunas simplificaciones clave. Piensa en ellas como las reglas del juego que hacen válido nuestro modelo. 

La primera es aceptar el **Sistema de Afinación 12-TET** (Temperamento Igual de 12 Tonos). Sí, sabemos que la rugosidad en el mundo real es continua, pero discretizarla era obligatorio para que el sistema pudiera hablar fluidamente con archivos MIDI, además de garantizar la simetría al cambiar de tono. 

La segunda es asumir un **Timbre Genérico**. Imaginamos un sonido puro con un decaimiento perfecto de amplitud en sus armónicos ($A_n = 1/n$). Esto fue intencional: nos permite crear un estándar que aplica a un montón de instrumentos de la música occiodental. Aunque cabe advertir (y se detalla más en los Apéndices) que si cruzáramos esto con instrumentos inarmónicos como campanas, las curvas de rugosidad serían completamente distintas. 

El último gran supuesto es usar la **Rugosidad como Proxy de Disonancia**. Básicamente, apostamos a que, al buscar reemplazar texturas sonoras, este fenómeno físico primitivo de nuestros oídos pesa más que las construcciones culturales sobre qué suena "consonante".

### 3.1.3 Decisiones Críticas de Diseño
\label{subsec:decisiones_diseno}

A la hora de diseñar el modelo, tomamos dos decisiones que definen totalmente cómo leeremos los resultados luego. 

La primera tiene que ver con la **Normalización**. Decidir si proyectamos los vectores (por ejemplo, al simplex) o si dejamos sus valores crudos absolutos cambia todo el panorama. Las mediciones basadas en la "forma" (Simplex + JSD/Coseno) son excelentes para darse cuenta de la identidad de un acorde y sus inversiones. En cambio, si usamos la magnitud cruda (Identity + Euclidean), el sistema se vuelve un experto en clasificar pura tensión global.

La segunda decisión fue imponer **Determinismo y Reproducibilidad**. Entrenar algoritmos de reducción dimensional (MDS o UMAP) es un poco como tirar los dados; tienen un componente aleatorio. Así que establecimos un protocolo con semillas fijas. De esta forma, si vemos una estructura extraña o interesante en el mapa resultante, sabremos que es una propiedad real de la armonía, y no un espejismo creado porque el algoritmo arrancó distinto ese día \cite{Wang2021}.
## 3.2 Formalización del Objeto Acorde
\label{sec:marco_conceptual}

A diferencia de los enfoques tradicionales de la teoría de conjuntos (\textit{PC-sets}) \cite{AllenForteSTRUCTURE} que reducen un acorde a una idea abstracta sin importar su registro, nuestro trabajo parte de un **axioma fundacional** distinto. Sostenemos que la similitud sonora, esa que realmente importa cuando queremos sustituir un acorde por otro, es completamente sensible al registro absoluto. Por eso, nuestro objeto de estudio no será una abstracción teórica, sino un grupo de notas con frecuencias físicas reales, al que llamaremos \textbf{Pitch-Chord}.

Anclar nuestra teoría en el modelo psicoacústico de rugosidad nos obliga a tomar este camino. A nivel compositivo, esto significa reconocer lo evidente: un Do mayor tocado en el bajo profundo (-E_2-G_2$) y uno tocado en el registro medio (-E_4-G_4$) no son la misma entidad sonora dentro de nuestro espacio $\mathcal{A}$. Sus niveles de rugosidad y tensión difieren radicalmente porque las bandas críticas de nuestro oído no responden de forma lineal.

### 3.2.1 Espacio de Notas y Sistema de Referencia

La unidad básica de nuestro modelo es la nota musical, vista como un evento sonoro particular. Elegimos el estándar MIDI bajo la afinación 12-TET como nuestro mapa de coordenadas. No lo hicimos por pereza teórica, sino porque necesitamos preservar la simetría al transportar acordes.

\begin{definition}[Conjunto de Notas y Frecuencia]
Definimos el conjunto de todas las notas posibles como $\mathcal{N} = \{n \in \mathbb{N}_0 : 0 \leq n \leq 127\}$. A cada una de estas notas $ le asignamos matemáticamente su frecuencia fundamental acústica mediante la fórmula (n) = 440 \cdot 2^{(n-69)/12}$.
\end{definition}

Incluso si a veces nos apoyamos en el grupo cíclico $\mathbb{Z}_{12}$ para cálculos menores, todo el peso del modelo de rugosidad recae directamente sobre el dominio de frecuencias (\mathcal{N})$.

### 3.2.2 Formalización del Acorde (Pitch-Chord)

Queremos que la estructura del acorde refleje exactamente cómo están dispuestas las voces (*voicing*).

\begin{definition}[Acorde]
Un acorde compuesto por $ notas es una secuencia ordenada $\mathbf{n} = (n_1, n_2, \dots, n_m) \in \mathcal{N}^m$ que debe cumplir obligatoriamente una regla de orden estricto de menor a mayor:
\begin{equation}
n_1 < n_2 < \dots < n_m
\end{equation}
\end{definition}

Imponer este orden estricto ($<$) trae dos ventajas inmediatas. Primero, **excluye automáticamente los unísonos** ( = n_{i+1}$). Físicamente, tiene sentido: un unísono perfecto no produce rugosidad alguna, así que no añade "textura" a la disonancia. Segundo, nos entrega una **forma canónica única** para cada acorde. Al eliminar el desorden, forzamos al sistema a ver el acorde como un objeto acústico tangible, y no como una simple categoría teórica.

### 3.2.3 Descriptores de Estructura

Necesitamos medir cómo están construidos estos acordes por dentro, capturando tanto sus distancias locales como su espectro cromático total.

\begin{definition}[Vector de Estructura Interválica]
El vector $\mathbf{ic}(\mathbf{n}) \in \mathbb{N}_0^{12}$ lleva la cuenta exacta de cuántos pares de notas están separados por $ semitonos. Ojo aquí: usamos deliberadamente un vector de **12 dimensiones** (no las clásicas 6 clases de intervalos de Forte).
\end{definition}

**Consecuencia Musical:** Al retener las 12 dimensiones, el modelo sabe distinguir un intervalo de su inversión (por ejemplo, una tercera menor de una sexta mayor). Esto es compositivamente vital. Una tercera menor muy junta y una sexta mayor muy abrierta pueden ser funcionalmente lo mismo en el papel, pero en el aire tienen colores de tensión y estabilidades totalmente distintas. Gracias a esto, el modelo entiende hacia dónde "apunta" la tensión.

### 3.2.4 El Espacio Total de Acordes

Llamaremos $\mathcal{A}$ a nuestro terreno de búsqueda general: la colección de todos los acordes matemáticamente posibles (sobre todo tríadas y cuatríadas) dentro de los registros que nos interesan. Más adelante, en la fase experimental, no exploraremos todo este infinito; tomaremos rincones específicos de $\mathcal{A}$ para ver qué hace el modelo cuando le imponemos reglas estilísticas estrictas (ver Sección \ref{sec:diseno_experimental}).

## 3.3 Modelo Psicoacústico de Rugosidad
\label{sec:modelo_psicoacustico}

No nos entusiasma ver el acorde solo como un juego combinatorio; nos interesa porque suena, porque tiene textura y color. Por eso, metimos en el motor un modelo físico de \textit{disonancia sensorial} o **rugosidad**. Todo descansa en una apuesta audaz: si dos acordes generan perfiles de rugosidad parecidos, nuestra percepción tenderá a juzgarlos como funcionalmente similares.

### 3.3.1 Fundamentos Fisiológicos y el Modelo Plomp-Levelt

La biología de este asunto ocurre en la cóclea, específicamente jugando con el concepto de \textit{banda crítica} (Critical Bandwidth). Simplificando la obra de la naturaleza, cuando dos frecuencias puras entran al oído, la membrana basilar intenta filtrarlas. Si están bien separadas, el cerebro las procesa limpiamente como dos notas. Pero, ¿y si están casi pegadas? Las zonas donde resuenan se solapan, enviando un revoltijo de señales a las neuronas.

\citeA{Plomp1965} probaron en el laboratorio que esa "interferencia" eléctrica la sentimos físicamente como un sonido áspero: la rugosidad. Qué tan áspero suena depende de cuánta diferencia de frecuencia ($\Delta f = |f_1 - f_2|$) haya respecto al tamaño de la banda crítica ($) en esa zona específica del oído. Y aquí toca marcar un límite clarísimo: esta **disonancia sensorial** es un accidente anatómico inevitable, mientras que la \textit{disonancia musical} que te enseñan en el conservatorio es, en buena medida, una convención cultural.

### 3.3.2 Formalización de la Disonancia (Modelo Sethares)

Para pasar este fenómeno biológico al código informático, nos basamos en el esquema matemático de \cite{setharesLocalConsonanceRelationship1993}. Su trabajo dibujó curvas que imitan a la perfección los datos de Plomp y Levelt, dejándonos calcular la disonancia $ entre dos sinusoides con frecuencias , f_b$ y amplitudes , A_b$.

\begin{equation}
\label{eq:sethares_pair}
d(f_a, f_b, A_a, A_b) = A_a A_b \left( C_1 e^{-A_1 S \Delta f} + C_2 e^{-A_2 S \Delta f} \right)
\end{equation}

En esta ecuación, $\Delta f = |f_a - f_b|$, y esa variable $ se encarga de acoplar la diferencia a cómo se comporta la banda crítica ahí mismo en el canal auditivo. Esos parámetros (, C, S$) moldean las subidas y bajadas de la gráfica de rugosidad, simulando maravillosamente cómo las bandas críticas son anchas en el registro bajo y se van estrechando a medida que subimos a los agudos.

### 3.3.3 Extensión a Tonos Complejos

La música real no está hecha de sinusoides de laboratorio. Las notas de los instrumentos son tonos complejos, construidos por una frecuencia fundamental y una larga cola de parciales. Esto complica las cosas: la rugosidad de un intervalo no ocurre solo porque chocan las dos fundamentales. Es, literalmente, el choque múltiple de \textit{todos} los armónicos de una nota contra \textit{todos} los armónicos de la otra.

A modo de simplificación, operamos con un **espectro armónico idealizado**. Lo imaginamos como el perfil de un violín o un clarinete clásico, donde la amplitud de cada armónico va cayendo suavemente. Si tenemos una nota con frecuencia base $, trazamos su espectro como $ parciales $\{(p \cdot f_0, \delta^{p-1})\}_{p=1}^H$.

\begin{definition}[Rugosidad de Intervalo Complejo]
Medir la rugosidad total $ entre dos notas (fundamentales , f_j$) implica sumar todas las fricciones parciales individuales, usando la siguiente relación:
\begin{equation}
\label{eq:complex_roughness}
R(f_i, f_j) = \sum_{p=1}^{H} \sum_{q=1}^{H} d(p f_i, q f_j, \delta^{p-1}, \delta^{q-1})
\end{equation}
\end{definition}

Como estándar, elegimos procesar =6$ armónicos y dejar caer la amplitud a una tasa $\delta=0.88$. ¿Por qué 6 y no 20? Es un balance necesario entre precisión y velocidad de la CPU. De todas formas, la literatura muestra que esos primeros seis armónicos albergan casi toda la energía que importa para sentir la disonancia de un intervalo normal.

### 3.3.4 Vectorización: La Firma de Rugosidad

Ahora, si queremos retratar por completo a un acorde $\mathbf{n} = (n_1, \dots, n_m)$, nos dimos cuenta de que resumirlo a un simple número de "rugosidad total" era contraproducente; perdíamos todo el mapa interno de fricciones. En cambio, diseñamos un vector que desglosa cuánta rugosidad exactamante aporta cada tipo de intervalo dentro de la estructura.

\begin{definition}[Vector de Características de Rugosidad $\Phi_{\text{raw}}$]
Construimos un mapa $\Phi_{\text{raw}}: \mathcal{A} \to \mathbb{R}_{\geq 0}^{12}$ que en su celda $ acumula todas las chispas (rugosidad) producidas exclusivamente por aquellos pares de notas que estén a $ semitonos de distancia:
\begin{equation}
\Phi_{\text{raw}, k}(\mathbf{n}) = \sum_{\substack{1 \leq i < j \leq m \\ (n_j - n_i) \equiv k \pmod{12}}} R(f(n_i), f(n_j))
\end{equation}
\end{definition}

Nos gusta pensar en este vector $\Phi_{\text{raw}}$ como la "huella digital psicoacústica" del acorde. Fíjate en la diferencia: el vector clásico ($\mathbf{ic}$) solo cuenta que hay "dos" terceras mayores; $\Phi_{\text{raw}}$ pesa cuánta molestia física generan en verdad esas terceras. De este modo, una tercera mayor tocada con las manos juntas en el bajo profundo de un piano (que vibra de forma horrible) pesará monumentalmente en el bin =4$, comparada con una cristalina tercera mayor en las octavas altas. Esta es la matemática capturando, por fin, el color del timbre.

### 3.3.5 Validación Interna del Modelo

No podíamos seguir avanzando sin verificar que nuestra programación de Sethares estuviera dando números que tuvieran sentido. Sometimos el código a pruebas teóricas clásicas: creamos díadas y movimos una frecuencia poco a poco, tal como se hace en el laboratorio.

El programa respondió de forma impecable. Vimos claramente el famoso "pico de disonancia" asomándose justo cuando las frecuencias estaban a un cuarto de ancho de banda crítico (.25 CB$). Luego, subiendo el nivel de prueba a díadas de 6 armónicos, la máquina recreó ella sola la antigua jerarquía de consonancias occidentales: la octava marcaba casi cero rugosidad, le seguía la quinta justa, luego las terceras y segundas mayores, y allá en el extremo de dolor auditivo, la segunda menor. Lo mejor fue ver cómo confirmó nuestra intuición sobre el registro, castigando duramente a las terceras mayores graves y perdonando a las agudas. El motor estaba listo.
## 3.2 Formalización del Objeto Acorde
\label{sec:marco_conceptual}

A diferencia de los enfoques tradicionales de la teoría de conjuntos (\textit{PC-sets}) \cite{AllenForteSTRUCTURE} que reducen un acorde a una idea abstracta sin importar su registro, nuestro trabajo parte de un **axioma fundacional** distinto. Sostenemos que la similitud sonora, esa que realmente importa cuando queremos sustituir un acorde por otro, es completamente sensible al registro absoluto. Por eso, nuestro objeto de estudio no será una abstracción teórica, sino un grupo de notas con frecuencias físicas reales, al que llamaremos \textbf{Pitch-Chord}.

Anclar nuestra teoría en el modelo psicoacústico de rugosidad nos obliga a tomar este camino. A nivel compositivo, esto significa reconocer lo evidente: un Do mayor tocado en el bajo profundo (-E_2-G_2$) y uno tocado en el registro medio (-E_4-G_4$) no son la misma entidad sonora dentro de nuestro espacio $\mathcal{A}$. Sus niveles de rugosidad y tensión difieren radicalmente porque las bandas críticas de nuestro oído no responden de forma lineal.

### 3.2.1 Espacio de Notas y Sistema de Referencia

La unidad básica de nuestro modelo es la nota musical, vista como un evento sonoro particular. Elegimos el estándar MIDI bajo la afinación 12-TET como nuestro mapa de coordenadas. No lo hicimos por pereza teórica, sino porque necesitamos preservar la simetría al transportar acordes.

\begin{definition}[Conjunto de Notas y Frecuencia]
Definimos el conjunto de todas las notas posibles como $\mathcal{N} = \{n \in \mathbb{N}_0 : 0 \leq n \leq 127\}$. A cada una de estas notas $ le asignamos matemáticamente su frecuencia fundamental acústica mediante la fórmula (n) = 440 \cdot 2^{(n-69)/12}$.
\end{definition}

Incluso si a veces nos apoyamos en el grupo cíclico $\mathbb{Z}_{12}$ para cálculos menores, todo el peso del modelo de rugosidad recae directamente sobre el dominio de frecuencias (\mathcal{N})$.

### 3.2.2 Formalización del Acorde (Pitch-Chord)

Queremos que la estructura del acorde refleje exactamente cómo están dispuestas las voces (*voicing*).

\begin{definition}[Acorde]
Un acorde compuesto por $ notas es una secuencia ordenada $\mathbf{n} = (n_1, n_2, \dots, n_m) \in \mathcal{N}^m$ que debe cumplir obligatoriamente una regla de orden estricto de menor a mayor:
\begin{equation}
n_1 < n_2 < \dots < n_m
\end{equation}
\end{definition}

Imponer este orden estricto ($<$) trae dos ventajas inmediatas. Primero, **excluye automáticamente los unísonos** ( = n_{i+1}$). Físicamente, tiene sentido: un unísono perfecto no produce rugosidad alguna, así que no añade "textura" a la disonancia. Segundo, nos entrega una **forma canónica única** para cada acorde. Al eliminar el desorden, forzamos al sistema a ver el acorde como un objeto acústico tangible, y no como una simple categoría teórica.

### 3.2.3 Descriptores de Estructura

Necesitamos medir cómo están construidos estos acordes por dentro, capturando tanto sus distancias locales como su espectro cromático total.

\begin{definition}[Vector de Estructura Interválica]
El vector $\mathbf{ic}(\mathbf{n}) \in \mathbb{N}_0^{12}$ lleva la cuenta exacta de cuántos pares de notas están separados por $ semitonos. Ojo aquí: usamos deliberadamente un vector de **12 dimensiones** (no las clásicas 6 clases de intervalos de Forte).
\end{definition}

**Consecuencia Musical:** Al retener las 12 dimensiones, el modelo sabe distinguir un intervalo de su inversión (por ejemplo, una tercera menor de una sexta mayor). Esto es compositivamente vital. Una tercera menor muy junta y una sexta mayor muy abrierta pueden ser funcionalmente lo mismo en el papel, pero en el aire tienen colores de tensión y estabilidades totalmente distintas. Gracias a esto, el modelo entiende hacia dónde "apunta" la tensión.

### 3.2.4 El Espacio Total de Acordes

Llamaremos $\mathcal{A}$ a nuestro terreno de búsqueda general: la colección de todos los acordes matemáticamente posibles (sobre todo tríadas y cuatríadas) dentro de los registros que nos interesan. Más adelante, en la fase experimental, no exploraremos todo este infinito; tomaremos rincones específicos de $\mathcal{A}$ para ver qué hace el modelo cuando le imponemos reglas estilísticas estrictas (ver Sección \ref{sec:diseno_experimental}).

## 3.3 Modelo Psicoacústico de Rugosidad
\label{sec:modelo_psicoacustico}

No nos entusiasma ver el acorde solo como un juego combinatorio; nos interesa porque suena, porque tiene textura y color. Por eso, metimos en el motor un modelo físico de \textit{disonancia sensorial} o **rugosidad**. Todo descansa en una apuesta audaz: si dos acordes generan perfiles de rugosidad parecidos, nuestra percepción tenderá a juzgarlos como funcionalmente similares.

### 3.3.1 Fundamentos Fisiológicos y el Modelo Plomp-Levelt

La biología de este asunto ocurre en la cóclea, específicamente jugando con el concepto de \textit{banda crítica} (Critical Bandwidth). Simplificando la obra de la naturaleza, cuando dos frecuencias puras entran al oído, la membrana basilar intenta filtrarlas. Si están bien separadas, el cerebro las procesa limpiamente como dos notas. Pero, ¿y si están casi pegadas? Las zonas donde resuenan se solapan, enviando un revoltijo de señales a las neuronas.

\citeA{Plomp1965} probaron en el laboratorio que esa "interferencia" eléctrica la sentimos físicamente como un sonido áspero: la rugosidad. Qué tan áspero suena depende de cuánta diferencia de frecuencia ($\Delta f = |f_1 - f_2|$) haya respecto al tamaño de la banda crítica ($) en esa zona específica del oído. Y aquí toca marcar un límite clarísimo: esta **disonancia sensorial** es un accidente anatómico inevitable, mientras que la \textit{disonancia musical} que te enseñan en el conservatorio es, en buena medida, una convención cultural.

### 3.3.2 Formalización de la Disonancia (Modelo Sethares)

Para pasar este fenómeno biológico al código informático, nos basamos en el esquema matemático de \cite{setharesLocalConsonanceRelationship1993}. Su trabajo dibujó curvas que imitan a la perfección los datos de Plomp y Levelt, dejándonos calcular la disonancia $ entre dos sinusoides con frecuencias , f_b$ y amplitudes , A_b$.

\begin{equation}
\label{eq:sethares_pair}
d(f_a, f_b, A_a, A_b) = A_a A_b \left( C_1 e^{-A_1 S \Delta f} + C_2 e^{-A_2 S \Delta f} \right)
\end{equation}

En esta ecuación, $\Delta f = |f_a - f_b|$, y esa variable $ se encarga de acoplar la diferencia a cómo se comporta la banda crítica ahí mismo en el canal auditivo. Esos parámetros (, C, S$) moldean las subidas y bajadas de la gráfica de rugosidad, simulando maravillosamente cómo las bandas críticas son anchas en el registro bajo y se van estrechando a medida que subimos a los agudos.

### 3.3.3 Extensión a Tonos Complejos

La música real no está hecha de sinusoides de laboratorio. Las notas de los instrumentos son tonos complejos, construidos por una frecuencia fundamental y una larga cola de parciales. Esto complica las cosas: la rugosidad de un intervalo no ocurre solo porque chocan las dos fundamentales. Es, literalmente, el choque múltiple de \textit{todos} los armónicos de una nota contra \textit{todos} los armónicos de la otra.

A modo de simplificación, operamos con un **espectro armónico idealizado**. Lo imaginamos como el perfil de un violín o un clarinete clásico, donde la amplitud de cada armónico va cayendo suavemente. Si tenemos una nota con frecuencia base $, trazamos su espectro como $ parciales $\{(p \cdot f_0, \delta^{p-1})\}_{p=1}^H$.

\begin{definition}[Rugosidad de Intervalo Complejo]
Medir la rugosidad total $ entre dos notas (fundamentales , f_j$) implica sumar todas las fricciones parciales individuales, usando la siguiente relación:
\begin{equation}
\label{eq:complex_roughness}
R(f_i, f_j) = \sum_{p=1}^{H} \sum_{q=1}^{H} d(p f_i, q f_j, \delta^{p-1}, \delta^{q-1})
\end{equation}
\end{definition}

Como estándar, elegimos procesar =6$ armónicos y dejar caer la amplitud a una tasa $\delta=0.88$. ¿Por qué 6 y no 20? Es un balance necesario entre precisión y velocidad de la CPU. De todas formas, la literatura muestra que esos primeros seis armónicos albergan casi toda la energía que importa para sentir la disonancia de un intervalo normal.

### 3.3.4 Vectorización: La Firma de Rugosidad

Ahora, si queremos retratar por completo a un acorde $\mathbf{n} = (n_1, \dots, n_m)$, nos dimos cuenta de que resumirlo a un simple número de "rugosidad total" era contraproducente; perdíamos todo el mapa interno de fricciones. En cambio, diseñamos un vector que desglosa cuánta rugosidad exactamante aporta cada tipo de intervalo dentro de la estructura.

\begin{definition}[Vector de Características de Rugosidad $\Phi_{\text{raw}}$]
Construimos un mapa $\Phi_{\text{raw}}: \mathcal{A} \to \mathbb{R}_{\geq 0}^{12}$ que en su celda $ acumula todas las chispas (rugosidad) producidas exclusivamente por aquellos pares de notas que estén a $ semitonos de distancia:
\begin{equation}
\Phi_{\text{raw}, k}(\mathbf{n}) = \sum_{\substack{1 \leq i < j \leq m \\ (n_j - n_i) \equiv k \pmod{12}}} R(f(n_i), f(n_j))
\end{equation}
\end{definition}

Nos gusta pensar en este vector $\Phi_{\text{raw}}$ como la "huella digital psicoacústica" del acorde. Fíjate en la diferencia: el vector clásico ($\mathbf{ic}$) solo cuenta que hay "dos" terceras mayores; $\Phi_{\text{raw}}$ pesa cuánta molestia física generan en verdad esas terceras. De este modo, una tercera mayor tocada con las manos juntas en el bajo profundo de un piano (que vibra de forma horrible) pesará monumentalmente en el bin =4$, comparada con una cristalina tercera mayor en las octavas altas. Esta es la matemática capturando, por fin, el color del timbre.

### 3.3.5 Validación Interna del Modelo

No podíamos seguir avanzando sin verificar que nuestra programación de Sethares estuviera dando números que tuvieran sentido. Sometimos el código a pruebas teóricas clásicas: creamos díadas y movimos una frecuencia poco a poco, tal como se hace en el laboratorio.

El programa respondió de forma impecable. Vimos claramente el famoso "pico de disonancia" asomándose justo cuando las frecuencias estaban a un cuarto de ancho de banda crítico (.25 CB$). Luego, subiendo el nivel de prueba a díadas de 6 armónicos, la máquina recreó ella sola la antigua jerarquía de consonancias occidentales: la octava marcaba casi cero rugosidad, le seguía la quinta justa, luego las terceras y segundas mayores, y allá en el extremo de dolor auditivo, la segunda menor. Lo mejor fue ver cómo confirmó nuestra intuición sobre el registro, castigando duramente a las terceras mayores graves y perdonando a las agudas. El motor estaba listo.
## 3.2 Formalización del Objeto Acorde
\label{sec:marco_conceptual}

A diferencia de los enfoques tradicionales de la teoría de conjuntos (\textit{PC-sets}) \cite{AllenForteSTRUCTURE} que reducen un acorde a una idea abstracta sin importar su registro, nuestro trabajo parte de un **axioma fundacional** distinto. Sostenemos que la similitud sonora, esa que realmente importa cuando queremos sustituir un acorde por otro, es completamente sensible al registro absoluto. Por eso, nuestro objeto de estudio no será una abstracción teórica, sino un grupo de notas con frecuencias físicas reales, al que llamaremos \textbf{Pitch-Chord}.

Anclar nuestra teoría en el modelo psicoacústico de rugosidad nos obliga a tomar este camino. A nivel compositivo, esto significa reconocer lo evidente: un Do mayor tocado en el bajo profundo (-E_2-G_2$) y uno tocado en el registro medio (-E_4-G_4$) no son la misma entidad sonora dentro de nuestro espacio $\mathcal{A}$. Sus niveles de rugosidad y tensión difieren radicalmente porque las bandas críticas de nuestro oído no responden de forma lineal.

### 3.2.1 Espacio de Notas y Sistema de Referencia

La unidad básica de nuestro modelo es la nota musical, vista como un evento sonoro particular. Elegimos el estándar MIDI bajo la afinación 12-TET como nuestro mapa de coordenadas. No lo hicimos por pereza teórica, sino porque necesitamos preservar la simetría al transportar acordes.

\begin{definition}[Conjunto de Notas y Frecuencia]
Definimos el conjunto de todas las notas posibles como $\mathcal{N} = \{n \in \mathbb{N}_0 : 0 \leq n \leq 127\}$. A cada una de estas notas $ le asignamos matemáticamente su frecuencia fundamental acústica mediante la fórmula (n) = 440 \cdot 2^{(n-69)/12}$.
\end{definition}

Incluso si a veces nos apoyamos en el grupo cíclico $\mathbb{Z}_{12}$ para cálculos menores, todo el peso del modelo de rugosidad recae directamente sobre el dominio de frecuencias (\mathcal{N})$.

### 3.2.2 Formalización del Acorde (Pitch-Chord)

Queremos que la estructura del acorde refleje exactamente cómo están dispuestas las voces (*voicing*).

\begin{definition}[Acorde]
Un acorde compuesto por $ notas es una secuencia ordenada $\mathbf{n} = (n_1, n_2, \dots, n_m) \in \mathcal{N}^m$ que debe cumplir obligatoriamente una regla de orden estricto de menor a mayor:
\begin{equation}
n_1 < n_2 < \dots < n_m
\end{equation}
\end{definition}

Imponer este orden estricto ($<$) trae dos ventajas inmediatas. Primero, **excluye automáticamente los unísonos** ( = n_{i+1}$). Físicamente, tiene sentido: un unísono perfecto no produce rugosidad alguna, así que no añade "textura" a la disonancia. Segundo, nos entrega una **forma canónica única** para cada acorde. Al eliminar el desorden, forzamos al sistema a ver el acorde como un objeto acústico tangible, y no como una simple categoría teórica.

### 3.2.3 Descriptores de Estructura

Necesitamos medir cómo están construidos estos acordes por dentro, capturando tanto sus distancias locales como su espectro cromático total.

\begin{definition}[Vector de Estructura Interválica]
El vector $\mathbf{ic}(\mathbf{n}) \in \mathbb{N}_0^{12}$ lleva la cuenta exacta de cuántos pares de notas están separados por $ semitonos. Ojo aquí: usamos deliberadamente un vector de **12 dimensiones** (no las clásicas 6 clases de intervalos de Forte).
\end{definition}

**Consecuencia Musical:** Al retener las 12 dimensiones, el modelo sabe distinguir un intervalo de su inversión (por ejemplo, una tercera menor de una sexta mayor). Esto es compositivamente vital. Una tercera menor muy junta y una sexta mayor muy abrierta pueden ser funcionalmente lo mismo en el papel, pero en el aire tienen colores de tensión y estabilidades totalmente distintas. Gracias a esto, el modelo entiende hacia dónde "apunta" la tensión.

### 3.2.4 El Espacio Total de Acordes

Llamaremos $\mathcal{A}$ a nuestro terreno de búsqueda general: la colección de todos los acordes matemáticamente posibles (sobre todo tríadas y cuatríadas) dentro de los registros que nos interesan. Más adelante, en la fase experimental, no exploraremos todo este infinito; tomaremos rincones específicos de $\mathcal{A}$ para ver qué hace el modelo cuando le imponemos reglas estilísticas estrictas (ver Sección \ref{sec:diseno_experimental}).

## 3.3 Modelo Psicoacústico de Rugosidad
\label{sec:modelo_psicoacustico}

No nos entusiasma ver el acorde solo como un juego combinatorio; nos interesa porque suena, porque tiene textura y color. Por eso, metimos en el motor un modelo físico de \textit{disonancia sensorial} o **rugosidad**. Todo descansa en una apuesta audaz: si dos acordes generan perfiles de rugosidad parecidos, nuestra percepción tenderá a juzgarlos como funcionalmente similares.

### 3.3.1 Fundamentos Fisiológicos y el Modelo Plomp-Levelt

La biología de este asunto ocurre en la cóclea, específicamente jugando con el concepto de \textit{banda crítica} (Critical Bandwidth). Simplificando la obra de la naturaleza, cuando dos frecuencias puras entran al oído, la membrana basilar intenta filtrarlas. Si están bien separadas, el cerebro las procesa limpiamente como dos notas. Pero, ¿y si están casi pegadas? Las zonas donde resuenan se solapan, enviando un revoltijo de señales a las neuronas.

\citeA{Plomp1965} probaron en el laboratorio que esa "interferencia" eléctrica la sentimos físicamente como un sonido áspero: la rugosidad. Qué tan áspero suena depende de cuánta diferencia de frecuencia ($\Delta f = |f_1 - f_2|$) haya respecto al tamaño de la banda crítica ($) en esa zona específica del oído. Y aquí toca marcar un límite clarísimo: esta **disonancia sensorial** es un accidente anatómico inevitable, mientras que la \textit{disonancia musical} que te enseñan en el conservatorio es, en buena medida, una convención cultural.

### 3.3.2 Formalización de la Disonancia (Modelo Sethares)

Para pasar este fenómeno biológico al código informático, nos basamos en el esquema matemático de \cite{setharesLocalConsonanceRelationship1993}. Su trabajo dibujó curvas que imitan a la perfección los datos de Plomp y Levelt, dejándonos calcular la disonancia $ entre dos sinusoides con frecuencias , f_b$ y amplitudes , A_b$.

\begin{equation}
\label{eq:sethares_pair}
d(f_a, f_b, A_a, A_b) = A_a A_b \left( C_1 e^{-A_1 S \Delta f} + C_2 e^{-A_2 S \Delta f} \right)
\end{equation}

En esta ecuación, $\Delta f = |f_a - f_b|$, y esa variable $ se encarga de acoplar la diferencia a cómo se comporta la banda crítica ahí mismo en el canal auditivo. Esos parámetros (, C, S$) moldean las subidas y bajadas de la gráfica de rugosidad, simulando maravillosamente cómo las bandas críticas son anchas en el registro bajo y se van estrechando a medida que subimos a los agudos.

### 3.3.3 Extensión a Tonos Complejos

La música real no está hecha de sinusoides de laboratorio. Las notas de los instrumentos son tonos complejos, construidos por una frecuencia fundamental y una larga cola de parciales. Esto complica las cosas: la rugosidad de un intervalo no ocurre solo porque chocan las dos fundamentales. Es, literalmente, el choque múltiple de \textit{todos} los armónicos de una nota contra \textit{todos} los armónicos de la otra.

A modo de simplificación, operamos con un **espectro armónico idealizado**. Lo imaginamos como el perfil de un violín o un clarinete clásico, donde la amplitud de cada armónico va cayendo suavemente. Si tenemos una nota con frecuencia base $, trazamos su espectro como $ parciales $\{(p \cdot f_0, \delta^{p-1})\}_{p=1}^H$.

\begin{definition}[Rugosidad de Intervalo Complejo]
Medir la rugosidad total $ entre dos notas (fundamentales , f_j$) implica sumar todas las fricciones parciales individuales, usando la siguiente relación:
\begin{equation}
\label{eq:complex_roughness}
R(f_i, f_j) = \sum_{p=1}^{H} \sum_{q=1}^{H} d(p f_i, q f_j, \delta^{p-1}, \delta^{q-1})
\end{equation}
\end{definition}

Como estándar, elegimos procesar =6$ armónicos y dejar caer la amplitud a una tasa $\delta=0.88$. ¿Por qué 6 y no 20? Es un balance necesario entre precisión y velocidad de la CPU. De todas formas, la literatura muestra que esos primeros seis armónicos albergan casi toda la energía que importa para sentir la disonancia de un intervalo normal.

### 3.3.4 Vectorización: La Firma de Rugosidad

Ahora, si queremos retratar por completo a un acorde $\mathbf{n} = (n_1, \dots, n_m)$, nos dimos cuenta de que resumirlo a un simple número de "rugosidad total" era contraproducente; perdíamos todo el mapa interno de fricciones. En cambio, diseñamos un vector que desglosa cuánta rugosidad exactamante aporta cada tipo de intervalo dentro de la estructura.

\begin{definition}[Vector de Características de Rugosidad $\Phi_{\text{raw}}$]
Construimos un mapa $\Phi_{\text{raw}}: \mathcal{A} \to \mathbb{R}_{\geq 0}^{12}$ que en su celda $ acumula todas las chispas (rugosidad) producidas exclusivamente por aquellos pares de notas que estén a $ semitonos de distancia:
\begin{equation}
\Phi_{\text{raw}, k}(\mathbf{n}) = \sum_{\substack{1 \leq i < j \leq m \\ (n_j - n_i) \equiv k \pmod{12}}} R(f(n_i), f(n_j))
\end{equation}
\end{definition}

Nos gusta pensar en este vector $\Phi_{\text{raw}}$ como la "huella digital psicoacústica" del acorde. Fíjate en la diferencia: el vector clásico ($\mathbf{ic}$) solo cuenta que hay "dos" terceras mayores; $\Phi_{\text{raw}}$ pesa cuánta molestia física generan en verdad esas terceras. De este modo, una tercera mayor tocada con las manos juntas en el bajo profundo de un piano (que vibra de forma horrible) pesará monumentalmente en el bin =4$, comparada con una cristalina tercera mayor en las octavas altas. Esta es la matemática capturando, por fin, el color del timbre.

### 3.3.5 Validación Interna del Modelo

No podíamos seguir avanzando sin verificar que nuestra programación de Sethares estuviera dando números que tuvieran sentido. Sometimos el código a pruebas teóricas clásicas: creamos díadas y movimos una frecuencia poco a poco, tal como se hace en el laboratorio.

El programa respondió de forma impecable. Vimos claramente el famoso "pico de disonancia" asomándose justo cuando las frecuencias estaban a un cuarto de ancho de banda crítico (.25 CB$). Luego, subiendo el nivel de prueba a díadas de 6 armónicos, la máquina recreó ella sola la antigua jerarquía de consonancias occidentales: la octava marcaba casi cero rugosidad, le seguía la quinta justa, luego las terceras y segundas mayores, y allá en el extremo de dolor auditivo, la segunda menor. Lo mejor fue ver cómo confirmó nuestra intuición sobre el registro, castigando duramente a las terceras mayores graves y perdonando a las agudas. El motor estaba listo.
