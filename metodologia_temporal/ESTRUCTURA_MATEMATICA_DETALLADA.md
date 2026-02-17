# Estructura Detallada: Capítulo de Metodología — Versión Rigurosa

**Enfoque:** Modelamiento Matemático Aplicado (Híbrido C+: formalización + operacionalización + validación).
**Audiencia:** Jurado de tesis de maestría en matemáticas aplicadas; lectores con doctorado en matemáticas.
**Principio narrativo:** Cada sección construye sobre la anterior. El lector nunca usa un objeto que no haya sido definido. La motivación precede a la formalización; la formalización precede a la implementación.

---

## Capítulo 3: Metodología y Modelado Matemático

### 3.0 Introducción al capítulo y flujo metodológico

**Propósito:** Orientar al lector sobre la estructura del capítulo, el problema que se resuelve computacionalmente, y la lógica de las decisiones.

**Contenido obligatorio:**
- Diagrama de flujo metodológico general (Figura): Población → Φ(c) → D → Embedding 2D → Evaluación.
- Enumeración de los objetivos específicos y su mapeo a las secciones del capítulo.
- Declaración explícita de la rúbrica de modelamiento como control de calidad: (i) formalización (§3.1–3.5), (ii) evaluación (§3.7), (iii) supuestos y límites (§3.8).
- Definición operativa de "experimento" en el contexto del repositorio (configuración que fija: población, normalización, métrica, reductor, semillas, carpeta de salida).

**Preguntas clave para la escritura:**
- ¿Puede el lector, solo leyendo esta introducción, entender qué va a leer y por qué en ese orden?
- ¿Queda claro que el capítulo NO es un manual de software sino la descripción de un modelo matemático con su realización computacional?

**Debilidades a vigilar:**
- No caer en un resumen ejecutivo vacío; cada frase debe aportar orientación narrativa.
- Evitar prometer más de lo que se entrega (no mencionar sustitución armónica si no se formaliza plenamente en este capítulo).

---

### 3.1 El Acorde como Objeto Matemático

*Motivación: Antes de medir, formalizar, o reducir, debemos definir con precisión el dominio sobre el cual operamos.*

#### 3.1.1 Espacio de Notas y Sistema de Referencia

**Definición 3.1 (Espacio de notas MIDI):** $\mathcal{N} = \{0, 1, \ldots, 127\} \subset \mathbb{N}_0$D.

**Definición 3.2 (Frecuencia fundamental):** La función $f: \mathcal{N} \to \mathbb{R}^+$ definida por $f(n) = 440 \cdot 2^{(n-69)/12}$, que fija la afinación 12-TET con ancla A4 = 440 Hz.

**Definición 3.3 (Clases de altura):** La relación de equivalencia $n \sim m \iff n \equiv m \pmod{12}$ induce el grupo cociente $\mathbb{Z}_{12}$. La proyección canónica $\pi: \mathcal{N} \to \mathbb{Z}_{12}$ asigna $\pi(n) = n \bmod 12$.

**Contenido adicional:**
- Observar que $(\mathbb{Z}_{12}, +)$ es un grupo cíclico, y que los intervalos se definen como diferencias en este grupo.
- Explicitar que el sistema 12-TET es un supuesto del modelo (no una verdad universal): excluye microtonalidad y afinación justa.

**Preguntas clave:**
- ¿Es necesario definir $\mathcal{N}$ como todo $\{0,...,127\}$ o basta con el subrango usado experimentalmente?
- ¿Cómo justificar la elección de 12-TET frente a un lector que podría preguntar por afinaciones alternativas?

<!-- REDACTADO: Q-003 | Fuente: respuestas_notebooklm.json | Workflow: /redactor-critico | Fecha: 2026-02-16 -->
<!-- Fase 1: 30 afirmaciones evaluadas → 6 Nivel A, 4 Nivel B, 20 Nivel C (descartadas: computación/dimred detallada, irrelevante para §3.1.1) -->
<!-- Fase 2: Claim central = "dominio teórico completo, restricción experimental como supuesto metodológico" -->
<!-- Fase 3: 3 argumentos convergentes (perceptual ×3 fuentes, computacional ×2 fuentes, práctica estándar ×2 fuentes) -->

**Resolución (Q-003):** La Definición 3.1 establece el dominio teórico sobre el rango MIDI completo $\mathcal{N} = \{0, \ldots, 127\}$ para mantener la generalidad del modelo; sin embargo, la experimentación numérica opera sobre subrangos restringidos. Esta decisión se sustenta en tres argumentos convergentes. En primer lugar, la validez perceptual: dado que la rugosidad de Sethares depende de frecuencias absolutas, la curva de disonancia para un intervalo dado se escala con el registro ---formalmente, $D(x) \neq D(x+k)$ para traslaciones $k$ en semitonos \cite{Sethares1993, Cubarsi2019}. Eerola y Lahdelma \cite{Eerola2022} demuestran empíricamente que la consonancia sensorial presenta una relación cúbica con el registro, con un óptimo en el rango C4--C5 (MIDI 60--72) y caída significativa por debajo de C3 (MIDI 48) ---donde el ancho de banda crítico abarca intervalos musicales amplios y la rugosidad satura--- y por encima de C6 (MIDI 84) por predominancia de *sharpness*. En segundo lugar, la tratabilidad computacional: el número de acordes de cardinalidad $m$ en un universo de $N$ notas es $\binom{N}{m}$, alcanzando magnitudes del orden de $10^{26}$ para $N = 88$ \cite{BuongiornoNardelli2020}, lo cual hace inviable el análisis exhaustivo sin acotar el dominio. Finalmente, la restricción a subconjuntos $S \subset \mathbb{Z}^n$ definidos por límites instrumentales o perceptuales es procedimiento estándar en la literatura de generación algorítmica y modelado computacional de acordes \cite{QuickHudak2011, Chew2014}. Se adopta por tanto la estrategia de declarar $\mathcal{N}$ en su rango completo dentro de la formalización y especificar el subrango experimental como supuesto metodológico reproducible (véase §3.8).

<!-- FIN REDACTADO: Q-003 -->

<!-- REDACTADO: Q-004 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-16 -->
<!-- Fase 1: ~45 afirmaciones evaluadas → 5 Nivel A, 3 Nivel B, ~37 Nivel C (descartadas: orbifolds, Railsback, espacios continuos, dimred sin fuentes musicales) -->
<!-- Fase 2: Claim central = "12-TET como supuesto operativo justificado por agnosticismo del modelo, estandarización MIDI y simetría transposicional" -->
<!-- Fase 3: 3 argumentos convergentes (perceptual ×4 fuentes, estandarización ×2 fuentes, simetría ×2 fuentes) -->

**Resolución (Q-004):** La elección de 12-TET como sistema de afinación no es una hipótesis del modelo sino un supuesto operativo, justificable desde tres ángulos. Primero, el modelo de disonancia sensorial de Sethares es *agnóstico al temperamento*: calcula rugosidad a partir de frecuencias absolutas sin asumir escala alguna \cite{Sethares1993}. Los mínimos de su curva de disonancia para timbres armónicos coinciden con las proporciones de afinación justa, y los intervalos de 12-TET caen suficientemente cerca de esos mínimos para ser perceptualmente indistinguibles en la mayoría de los casos. Cubarsí \cite{Cubarsi2019} formaliza esta proximidad demostrando que la curva de disonancia de un tetracordo en 12-TET es "muy cercana" a la de afinación justa, con el error más significativo en la tercera mayor (+13.7 cents), aún dentro del umbral de percepción categórica documentado por Tenney (1983) y confirmado experimentalmente (Moore, Peters y Glasberg, 1985: tolerancia del 1--3\% para armónicos bajos). Segundo, el protocolo MIDI ---estándar de facto para la representación simbólica de música--- codifica alturas como enteros $n \in \{0, \ldots, 127\}$ y asume explícitamente 12-TET en la conversión $f(n) = 440 \cdot 2^{(n-69)/12}$, imponiendo equivalencia enarmónica \cite{Stolzenburg2015}. Tercero, 12-TET garantiza invariancia transposicional: todos los semitonos tienen la misma magnitud frecuencial relativa, lo cual habilita la exploración sistemática de acordes sin sesgo por tonalidad. Tymoczko \cite{Tymoczko2011} denomina "sobredeterminación" al hecho de que los acordes más uniformes en 12-TET resultan ser simultáneamente los más consonantes acústicamente y los más eficientes en conducción de voces. La extensión del modelo a otros temperamentos es conceptualmente directa ---bastaría redefinir $f(n)$--- pero queda fuera del alcance de este trabajo (véase §3.8).

<!-- FIN REDACTADO: Q-004 -->

#### 3.1.2 Formalización del Acorde

**Definición 3.4 (Acorde):** Un acorde de cardinalidad $m$ es una $m$-tupla estrictamente creciente $\mathbf{n} = (n_1, \ldots, n_m) \in \mathcal{N}^m$ con $n_1 < n_2 < \cdots < n_m$.

**Notación:** $\mathcal{A}_m$ denota el conjunto de todos los acordes de cardinalidad $m$; $\mathcal{A} = \bigcup_{m=2}^{M} \mathcal{A}_m$ el espacio total de acordes considerados ($M \leq 127$ en la práctica). (Nota de santimath: estoy leyendo hasta ahora, pero acá me parece que hay que abordar el tamaño del espácio y modelar como crecen los acordes de acuerdo a los parametros y al tipo de representación que se les imponga, no? en el repo y en la GUI tenemos un modo estructural y eso hace que sean menos acordes, y también hay un modo de rango, que hace que sean menos acordes, etc.)

**Definición 3.5 (Vector de intervalos adyacentes):** $\Delta: \mathcal{A}_m \to \mathbb{N}^{m-1}$, donde $\Delta_i(\mathbf{n}) = n_{i+1} - n_i$ para $i = 1, \ldots, m-1$. (Nota de santimath: El vector es una funcion una aplciacion la notacion no es clara o si?)

**Definición 3.6 (Rango del acorde):** $\text{span}(\mathbf{n}) = n_m - n_1$ (en semitonos). 

**Contenido adicional:**
- Observar que la cardinalidad del espacio $|\mathcal{A}_m|$ para un universo de $N$ notas es $\binom{N}{m}$, lo cual crece exponencialmente.
- Tabla: para el rango MIDI completo $N=128$, $|\mathcal{A}_2| = 8128$, $|\mathcal{A}_3| = 341376$, $|\mathcal{A}_4| \approx 10.7\text{M}$. (Nota de santimath: defintivamente hay que habalr cerca a estos topicos sobre el modo estructural y que tipo de objeto seria ahí o no sé si va en las asumciones del modelo o dónde.)

**Preguntas clave:**
- ¿Se justifica exigir $n_1 < \cdots < n_m$ (estrictamente creciente) o podrían admitirse unísonos? (Not de santimath: exclente pregunta esto nos difenrecia de muchos otros enfoques de representacion porque para nosotros no estamos permtiendo unisonos asumimos rugosidad nula hay que njustificarlo).
    
    <!-- REDACTADO: Q-005 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-16 -->
    > **Resolución (Q-005):** La restricción de monotonía estricta $n_1 < \cdots < n_m$ excluye por diseño las notas MIDI duplicadas (unísonos) y se justifica desde tres perspectivas. Formalmente, en el modelo de Sethares la disonancia sensorial entre dos sinusoides se anula cuando su diferencia frecuencial es cero: $d(\Delta f = 0) = 0$, siendo el unísono el mínimo global de la curva de disonancia para cualquier timbre \cite{Sethares1993}. Dado que el feature principal del modelo ---la rugosidad por clase de intervalo $\Phi_{\text{raw}}$--- se construye sumando contribuciones $d_{ij}$ sobre pares de parciales, un par de notas idénticas aporta $d=0$ a todos los términos: la nota duplicada es invisible para el extractor de características. Perceptualmente, Harrison y Pearce \cite{Harrison2020} justifican la no codificación de alturas duplicadas argumentando que "las alturas duplicadas tienden a fusionarse perceptivamente en la mente del oyente", reduciendo el acorde percibido a su conjunto subyacente de alturas distintas. Computacionalmente, la representación por vectores de croma ---estándar en MIR--- colapsa multiplicidades a activaciones binarias \cite{Muller2015, Bernardes2016}, tratando los acordes como conjuntos (no multiconjuntos). En consecuencia, el espacio $\mathcal{A}_m$ opera sobre combinaciones sin repetición, con cardinalidad $|\mathcal{A}_m| = \binom{N}{m}$. El modelado como multiconjuntos sería necesario si el objetivo fuera voice leading polifónico \cite{Callender2008}, pero excede el alcance del presente trabajo orientado a similitud sonora.
    <!-- FIN REDACTADO: Q-005 -->

- ¿La restricción de ordenamiento implica que dos voicings con las mismas notas en distinto orden son el mismo acorde? (Nota de santimath: estamos usando la palabra voivings en este capitlo? hay que definirlo al lector matemático, y no, dos acordes con misma notas en otrdens disntios son distintos, la idea del orden es que estan ordenados porfrecuencias cmo cuando uno toca en el piano desde el bajo hacia las voces agudas del acorde mover una nota a otra octava hace que sea disitintto el acorde no se si esa justificaion ira aca pero hay que buscar referencias y uivar esta claidad en el lugar adecuado para el lector.)

    <!-- REDACTADO: Q-006 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-16 -->
    > **Resolución (Q-006):** La definición de identidad en ChordSpace se basa estricta y únicamente en el conjunto de alturas absolutas, no en su orden de enumeración ni en la asignación a voces específicas. La tupla ordenada es simplemente la **forma canónica** para representar el multiconjunto de notas en el **dominio fundamental** del espacio cociente $\mathbb{R}^m/S_m$, eliminando la ambigüedad por permutación \cite{Tymoczko2011}. Esto implica que dos voicings con las mismas notas exactas en distinto orden (e.g., $(60, 64, 67)$ vs $(67, 60, 64)$ en sentido de lista de entrada) son matemáticamente idénticos y poseen rugosidad idéntica, dado que la función de costo de Sethares es una sumatoria sobre todos los pares de parciales, operación conmutativa e independiente del índice \cite{Sethares1993, PlompLevelt1965}. Sin embargo, esta invariancia de permutación no debe confundirse con la **invariancia de inversión**: cambiar una nota de octava (e.g., $60 \to 72$) altera el conjunto de valores y, por tanto, crea un objeto distinto con propiedades psicoacústicas diferentes, lo cual es vital para capturar la sensibilidad al registro \cite{Harrison2020}.
    <!-- FIN REDACTADO: Q-006 -->




#### 3.1.3 Decisión de Identidad: Distinción frente a la Teoría de PC-sets

**Proposición 3.1 (No equivalencia con PC-sets):** El espacio $\mathcal{A}$ NO satisface los axiomas de equivalencia de la teoría de conjuntos de clases de altura (Forte, 1973). Específicamente:
- (a) Dos tuplas con las mismas clases de altura pero diferente registro (e.g., $(48, 52, 55)$ vs $(60, 64, 67)$) son objetos DISTINTOS en $\mathcal{A}$.
- (b) Las inversiones $T_nI$ de la teoría de Forte NO se aplican. ( Nota de santimath: hay que estar muy pndeinte con la ntoacion y las deficniones e de estos simbolos decidir si van en este capitulo o en otro recordando que el lector es matemático decidir si se hace una nota acalratoria de lo que es el coentpo o la fucnioin o quiza en el marto teorio ya se haya definido no se.)
- (c) El intervalo $k$ y su complementario $12-k$ se mantienen como clases distintas en la representación. ( Nota de santimath: Ya definimos complementario? asumimos que el lector comprende que se refiere al complemento sobre el grupo ciclico $\mathbb{Z}_{12}$?  quizahaya que decir algo mas para la claridad del lector.)

**Justificación:** Esta decisión es fundamental porque la rugosidad de Sethares depende de las frecuencias absolutas (no solo de las clases de altura), y la interacción armónica entre parciales cambia con el registro.

**Preguntas clave:**
- ¿Cómo responder a un evaluador que objete la falta de invariancia por transposición?

    <!-- REDACTADO: Q-007 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-16 -->
    > **Resolución (Q-007):** El rechazo de la invariancia por transposición no es una omisión sino una decisión de diseño fundamentada en tres líneas de evidencia. Primero, desde la psicoacústica: la disonancia sensorial depende de las frecuencias absolutas a través del ancho de banda crítico coclear (ERB), el cual se ensancha en cents conforme desciende el registro. Eerola y Lahdelma \cite{Eerola2022} demuestran experimentalmente que la consonancia percibida sigue una relación cubica con el registro: acordes con fundamentales bajo 130 Hz (C3) reciben calificaciones de consonancia significativamente menores que sus transposiciones en registros medios, porque el ERB en la segunda octava abarca aproximadamente 646 cents ---mas de un tritono--- frente a 216 cents en la sexta octava \cite{Rogala2017}. Asi, una tercera mayor en C2 cae enteramente dentro de la banda critica, generando rugosidad maxima que desaparece al transportarla a C5. Segundo, desde la teoria geometrica: el marco OPTIC de Callender, Quinn y Tymoczko \cite{Callender2008} formaliza las equivalencias musicales como operaciones modulares e independientes; adoptar P-equivalencia (permutacion) sin T-equivalencia (transposicion) es legitimo y produce el espacio $\mathbb{T}^n / S_n$, apropiado para modelar instancias especificas de acordes en lugar de tipos abstractos. Tercero, desde la cognicion computacional: Harrison y Pearce \cite{Harrison2020} argumentan que las representaciones de bajo nivel (pitch chord) deben preceder a las abstracciones categoricas (pitch-class set), permitiendo que las consideraciones psicoacusticas ---rugosidad, armonicidad--- operen sobre frecuencias reales antes de cualquier colapso por transposicion. Milne et al. \cite{Milne2023} confirman empiricamente que las clasificaciones $T_n$ oscurecen patrones perceptivos: la estabilidad evaluada para un mismo tipo de acorde varia sustancialmente segun su disposicion y registro.
    <!-- FIN REDACTADO: Q-007 -->

- Respuesta prevista: el modelo captura propiedades perceptuales que SON sensibles al registro; la invariancia se recupera opcionalmente vía poblaciones ancladas. ( Nota de santimath: Yo diría que no nos interesa ese enfoque de invariancia por transposición, pero hay que justificarlo bien.) 

**Debilidades a vigilar:**
- No basta con decir "no usamos PC-sets"; se debe explicar positivamente QUÉ noción de identidad SÍ se usa y POR QUÉ es apropiada para los objetivos del trabajo. ( Nota de santimath: trabajo para el notebooklm) 

    <!-- REDACTADO: Q-008 | Fuente: 4 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-16 -->
    > **Resolución (Q-008):** La entidad fundamental de ChordSpace es un **pitch chord** ---termino acunado por Harrison y Pearce \cite{Harrison2020} para designar un conjunto finito de numeros de nota MIDI que retiene la informacion de registro absoluto. A diferencia de los pitch-class sets, esta representacion no asume equivalencia de octava ni invariancia trasposicional. Geometricamente, el pitch chord se formaliza como un punto en el **dominio fundamental** del espacio P-equivalente $\mathbb{R}^n / S_n$, definido por la restriccion $x_1 < x_2 < \dots < x_n$ \cite{Callender2008, Tymoczko2011}. No debe confundirse con el espacio OP de Callender et al., el cual impone ademas equivalencia de octava ($\mathbb{T}^n / S_n$); ChordSpace opera en el espacio de alturas lineal $\mathbb{R}^n$, no en el toro. Esta granularidad es necesaria porque el input de la funcion de rugosidad de Sethares \cite{Sethares1993} requiere frecuencias absolutas: un pc-set no puede generar el espectro de parciales necesario para calcular $d(f_a, f_b)$. La evidencia empirica confirma esta necesidad: Eerola y Lahdelma \cite{Eerola2022} demuestran que la consonancia percibida varia significativamente con el registro a traves de la rugosidad (en graves) y la agudeza (en agudos), informacion que los pc-sets descartan inherentemente.
    <!-- FIN REDACTADO: Q-008 -->

#### 3.1.4 Vector de Conteo de Clases de Intervalo (12 bins)

**Definición 3.7 (Conteo de clases de intervalo):** Para un acorde $\mathbf{n} \in \mathcal{A}_m$, el vector $\mathbf{ic}(\mathbf{n}) \in \mathbb{N}_0^{12}$ se define por:
$$\mathbf{ic}_k(\mathbf{n}) = \#\{(i,j) : 1 \leq i < j \leq m,\ (n_j - n_i) \bmod 12 = k'\}$$
donde $k' = ((n_j - n_i) \bmod 12 - 1) \bmod 12$ con la convención de interfaz: intervalo $0$ (unísono/octava) $\to$ índice 11; intervalos $1, \ldots, 11 \to$ índices $0, \ldots, 10$. ( Nota de santimath: yo creo que hay que poner un ejemplo para el lector. y definitivamente hay que explicarlo mejor.) 

**Propiedad 3.1:** $\sum_{k=0}^{11} \mathbf{ic}_k(\mathbf{n}) = \binom{m}{2}$ (total de pares internos).

**Propiedad 3.2 (Distinción de complementarios):** A diferencia del vector IC de Forte (6 componentes, colapsando $k$ y $12-k$), este vector de 12 componentes preserva la distinción entre intervalos complementarios. Justificación: la rugosidad de un par a distancia de 3 semitonos no es idéntica a la de un par a 9 semitonos debido a que las frecuencias absolutas y la interacción de armónicos difieren.(Nota de santimath: y tambien lo muestra el modelo de seathaeres y su graficasu curva de percicion de rugosidad en los intervalos, tambien hay que decir que hay trabajos que si represnetan con 6 dimencioens a los acordes no estamos tan de aceurdo anivel percepcion habra que ver como podemos probar que destacamos en algo en ese aspecto respecto a los trabajos como el TIS)

**Preguntas clave:**
- ¿Por qué 12 bins y no 6 (como en Forte) o 13 (incluyendo unísono separado de octava)? ( Nota de santimath, excelente pregunta. Notebooklm deberia aydaurnos a repsonder con refernecias acertadas.)

    <!-- REDACTADO: Q-010 | Fuente: 4 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-16 -->
    > **Resolución (Q-010):** La eleccion de 12 bins responde a la **asimetria acustica de la rugosidad**: un intervalo de $k$ semitonos y su complementario $12-k$ no producen la misma disonancia sensorial. Los modelos de Plomp-Levelt y Sethares \cite{Sethares1993} calculan la rugosidad sumando las interferencias entre pares de parciales; al invertir un intervalo cambian las distancias frecuenciales absolutas entre parciales, alterando cuantos caen dentro de la banda critica. Datos empiricos compilados por Huron \cite{Huron1994} confirman esta asimetria: la tercera menor ($k=3$) obtiene valores de consonancia (Hutchinson-Knopoff) de 0.1109, mientras que la sexta mayor ($k=9$) obtiene 0.0477. El vector IC de Forte (6 bins) colapsa esta distincion porque impone implicita e intrinsecamente la I-equivalencia del marco OPTIC \cite{Forte1973, Callender2008}, equiparando intervalos que son acusticamente distintos. Rechazar este colapso equivale algebraicamente a operar en un espacio OPT (sin I), lo que preserva la direccionalidad intervalica y el poder discriminativo del vector de caracteristicas. Se descartan 13 bins porque la equivalencia de octava ---propiedad perceptual fundamental--- ya esta asumida al trabajar con clases de intervalo modulo 12: el unisono y la octava generan la misma posicion cromática, aunque el vector $\Phi_{\text{raw}}$ opera sobre intervalos (no sobre notas individuales), y la distincion octava/unisono ya esta capturada por la sensibilidad al registro del pitch chord. Notablemente, incluso el TIS de Bernardes et al. \cite{Bernardes2016}, que visualiza 6 dimensiones complejas, parte de un vector de croma de 12 elementos y reconoce explicitamente que su medida de consonancia "contradice una limitacion de los modelos de disonancia sensorial".
    <!-- FIN REDACTADO: Q-010 -->

- ¿Cuál es la implicación algebraica de NO colapsar complementarios? (El espacio de representación tiene mayor dimensión pero mayor poder discriminativo).(Nota de santimath: exceltente pregunta quiza es el enfoque que no nos itneresa por ahora para este trabajo de exploracion de acordes pero hay que justificar de alguna manera)

    <!-- REDACTADO: Q-011 | Fuente: 2 notebooks NotebookLM (armonia, math) | Workflow: /redactor-critico | Fecha: 2026-02-16 -->
    > **Resolución (Q-011):** No colapsar $k$ con $12-k$ equivale a rechazar la equivalencia de **Inversión (I)** del marco OPTIC \cite{Callender2008}. El vector IC de Forte opera en el espacio OPTI ("Set Classes"), donde una triada mayor y su inversion (la menor) son la misma clase de conjunto. Al mantener un vector $\Phi_{\text{raw}} \in \mathbb{R}^{12}$ que distingue todos los intervalos dirigidos, operamos en el espacio **OPT** ("Chord Types"), donde mayor y menor son entidades distintas \cite{Tymoczko2011}. Cambouropoulos \cite{Cambouropoulos2016} formaliza esta distincion mediante el vector DIC (Directed Interval Class) de 12 componentes, argumentando que preservar la direccionalidad es esencial para capturar propiedades de conduccion de voces e idiomas armonicos donde la inversion altera la funcion. Tymoczko \cite{Tymoczko2009} refuerza este argumento: los intervalos deben modelarse como vectores dirigidos, no como distancias simetricas, para representar movimientos especificos en el espacio de alturas. La reduccion de Forte ha sido calificada como "drastica" y "problematica" para aplicaciones tonales por Giannos y Cambouropoulos. La ganancia neta es un espacio de mayor dimension ($\mathbb{R}^{12}$ vs $\mathbb{R}^6$) pero con mayor poder discriminativo: se distinguen acordes que Forte agrupa (e.g., los conjuntos Z-relacionados que comparten IC pero difieren en estructura dirigida).
    <!-- FIN REDACTADO: Q-011 -->

**Debilidades a vigilar:**
- La convención de binning (intervalo 0 → índice 11) es una decisión de implementación que debe documentarse cuidadosamente para evitar confusiones en la reproducibilidad.

---

### 3.2 Modelo Psicoacústico de Rugosidad

*Motivación: ¿Qué propiedad medimos en cada acorde y por qué esa propiedad captura "similitud sonora"?* (Nota de santimath: esta pregunta es clave para el lector.) (Nota de santimath: y que tiene que ver con la rugosidad? )     

#### 3.2.1 Fundamentos del Modelo de Plomp-Levelt y Sethares

**Contenido:**
- Breve reseña del resultado experimental de Plomp y Levelt (1965): la disonancia sensorial entre dos tonos puros depende de la separación frecuencial relativa a la banda crítica auditiva.(Nota de santimath: hay que explicar que es la banda critica auditiva. y que es la disonancia sensorial. y que tiene que ver con la rugosidad?  aunque eso debio habalrse a mas profundidad en otros capituslo prvios no? creeria yo.)

    <!-- REDACTADO: Q-012 | Fuente: 2 notebooks NotebookLM (psicoacustica, armonia) | Workflow: /redactor-critico | Fecha: 2026-02-16 -->
    > **Resolución (Q-012):** La **banda critica** (*critical bandwidth*) es el ancho de banda efectivo dentro del cual el sistema auditivo integra la energia sonora; su base fisiologica reside en la coclea, donde la excitacion tonotopica en la membrana basilar impide resolver individualmente dos componentes demasiado cercanos en frecuencia \cite{Sethares1993}. Cuando dos sinusoides caen dentro de la misma banda critica, su interferencia genera fluctuaciones rapidas de amplitud: si estas fluctuaciones son lentas ($\leq 20$ Hz), se perciben como **batimientos**; al acelerarse ($\sim 20$--$300$ Hz), la sensacion se transforma en **rugosidad** (*roughness*), descrita como aspereza auditiva \cite{Vassilakis2001}. La **disonancia sensorial** es la metrica agregada de dicha rugosidad: en el modelo de Sethares, se calcula sumando la rugosidad generada por todos los pares de parciales de un espectro sonoro \cite{Sethares1993}. Asi, aunque frecuentemente se usan como sinonimos en la literatura tecnica, existe una distincion causal: la rugosidad es el fenomeno perceptivo directo de la fluctuacion de la envolvente, mientras que la disonancia sensorial es su acumulacion cuantitativa en un intervalo o acorde complejo. La maxima disonancia entre dos sinusoides ocurre aproximadamente al 25\% del ancho de banda critico. Harrison y Pearce \cite{Harrison2020} distinguen ademas la disonancia sensorial (componente fisiologica) de la disonancia musical (que incluye armonicidad y familiaridad cultural), enfatizando que ChordSpace modela exclusivamente la primera.
    <!-- FIN REDACTADO: Q-012 -->
- La parametrización de Sethares (1993): extensión a tonos complejos (con parciales armónicos) mediante suma de contribuciones par-a-par entre parciales. ( Nota de santimath: Nosotros estamos tomando tonos sinteticos esto como interfeire o apoya esto, si combinan bien?)
- Referencia al marco teórico para la discusión completa; aquí solo la formalización operativa.

**Preguntas clave:**
- ¿Hasta qué punto la parametrización de Sethares es "el" modelo o "un" modelo posible? (Es "un" modelo; existen alternativas como Vassilakis, Hutchinson-Knopoff).
- ¿Cómo se justifica la elección de Sethares sobre otras parametrizaciones? (Nota de santimath: esta pregunta es clave para el lector.)

    <!-- REDACTADO: Q-013 | Fuente: 3 notebooks NotebookLM (armonia, psicoacustica, computacion) | Workflow: /redactor-critico | Fecha: 2026-02-16 -->
    > **Resolución (Q-013):** La parametrizacion de Sethares (1993) es **"un" modelo** entre varios, pero se lo describe como el "mas extendido" para calcular la disonancia sensorial de espectros arbitrarios \cite{Sethares1993}. Los principales modelos alternativos son: (i) **Hutchinson-Knopoff** (1978), que pondera 10 armonicos con amplitudes $1/n$ y varia el ancho de banda critico segun el registro, pero asume incorrectamente una relacion lineal con el SPL y predice mal la jerarquia de triadas \cite{Hutchinson1978}; (ii) **Vassilakis** (2001), que refina el tratamiento de la fluctuacion de amplitud pero ha sido criticado por falta de invarianza ante la combinacion de parciales \cite{Vassilakis2001}; (iii) **Parncutt** (1989), que introduce harmonicidad/periodicidad (correlacion $r = .675$ con datos empiricos vs $r = .352$ de HK) \cite{Parncutt1989}; y (iv) modelos combinados recientes (Masina \& Lo Presti 2024) que suman rugosidad y compacidad, logrando el mejor ajuste con datos perceptuales de triadas \cite{MasinaLoPresti2024}. La eleccion de Sethares para ChordSpace se justifica porque: (a) proporciona una funcion suave y diferenciable (la formula $d(x) = e^{-ax} - e^{-bx}$ aplicada a todos los pares de parciales), util para algoritmos de optimizacion y navegacion continua en el espacio de acordes; (b) permite la relacion timbre-escala (consonancia local); y (c) es el modelo de referencia base sobre el cual se construyen o comparan los demas \cite{Sethares1993, Harrison2020}. Se reconoce como limitacion que la rugosidad por si sola es insuficiente para predecir el orden de consonancia de triadas; futuras versiones podrian incorporar un componente de armonicidad.
    <!-- FIN REDACTADO: Q-013 -->

    <!-- REDACTADO: Q-014 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-16 -->
    > **Resolución (Q-014):** La eleccion de Sethares sobre las alternativas se justifica en el contexto especifico de ChordSpace por cuatro razones operativas. Primero, la **parsimonia computacional**: frente a Parncutt (template-matching con enmascaramiento), Sethares requiere unicamente evaluar la funcion $d$ sobre pares de parciales, sin sobrecarga cognitiva \cite{Cook2006, Sethares1993}. Segundo, la **sensibilidad al registro**: los parametros $s_1, s_2$ escalan la curva de disonancia segun la frecuencia base, ventaja critica para pitch chords que retienen informacion de octava; Hutchinson-Knopoff asume una relacion lineal con el SPL que deforma la topologia en registros graves \cite{Vassilakis2001, Sethares1993}. Tercero, la **relacion timbre-escala** (consonancia local): Sethares permite calcular curvas de disonancia para espectros arbitrarios y encontrar escalas optimas, extendiendo la exploracion a timbres inarmónicos o microtonales \cite{Sethares1993}. Cuarto, Vassilakis anade precision en la fluctuacion de amplitud pero requiere variables dinamicas (AF-degree) poco relevantes para inputs simbolicos/MIDI donde las amplitudes se fijan teoricamente como $\delta^{k-1}$ \cite{Vassilakis2001}. La implementacion computacional confirma esta preferencia: el paquete *hrep* de Harrison y Pearce \cite{Harrison2020} implementa Sethares como backend principal, y Gaulhiac et al. usan la misma formulacion para generar mapas armonicos interactivos \cite{Gaulhiac2021}. Se reconoce que Sethares comparte con HK la anomalia de clasificar la triada aumentada como mas consonante que la menor; futuras extensiones podrian incorporar compacidad (Masina 2024) para corregir esta limitacion.
    <!-- FIN REDACTADO: Q-014 -->

#### 3.2.2 La Función de Disonancia entre Parciales

**Ecuación 3.1 (Disonancia entre dos sinusoides):**
$$d(f_a, f_b, A_a, A_b) = A_a A_b \left[ C_1 e^{A_1 S \Delta f} + C_2 e^{A_2 S \Delta f} \right]$$
donde:
- $\Delta f = |f_b - f_a|$
- $S = \frac{D^*}{S_1 \cdot \min(f_a, f_b) + S_2}$ (escala crítica)
- Parámetros: $D^* = 0.24$, $S_1 = 0.0207$, $S_2 = 18.96$, $C_1 = 5$, $C_2 = -5$, $A_1 = -3.51$, $A_2 = -5.75$.

**Contenido adicional:**
- Interpretar cada parámetro: $S$ es una escala inversamente proporcional a la banda crítica; $C_1, C_2$ controlan la forma de campana de la curva de disonancia.
- Gráfica conceptual: curva de disonancia para un par de sinusoides como función de $\Delta f / f_{\min}$. (Nota de santimath: esta grafica es muy importante para el lector. y alguito más habrá que decir sobre los parametros y como yo creo que en marco teorico mencionamos otras formas de modelar este fonomeno.. habra que decir algo sobre esos y nuestro enfoque enel fenomono fioslogigo auditivo)

**Preguntas clave:**
- ¿De dónde vienen estos valores numéricos? (Ajuste empírico de Sethares a los datos de Plomp-Levelt). (Nota de santimath: y que tan robusto es este modelo?)

    <!-- REDACTADO: Q-015 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-16 -->
    > **Resolución (Q-015):** Los parametros de la funcion de disonancia de Sethares no son constantes fisicas universales, sino artefactos de un **ajuste de curvas** (*curve fitting*) sobre los datos psicoacusticos promediados de Plomp y Levelt (1965). Los coeficientes $A_1 = -3.51$ y $A_2 = -5.75$ (tasas de ascenso y descenso de la curva) se obtuvieron mediante una **minimizacion del gradiente del error cuadratico medio** entre la curva parametrica $d(x) = e^{-ax} - e^{-bx}$ y los datos experimentales \cite{Sethares1993}. El parametro $D^* = 0.24$ no se ajusta independientemente: se deriva analiticamente igualando a cero la derivada de $d(x)$, y corresponde aproximadamente al 25\% del ancho de banda critico, donde ocurre la maxima disonancia \cite{Sethares1993, Mukherjee2023}. Los parametros de escalamiento $S_1 = 0.0207$ y $S_2 = 18.96$ provienen de un ajuste de minimos cuadrados lineal para interpolar el ancho de banda critico como funcion de la frecuencia: $s = D^*/(S_1 f_{\min} + S_2)$ \cite{Sethares1993}; Sethares (1993) los redondeo a 0.021 y 19. Los coeficientes $C_1 = 5$ y $C_2 = -5$ son factores de normalizacion para que la amplitud maxima de $d$ sea unitaria. Diferentes autores reportan ligeras variaciones: Mukherjee cita $D^* = 0.22035$; Vassilakis refina $S_1, S_2$ a mayor precision; Gaulhiac confirma $b_1 = 3.5, b_2 = 5.75$ como estandar \cite{Mukherjee2023, Vassilakis2001, Gaulhiac2021}.
    <!-- FIN REDACTADO: Q-015 -->

- ¿Cuál es la sensibilidad del modelo a variaciones en estos parámetros? (Señalar que el repositorio permite sobreescritura; discutir en §3.8 Supuestos). (Nota de santimath: que permite ponert otros modelos de rugosidad pero que apra este rabajo nos quedamos con este jmmm y toca decir porque)

    <!-- REDACTADO: Q-016 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-16 -->
    > **Resolución (Q-016):** La sensibilidad del modelo debe distinguirse en dos niveles. A nivel **estadistico** (ajuste global a datos perceptuales), el modelo es robusto: Masina y Lo Presti \cite{MasinaLoPresti2024} demuestran que al variar la funcion de peso de los armonicos (de $1/n$ a $1/\beta^{n-1}$), los valores de chi-cuadrado no cambian significativamente, indicando que la prediccion general de consonancia es estable ante perturbaciones moderadas de los parametros. A nivel **topologico** (existencia y ubicacion de minimos locales), la sensibilidad reside primariamente en las **amplitudes** de los parciales, no en $A_1, A_2$. Mukherjee \cite{Mukherjee2023} formaliza esto: perturbaciones locales en la envolvente espectral pueden crear o eliminar singularidades incidentales, alterando la estructura de minimos de la curva de disonancia. Sin embargo, la **ubicacion** de las consonancias principales (unisono, octava, quinta) es robusta porque depende mas de la estructura espectral del timbre que de variaciones finas en la pendiente de la curva de rugosidad \cite{Sethares1993}. Respecto a modelos alternativos: Vassilakis logra correlacion $r = 0.98$ vs $r = 0.87$ de HK con datos experimentales \cite{Vassilakis2001}; el repositorio ChordSpace permite sobreescritura de parametros (arquitectura modular), y el paquete *hrep* implementa Sethares, HK y Vassilakis como backends intercambiables \cite{Harrison2020}. La eleccion pragmatica de Sethares con parametros estandar se justifica para el alcance exploratorio del trabajo actual, delegando comparaciones sistematicas como linea futura.
    <!-- FIN REDACTADO: Q-016 -->

#### 3.2.3 Extensión a Tonos Complejos: Modelo de Parciales Armónicos

**Definición 3.8 (Espectro armónico):** Para una frecuencia fundamental $f_0$, se genera un espectro de $H$ parciales: $\{(f_0 \cdot k, \delta^{k-1})\}_{k=1}^{H}$, donde $\delta \in (0,1)$ es la tasa de decaimiento de amplitud. (  Nota de santimath: esta deficnion es estandar? para un matemaotico quedara claro la palabra espectro armonio aca? algo más de detalle habra que decir o almenos rcordar que antras se explico con mas detalle en otro capitulo )

**Parámetros del repositorio:** $H = 6$ armónicos, $\delta = 0.88$.

**Ecuación 3.2 (Rugosidad entre dos notas):**
$$R(f_i, f_j) = \sum_{p=1}^{H} \sum_{q=1}^{H} d(f_i \cdot p,\ f_j \cdot q,\ \delta^{p-1},\ \delta^{q-1})$$

**Contenido adicional:**
- Observar que $R$ depende de las frecuencias absolutas, no solo del intervalo: un mismo intervalo suena diferente en registro grave vs agudo. Esta es una diferencia fundamental con modelos basados puramente en clases de intervalo. (Nota de santimath: esto hayq ue desarrolarlo más y fundamentarlo con refenrencias)
- Complejidad: para un acorde de $m$ notas y $H$ armónicos, el número total de evaluaciones de $d$ es $\binom{m}{2} \cdot H^2$. (Nota de santimath: real? verifiquemos esto con el repo y mostremos un poco el detalle ) 

**Preguntas clave:**
- ¿Es el supuesto de espectro armónico con decaimiento exponencial razonable? (Razonable para instrumentos de cuerda/viento; no para percusión o espectros inarmónicos). (Nota de santimath: exlenete preguna y hay que desarrolarla  ) 

    <!-- REDACTADO: Q-017 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-16 -->
    > **Resolución (Q-017):** El supuesto de espectro armonico con decaimiento exponencial ($a_k = \delta^{k-1}$, $\delta = 0.88$) es una idealizacion razonable y ampliamente utilizada para modelar timbres "neutros" de instrumentos occidentales (cuerdas, vientos), donde la energia de los parciales disminuye monotonamente con el orden \cite{Sethares1993, Cook2006}. Cook y Fujisawa validan este parametro: con 6 parciales y amplitudes $(1.0, 0.88, 0.76, 0.64, 0.58, 0.52)$ obtienen resultados de clasificacion de triadas concordantes con datos experimentales \cite{Cook2006}. Harrison y Pearce \cite{Harrison2020} prefieren $1/n$ con 11 parciales para modelado cognitivo general; Milne et al. \cite{Milne2023} usan hasta 36 armonicos ($1/h$) para espectros idealizados, encontrando correlaciones de $r = 0.85$ para rugosidad entre espectros idealizados y audio real, pero solo $r = 0.49$ para armonicidad. Para **timbres inarmonicos** (percusion, campanas, barras: parciales $f, 2.758f, 5.406f$...), el modelo estandar es invalido: los minimos de disonancia no coinciden con la escala cromatica \cite{Sethares1993}. Sethares aborda esto mediante el principio de consonancia local: dado cualquier espectro, su algoritmo encuentra la escala optima (o viceversa). ChordSpace asume armonicidad como condicion de contorno explicita; la extension a timbres inarmonicos se identifica como linea futura. Masina y Lo Presti \cite{MasinaLoPresti2024} confirman que variar la funcion de peso ($1/n$ vs $1/\beta^{n-1}$) agudiza los picos de consonancia pero no altera el ordenamiento relativo, sugiriendo robustez ante variaciones moderadas de $\delta$.
    <!-- FIN REDACTADO: Q-017 -->

- ¿Cómo afecta $H=6$ a la resolución del modelo? ¿Sería diferente con $H=10$? (Nota de santimath:hay quemsotrar graficas o en est capitulo proponer una explciacion y quiza un exprimento para comporbar esto creo que el repo tiene algo de eso, recuerdo haberlo pensado y algo de codigo al respecto debe haber ) 

    <!-- REDACTADO: Q-018 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-16 -->
    > **Resolución (Q-018):** La eleccion de $H = 6$ armonicos tiene justificacion historica, perceptual y computacional. Historicamente, tanto Sethares (1993) como Plomp y Levelt (1965) utilizaron 6 componentes en sus experimentos fundamentales, convencion replicada por Cook y Fujisawa \cite{Cook2006, Sethares1993}. Perceptualmente, Helmholtz argumento que los armonicos a partir del septimo tienen influencia despreciable en la percepcion de batimientos debido a su baja amplitud en instrumentos reales \cite{Cook2006}. Masina y Lo Presti \cite{MasinaLoPresti2024} confirman un punto de **saturacion**: fijaron $n_{\max} = 8$ y demostraron que "valores mas grandes no afectan significativamente los resultados ya que la contribucion de tales armonicos superiores es minima." Hutchinson y Knopoff usan $H = 10$ con decaimiento $1/n$, pero la contribucion marginal de los armonicos 7-10 es despreciable dado que la disonancia depende del producto de amplitudes: para $H = 10$, la amplitud del decimo armonico es $\delta^9 = 0.88^9 \approx 0.31$, y su contribucion cruzada es $< 0.10$ \cite{Hutchinson1978}. Computacionalmente, el costo es $\binom{m}{2} \cdot H^2$: para una triada ($m = 3$), $H = 6$ produce 108 evaluaciones vs 300 con $H = 10$ (incremento de $2.8\times$ sin ganancia perceptual significativa). Cubarsi \cite{Cubarsi2019} anade que la tolerancia auditiva a desafinaciones (1-3\%) es mayor en armonicos bajos ($n < 5$), justificando priorizar precision en los primeros parciales. La eleccion $H = 6$ es por tanto conservadora y eficiente; un analisis de sensibilidad numerico ($H \in \{4, 6, 8, 10\}$) se propone como experimento complementario.
    <!-- FIN REDACTADO: Q-018 -->

#### 3.2.4 Rugosidad Total y Vectorización por Clase de Intervalo

**Definición 3.9 (Rugosidad total del acorde):**
$$R_{\text{total}}(\mathbf{n}) = \sum_{1 \leq i < j \leq m} R(f(n_i), f(n_j))$$

**Definición 3.10 (Histograma de rugosidad — vector de características):** $\Phi_{\text{raw}}: \mathcal{A} \to \mathbb{R}_{\geq 0}^{12}$, donde la componente $k$ agrega la rugosidad de todos los pares cuya distancia en semitonos módulo 12 cae en el bin $k$ (según la convención de §3.1.4):
$$\Phi_{\text{raw},k}(\mathbf{n}) = \sum_{\substack{i < j \\ \text{bin}(n_j - n_i) = k}} R(f(n_i), f(n_j))$$
(Nota de santimath: esa convenion que se mecniona deberia estar hipervinvulada en el latex enla version final revisada. tambien creo que hay que poner un ejemplo para el lector. porque se llama "raw" ? creoq ue la notacion no es la mejor o si? es clara? a que se refiere el sub indice k aca?)

**Proposición 3.2:** $\sum_{k=0}^{11} \Phi_{\text{raw},k}(\mathbf{n}) = R_{\text{total}}(\mathbf{n})$.

**Contenido adicional:**
- Interpretar: $\Phi_{\text{raw}}$ es una "firma psicoacústica" del acorde que distribuye la rugosidad total por clase de intervalo.
- Ejemplo numérico: para una tríada mayor en posición fundamental $(60, 64, 67)$, mostrar los valores de $\Phi_{\text{raw}}$ y señalar que los bins correspondientes a la tercera mayor (4 st) y la quinta justa (7 st) dominan.(Nota de santimath: Esto me hace pensar en que no se si hemos meniconado epxlcaimtemene ejemplos de la cantidadde intervalos que tiene un acorde y que digamos ideas para otros capitulos com la discucision seria que quiza el peso o la improtancia del intervalo varia dpeendeindosi esta en los extremos medios del acorde me gustaria ver que hay en los notebooklm sobre esto la perepccion de un intervalo interno que onda que hace el verrobo con eso)

    <!-- REDACTADO: Q-019 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-17 -->
    > **Resolución (Q-019):** La investigación psicoacústica demuestra que la percepción de un acorde de tres o más notas no es una suma lineal de intervalos constituyentes, y que la posición relativa de los intervalos internos modifica su peso perceptual. El modelo de Sethares calcula la rugosidad total sumando $d(f_i,f_j)\cdot a_i a_j$ sobre *todos* los pares de parciales con pesos iguales (salvo amplitud), sin distinguir intervalos "internos" de "externos" \cite{Sethares1993}. Esta simetría del modelo genera anomalías predictivas: la tríada aumentada (equidistante: 4-4 st) se predice como más consonante que la menor, contradiciendo la percepción empírica \cite{Cook2009, Masina2024}. Cook (2009) demuestra que la **tensión triádica** es un fenómeno de tres tonos irreducible a díadas: la equidistancia interválica genera inestabilidad perceptual, mientras que la asimetría (4-3 vs 3-4 st) define la modalidad mayor/menor. Los modelos de **compacidad** (periodicidad/harmonicidad) revelan que la posición de la nota media altera el bajo fundamental virtual, cambiando la consonancia según la inversión \cite{Masina2023}. Cambouropoulos (GCT) prioriza el intervalo con el bajo como determinante de la identidad del acorde \cite{Cambouropoulos2016}, y Lerdahl propone una jerarquía cognitiva donde la raíz pesa más que la quinta y la tercera. Para ChordSpace, dado que $\Phi_{\text{raw}}$ agrega rugosidad por clase de intervalo sin distinguir posición, esta limitación debe documentarse: el vector captura el *contenido* total de intervalos pero no la *configuración* interna (e.g., 3-4 vs 4-3), lo cual es una simplificación deliberada del modelo.
    <!-- FIN REDACTADO: Q-019 -->

**Debilidades a vigilar:**
- El vector $\Phi_{\text{raw}}$ puede tener componentes cero para clases de intervalo ausentes en el acorde, especialmente en acordes de baja cardinalidad. Esto afecta la normalización posterior. (Nota de santimath: esto es un detalle a la hroa de esocger la metrica de distancias apara el proeblam de minimizacion y otros detalles tecniso no? hay que pensarlo muy bien y anticiparlo para que el lector se vaya haciendo una idea)

    <!-- REDACTADO: Q-020 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-17 -->
    > **Resolución (Q-020):** Los ceros en $\Phi_{\text{raw}}$ afectan diferencialmente a cada métrica. La **JSD** requiere manejo explícito porque $D_{\text{KL}}$ involucra $\log(p_k/q_k)$, indefinido cuando $p_k=0$ o $q_k=0$; la solución estándar es el *add-one smoothing* (prior de Bayes-Laplace) o sumar $\epsilon = 10^{-12}$ \cite{Burgoyne2013}. `scipy.spatial.distance.jensenshannon` ya calcula $\sqrt{\text{JSD}}$ e internamente maneja ceros mediante convención $0\log 0 = 0$. La **distancia de Hellinger** $d_H = \frac{1}{\sqrt{2}}\|\sqrt{p}-\sqrt{q}\|_2$ opera con raíces cuadradas, evitando la singularidad logarítmica, siendo "naturalmente" robusta a ceros \cite{Kim2018}. La **distancia coseno** tolera ceros mientras los vectores no sean nulos (magnitud cero), pero dos acordes sin intervalos compartidos serán ortogonales (similitud = 0), lo cual es matemáticamente correcto pero potencialmente exagerado perceptualmente \cite{Wolkowicz2013}. La **euclidiana** no tiene problemas con ceros per se, pero dominarán las diferencias de magnitud. Desde el análisis de datos composicionales (CoDA), Burgoyne et al. advierten que las correlaciones espurias surgen al trabajar sobre el simplex con estadísticas estándar; la transformación **ilr** (*isometric log-ratio*) requiere componentes $>0$, por lo que los ceros deben tratarse con *balances* o reemplazos \cite{Burgoyne2013}. En la práctica de ChordSpace, las estrategias son: (1) suavizado gaussiano previo (`simplex_smooth`) que difunde energía a bins adyacentes eliminando ceros duros, y (2) $\epsilon$-padding para JSD.
    <!-- FIN REDACTADO: Q-020 -->
- La dependencia en frecuencias absolutas significa que dos tríadas mayores en octavas distintas tendrán $\Phi_{\text{raw}}$ diferentes. ¿Es esto deseable? (Sí, si se quiere capturar la percepción real; no, si se busca abstracción por transposición). (Nota de santimath: claro esto es deseable nos diferencia de otros enfoques prometemos reprensentar el acorde tal cual el musico lo intepretaria en el controlador midi, a proposito ya le contamos al lector sobre el protocolo MIDI? eso se hace en este capitulo o en el marco teorico?)

#### 3.2.5 Validación Interna del Componente de Rugosidad

**Contenido:**
- Curva de referencia para díadas: comparación con la curva canónica de Plomp-Levelt/Sethares para dos tonos puros.
- Tests automatizados que verifican relaciones cualitativas: $R(\text{2da menor}) > R(\text{octava}) > R(\text{5ta justa})$.
- Consistencia numérica: versión vectorizada vs. escalar con tolerancia $10^{-6}$.

**Preguntas clave:**
- ¿Es suficiente validar con díadas? ¿Cómo se verifica que la extensión a acordes de mayor cardinalidad es correcta? (Nota de santimath: excelente pregunta)
- ¿Existe un "ground truth" perceptual contra el cual validar para tríadas o cuatríadas?(Nota de santimath: creo que tambien vale la pena preguntarse en que unidades de medida se mide la rugosidad? que significa un valor de 0.5? o 1? o 100? )

    <!-- REDACTADO: Q-021 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-17 -->
    > **Resolución (Q-021):** Sí existen datos de *ground truth* para acordes de mayor cardinalidad. **Bowling et al. (2018)** proporcionan calificaciones perceptuales promediadas para las 66 tríadas y 220 cuatríadas cromáticas posibles dentro de una octava, superando estudios previos limitados a acordes familiares \cite{Bowling2018}. **Johnson-Laird et al. (2012)** evaluaron las 55 tríadas de tres notas, y **Roberts (1986)** estableció la jerarquía empírica estándar: mayor > menor > disminuida > aumentada, confirmada por Cook (2009) \cite{Cook2009}. Respecto a las unidades, la unidad psicofísica formal de rugosidad es el **asper** (1 asper = rugosidad de un tono SAM de 1 kHz a 60 dB SPL, modulado al 100% a 70 Hz) \cite{Vencovsky2014}. Sin embargo, en los modelos computacionales (Sethares, HK) los valores son arbitrarios o normalizados; un valor de 1.0 típicamente representa la disonancia máxima del sistema analizado, no un valor físico absoluto. La validación con díadas es **insuficiente** para acordes: la suma aditiva de disonancia de intervalos predice incorrectamente que la tríada aumentada es más consonante que la mayor \cite{Cook2009, Masina2024}. Para $N \geq 3$ emergen fenómenos de tensión estructural y compacidad irreducibles a pares. Se recomienda usar el dataset de Bowling et al. como *ground truth* para validar el modelo de tríadas en ChordSpace, y considerar la combinación de rugosidad + compacidad (Masina 2024) para mejor ajuste estadístico.
    <!-- FIN REDACTADO: Q-021 -->

---

### 3.3 Transformaciones del Vector de Características: Propuestas de Normalización

*Motivación: El vector crudo $\Phi_{\text{raw}}$ no es directamente comparable entre acordes de diferente cardinalidad o diferente rugosidad total. Las normalizaciones son una contribución metodológica central de este trabajo.* 

(Nota de santimath: si es comparable, porque de todas maneras todos los vectores tienen 12dim, pero lo que he visto es que cuando se jutnan en un experimento acordes con intercalos muy cercanos muy rugosos el espacio cisual MDS 2d se distoriciona un poco per es viusalmente diria yo que no es un problema mayor, pero si hay que tenerlo en cuenta y desarrollarlo hay que hacer pruebas yo creo en el repositorio)

#### 3.3.1 El Problema de la Comparabilidad

**Contenido:**
- Acordes con más notas tienen más pares → mayor rugosidad total → magnitud de $\Phi_{\text{raw}}$ crece con $m$.
- Para comparar "formas" de distribución de rugosidad (no magnitudes absolutas), se requiere alguna normalización.
- Dilema: normalizar elimina información de magnitud; no normalizar introduce sesgo por cardinalidad.

**Preguntas clave:**
- ¿La normalización destruye información relevante? ¿Cuál?
- ¿Es posible que la mejor normalización dependa de la pregunta que se quiere responder?

    <!-- REDACTADO: Q-022 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-17 -->
    > **Resolución (Q-022):** La normalización L1 (proyección al simplex) preserva el *perfil relativo* de rugosidad pero elimina la **magnitud global de disonancia sensorial**, que es perceptualmente informativa. Los modelos psicoacústicos calculan la rugosidad total como suma ponderada; esta magnitud acumulada predice directamente la percepción de estabilidad y tensión \cite{Sethares1993, Milne2023}. Milne et al. (2023) demuestran que la magnitud absoluta de rugosidad tiene una asociación universal con la percepción de estabilidad musical, independiente de la cultura. La estrategia de normalización depende estrictamente de la pregunta: para **similitud** tímbrica, la distribución normalizada compara perfiles espectrales sin sesgo por cardinalidad; para **ordenamiento** por consonancia o **clasificación** de tensión, la magnitud no normalizada es esencial \cite{Masina2022}. Desde el análisis de datos composicionales (CoDA), Burgoyne et al. advierten que la normalización al simplex introduce correlaciones espurias si se aplican estadísticas estándar; la transformación **ilr** (*isometric log-ratio*) de Aitchison preserva la geometría composicional en un espacio euclidiano de $N-1$ dimensiones \cite{Burgoyne2013}. En el dominio de la DFT (Amiot, Bernardes), la información se separa en magnitud (invariante a transposición) y fase, permitiendo normalizar selectivamente. Para ChordSpace, la estrategia explícita es evaluar *todas* las normalizaciones comparativamente (§3.6), documentando que `identity` preserva magnitud mientras `simplex` permite métricas de información.
    <!-- FIN REDACTADO: Q-022 -->

#### 3.3.2 Catálogo de Propuestas Evaluadas

**Definición 3.11 (Propuestas de normalización).** Sea $H = \Phi_{\text{raw}}(\mathbf{n}) \in \mathbb{R}_{\geq 0}^{12}$. Se definen las siguientes transformaciones:

| ID | Fórmula | Interpretación |
|---|---|---|
| `identity` | $X = H$ (sin modificar) | **Baseline de control.** Mantiene magnitud absoluta. |
| `simplex` | $X = H / \|H\|_1$ | Proyección al simplex unitario $\Delta^{11}$. Interpreta $X$ como distribución de probabilidad sobre clases de intervalo. |
| `simplex_sqrt` | $X = \sqrt{H} / \|\sqrt{H}\|_1$ | Comprime rango dinámico antes de normalizar. Atenúa picos dominantes. |
| `simplex_smooth` | $X = \text{Gauss}_\sigma(\sqrt{H}) / \|\cdot\|_1$ | Suaviza con kernel gaussiano ($\sigma = 0.75$) y normaliza. Modela "difusión perceptual" entre bins adyacentes. |
| `perclass_α` ($\alpha \in \{0.25, 0.5, 0.75, 1\}$) | $X_k = H_k / \mathbf{ic}_k^\alpha$ (luego simplex) | Normaliza por la multiplicidad de pares en cada clase, elevada a $\alpha$. Para $\alpha=1$, es rugosidad promedio por par en cada clase. |
| `global_pairs` | $X = H / P$ donde $P = \binom{m}{2}$ | Divide por el número total de pares. Corrige por cardinalidad globalmente. |
| `divide_mminus1` | $X_k = H_k / (\mathbf{ic}_k - 1)$ si $\mathbf{ic}_k \geq 2$ | Heurística para penalizar duplicidades. |

(Nota de santimath: sera que toca poner graficas de esto o explirar que la GUI permite elegir las normalziacion y con un delizador configrar esto para acomodar a lo que el musico quiere ? o este capitulo no debe llevar graficas? sino donde iria?)

**Proposición 3.3:** Las propuestas `simplex`, `simplex_sqrt` y `simplex_smooth` producen vectores en el simplex $\Delta^{11} = \{x \in \mathbb{R}_{\geq 0}^{12} : \sum x_k = 1\}$, lo cual habilita el uso de divergencias de la teoría de la información (Jensen-Shannon, Hellinger).

**Proposición 3.4:** Las propuestas `perclass_α` con $\alpha = 1$ producen la rugosidad promedio por par en cada clase de intervalo, eliminando la dependencia lineal en la multiplicidad pero preservando la dependencia en la rugosidad absoluta por par (que depende del registro).

(Nota de santimath: no se si estas proposiciones son realmente proposiciones o simplemente definiciones, pero suenan bien,y eso me asusta, no enteindo bien a que se refienre hay que explcairlo más a fondo y buscar ejemplos para entenderlo mejor quiza y validar que es matemáticamente correcto lo que se propuso en esas normalizacion buscar referentes quiza no de lamusic a sino de un probelma general de resolver que se carga mucho la escala de color en las graficas y una normalziacion lo resolvio? porque ahora que recuerdo lo hice porque la grafica toda quedaba muy en extremos de colroazion por rugosidad y esas normalziacion lo que hacian era suavisar esos colores, uscar referemtes quiza)

**Contenido adicional:**
- Tabla comparativa con ejemplo numérico para una tríada y una cuatríada.
- Discusión: ¿cuál normalización es "mejor"? → No hay respuesta a priori; esta es una pregunta experimental que motiva el diseño comparativo del §3.6.

**Preguntas clave:**
- ¿El suavizado gaussiano tiene justificación perceptual o es una heurística?

    <!-- REDACTADO: Q-023 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-17 -->
    > **Resolución (Q-023):** El suavizado gaussiano posee justificación perceptual sólida, no es puramente heurístico. La percepción del tono es "indulgente e imprecisa", correspondiendo a una distribución de probabilidad que justifica modelar la "computación blanda" del cerebro mediante suavizado \cite{Himpel2022}. Este suavizado simula el **umbral de discriminación** (*Difference Limen*, DL) y la **diferencia apenas perceptible** (*JND*), permitiendo que modelos discretos se extiendan al continuo perceptivo. **Harrison y Pearce (2020)** implementan espectros suavizados mediante convolución gaussiana ($\sigma \approx 10$ cents) para simular la imprecisión perceptual y representar la "imagen sensorial" \cite{Harrison2020}. **Masina (2023)** extiende indicadores discretos al continuo con $\sigma$ calibrado al umbral de discriminación auditivo \cite{Masina2023}. El precedente en MIR es extenso: los *chroma vectors* suavizados son práctica estándar; Bernardes et al. (TIS) transforman vectores discretos a espacios continuos vía DFT \cite{Bernardes2016}. Sin embargo, el valor específico $\sigma = 0.75$ semitonos (75 cents) en ChordSpace es significativamente mayor que el JND típico (3-20 cents). Se justifica como captura de la tolerancia **categórica** (clasificación dentro del semitono) más que como precisión psicoacústica fina; equivale a una regularización que difunde energía entre bins adyacentes, eliminando ceros duros y estabilizando las métricas de distribución. No obstante, debe documentarse como parámetro configurable y su sensibilidad explorada experimentalmente.
    <!-- FIN REDACTADO: Q-023 -->
- ¿Hay riesgo de sobreajuste al evaluar tantas propuestas? (Discutir en §3.8 corrección por comparaciones múltiples).

**Debilidades a vigilar:**
- Algunas propuestas (como `divide_mminus1`) están definidas solo cuando $\mathbf{ic}_k \geq 2$; documentar el manejo de casos degenerados. ( Nota de santimath: cual sera la idea ahi? hay que buscarlo en el repo explircarlo con el detalle suficiente)
- El lector podría preguntarse: "¿por qué no usar directamente PCA/selección de features en lugar de heurísticas manuales?" → Argumentar que las normalizaciones tienen interpretación musical/perceptual, lo cual es preferible a transformaciones opacas. (Nota de santimath: esto es importante hay que explciarlo bien)

    <!-- REDACTADO: Q-024 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-17 -->
    > **Resolución (Q-024):** La literatura respalda contundentemente la preferencia por transformaciones interpretables frente a métodos opacos en el contexto de representación de acordes. **Harrison y Pearce (2020)** argumentan que los modelos cognitivos deben basarse en representaciones auditivas de bajo nivel (rugosidad, harmonicidad) en lugar de abstracciones teóricas opacas \cite{Harrison2020}. **Tymoczko** construye orbifolds geométricos con coordenadas musicalmente explícitas; **Chew** (*Spiral Array*) diseña un espacio basado en intervalos de quinta y tercera; ninguno usa PCA como base representacional \cite{Tymoczko2006, Chew2014}. **Lazzari et al.** demuestran que *pitchclass2vec*, que codifica la estructura interna del acorde, supera a embeddings de NLP tipo Word2Vec en segmentación estructural \cite{Lazzari2023}. **De Berardinis et al.** rechazan explícitamente Deep Learning por falta de interpretabilidad musicológica, optando por el Tonal Pitch Space de Lerdahl \cite{DeBerardinis2023}. **Himpel (2022)** prefiere modelos geométricos (variedades de Riemann) a ML para racionalizar fenómenos psicoacústicos \cite{Himpel2022}. Además, con dimensionalidad original $d=12$, PCA sería redundante: Burgoyne y Saul muestran que se necesitan múltiples componentes para capturar la información de un modelo teórico bien formulado. Para ChordSpace, las normalizaciones propuestas actúan como filtros con semántica musical (e.g., `simplex` = distribución de probabilidad sobre clases de intervalo, `perclass` = rugosidad promedio por par), preservando la interpretabilidad que PCA destruiría.
    <!-- FIN REDACTADO: Q-024 -->

---

### 3.4 Geometría del Espacio de Acordes: Métricas de Disimilitud

*Motivación: Con el vector $\Phi$ (crudo o normalizado) se induce una geometría sobre $\mathcal{A}$. La elección de métrica determina qué noción de "similitud sonora" operamos.*

#### 3.4.1 Métricas Vectoriales

**Definición 3.12 (Distancia coseno):**
$$d_{\cos}(\mathbf{x}, \mathbf{y}) = 1 - \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}$$

**Propiedad 3.5:** $d_{\cos}$ es una disimilitud en $[0, 2]$ pero NO es una métrica (no satisface la desigualdad triangular en general). Sin embargo, $\arccos(1 - d_{\cos})$ sí es métrica. En la práctica, se usa $d_{\cos}$ como disimilitud para `pdist` y para reducción dimensional.

**Definición 3.13 (Distancia euclidiana y Manhattan):**
$$d_{\text{euc}}(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2, \qquad d_{\text{man}}(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_1$$

**Contenido adicional:**
- Observar que la distancia euclidiana sobre vectores normalizados al simplex es proporcional a la distancia de Hellinger (a un factor $\sqrt{2}$ de las raíces cuadradas).
- Interpretar: coseno mide diferencia angular (forma de la distribución); euclidiana mide diferencia absoluta; Manhattan penaliza más las diferencias en componentes individuales.

**Preguntas clave:**
- ¿Tiene sentido usar la distancia euclidiana sobre vectores de rugosidad cruda (`identity`)? (Sí, pero dominarán las diferencias de magnitud, no de forma). (Nota de santimath: uf, esto es muy importante hay que gastarle tiempo al lector aca en que entienda que haria cada metrica que dominca en cada una que eonfque teiene y que´descubrimos esto hayq ue ahcerlo con ayuda de experimentar sobre el repo y buscr refenrtes en el notebooklm)

    <!-- REDACTADO: Q-025 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-17 -->
    > **Resolución (Q-025):** La elección de métrica depende críticamente del espacio de representación. La **euclidiana sobre vectores crudos** (*raw chroma*) es inapropiada perceptualmente: una segunda menor y una quinta justa pueden tener distancia euclidiana idéntica a pesar de sus diferencias radicales en disonancia \cite{Bernardes2016}. La **similitud del coseno** es respaldada por Harrison y Pearce para comparar espectros suavizados, pues captura la similitud de perfil espectral independientemente de la magnitud \cite{Harrison2020}. En el *Tonal Interval Space* (TIS) se hace una distinción funcional: la euclidiana (magnitud desde el centro) indica consonancia global, mientras que las medidas angulares capturan alineación armónica \cite{Bernardes2016, NavarroCaceres2020}. Formalmente, $\sqrt{\text{JSD}}$ es métrica válida en el simplex (Endres & Schindelin, 2003); Hellinger es métrica acotada en $[0,1]$, robusta a ceros; coseno es semimétrica (requiere $\arccos$ para la desigualdad triangular). **Tymoczko** usa distancia de voice-leading (norma $L^\infty$ en el orbifold); **Harrison** usa euclidiana en el espacio de embeddings. No hay consenso universal. Para ChordSpace, el diseño experimental compara múltiples métricas (coseno, euclidiana, JSD, Hellinger) sobre múltiples normalizaciones, tratando la elección como pregunta empírica en lugar de decisión a priori (§3.6).
    <!-- FIN REDACTADO: Q-025 -->

#### 3.4.2 Métricas de Distribuciones de Probabilidad

**Definición 3.14 (Divergencia de Jensen-Shannon):**
$$\text{JSD}(p, q) = \frac{1}{2} D_{\text{KL}}(p \| m) + \frac{1}{2} D_{\text{KL}}(q \| m), \quad m = \frac{p + q}{2}$$
donde $D_{\text{KL}}(p \| q) = \sum_k p_k \log_2 \frac{p_k}{q_k}$.

**Proposición 3.6 (JSD como métrica):** $\sqrt{\text{JSD}}$ es una métrica válida en el simplex $\Delta^{11}$ (Endres & Schindelin, 2003; Österreicher & Vajda, 2003). Es decir, satisface: (i) no negatividad, (ii) simetría, (iii) identidad de indiscernibles, y (iv) desigualdad triangular.

**Definición 3.15 (Distancia de Hellinger):**
$$d_H(p, q) = \frac{1}{\sqrt{2}} \left\| \sqrt{p} - \sqrt{q} \right\|_2$$

**Proposición 3.7:** $d_H$ es una métrica en $\Delta^{11}$ y está acotada en $[0, 1]$.

**Contenido adicional:**
- Justificación de por qué las métricas de distribuciones son apropiadas cuando $\Phi$ se normaliza al simplex: estamos comparando "perfiles" de distribución de rugosidad, no magnitudes absolutas.
- Relación entre JSD y Hellinger: ambas pertenecen a la familia de $f$-divergencias; JSD es más sensible a diferencias en colas, Hellinger más estable numéricamente.

**Preguntas clave:**
- ¿Cuándo preferir JSD vs Hellinger? (JSD es más discriminante para distribuciones con soporte diferente; Hellinger es más robusta a zeros numéricos). ( Nota de santimath: hay que ver como safar de eso repaidamente oy decir que basicmaente nos quedamos con cosenos eucliddan y quiza otra y explicar que pasa con las otras pero brevemente no dejar que se vuelva un hoyo de pregnutas eso hayq ue cerrar el asunto de las metricas muy bien dejarlo zanjado para que no hayan muchas preguntas)
- ¿El uso de $\log_2$ vs $\ln$ en JSD afecta la métrica? (Solo escala; la propiedad métrica de $\sqrt{\text{JSD}}$ se mantiene).

**Debilidades a vigilar:**
- Para vectores con componentes cero (acordes con pocas clases de intervalo activas), JSD requiere manejo numérico (epsilon). Documentar la implementación (`eps = 1e-12`).(Nota de santimath: por ejemplo!!! esa es una gran pregunta y creo que hay que resovlerlo de manera intelignte para no perder tiempo, decir que el modelo permite poner varias metricas y que esocgmos unaspara probar no como tal la mejor no se como decir esto pero queiza tu contu concociemitno de esto peudas decirnos cual es la mejor metrica o porque )

#### 3.4.3 Construcción de la Matriz de Distancias $D$

**Definición 3.16:** Para una población de $N$ acordes y una disimilitud $\rho$, la matriz de distancias es $D \in \mathbb{R}^{N \times N}$ con $D_{ij} = \rho(\Phi(c_i), \Phi(c_j))$.
(Nota de santimath: aca hay que ser muy precisos con la definicion de matriz de distancias y que es lo que representa, hay que ser muy claros con la notacion y que significa cada cosa, porque c_i que es? la notacion es consisntente con la que venismo trayendo? un lector matemático estará muy pendient de la notacion y su consistencia)
**Contenido:**
- Implementación eficiente via `scipy.spatial.distance.pdist` (vector condensado de $\frac{N(N-1)}{2}$ entradas) + `squareform`.
- Costo computacional: $O(N^2)$ en tiempo y espacio. Para $N = 2.6\text{M}$, la matriz ocupa $\sim 25$ TB → inviable sin aproximaciones.
- Poblaciones de trabajo experimental: $N \in [13, 10000]$ según el experimento.
- Caché por (normalización, métrica): si dos reducciones comparten $D$, se reutiliza.

**Preguntas clave:**
- ¿La elección de $\rho$ induce una topología diferente sobre $\mathcal{A}$? ¿Cómo verificarlo?
- ¿Es la matriz $D$ definida positiva / semidefinida? (Relevante para MDS clásico; no necesariamente para SMACOF). 

    <!-- REDACTADO: Q-026 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-17 -->
    > **Resolución (Q-026):** Sí, diferentes métricas inducen topologías genuinamente distintas. Burgoyne y Saul demuestran que MDS clásico genera una estructura toroidal estándar del espacio tonal, mientras que técnicas no lineales (MVU) revelan topologías más complejas con planos isomórficos separados por tono entero \cite{Burgoyne2005}. En el TIS, la euclidiana directa sobre chroma no captura propiedades armónicas, pero la DFT ponderada agrupa configuraciones armónicamente relacionadas \cite{Bernardes2016}. Los modelos de rugosidad y compacidad producen vecindarios *diferentes*: la rugosidad predice incorrectamente que la tríada disminuida es más consonante que la aumentada, mientras que los modelos de compacidad lo corrigen; solo la combinación lineal de ambas logra ajuste satisfactorio con datos empíricos \cite{Masina2024}. Respecto a la semidefinición positiva: la distancia de Lerdahl viola la desigualdad triangular (no es métrica verdadera), requiriendo "euclidización" de la matriz para MDS clásico. SMACOF (implementado en `sklearn.manifold.MDS`) opera por optimización iterativa con convergencia garantizada sobre cualquier matriz de disimilitud, sin requerir que $D$ sea euclidiana-embeddable. Sin embargo, si $D$ proviene de $\sqrt{\text{JSD}}$ (métrica válida) o Hellinger (métrica acotada), se satisfacen las condiciones de Schoenberg. Para coseno, $\arccos(1-d_{\cos})$ es métrica. ChordSpace debe reportar si el *stress* resultante se debe a incompatibilidad de la matriz o a compresión dimensional genuina.
    <!-- FIN REDACTADO: Q-026 -->
(Nota de santimath: Yo creo que aca faltan mas preguntas tecnias de computacion que se deben hacer)

---

### 3.5 Reducción Dimensional: Del Espacio $\mathbb{R}^{12}$ al Plano $\mathbb{R}^2$

*Motivación: El espacio de 12 dimensiones no es directamente visualizable. La reducción dimensional busca un mapeo $\Psi: \mathbb{R}^{12} \to \mathbb{R}^2$ que preserve la estructura geométrica inducida por $D$.* 
(Nota de santimath: aca hay que explicar que es la reduccion dimensional y que es lo que representa, pero además citar  y explciar la importancia de del analsisi visual y que sifgiffica preservar la estructura geométrica, entre otras cuestiones tecnicas de rigor matemático)

#### 3.5.1 Formulación como Problema de Optimización (MDS)

**Definición 3.17 (Multidimensional Scaling — Stress de Kruskal):** Dado $D \in \mathbb{R}^{N \times N}$, MDS busca $Y = \{y_1, \ldots, y_N\} \subset \mathbb{R}^2$ que minimice:
$$\text{Stress}(Y) = \sqrt{\frac{\sum_{i<j} (D_{ij} - \|y_i - y_j\|)^2}{\sum_{i<j} D_{ij}^2}}$$

(Nota de santimath: aqui hay que explicar que es el stress y que es lo que representa, pero además citar  y explciar ypo creo que lo gordo ya se tuvo que exlciar atars en el marco teorico imagino pero, RECUERDA este es nuestro primer capitulo en la actividad de escritura academica porque de aca sale lo minimo tecnico que habra que cubrir en el marco teorico, si? la verdad no se si es el minimo pero es el minimo que se me ocurre, tambien hay que revisar la notacion de stress, explciar el agloritmo adaptado a nuesotroc caso)

**Contenido:**
- Interpretación: el stress mide cuánto difieren las distancias en el embedding de las disimilitudes originales. Stress $= 0$ sería preservación perfecta.
- Algoritmo SMACOF (Scaling by MAjorizing a COmplicated Function): optimización iterativa con convergencia garantizada.
- Hiperparámetros: `n_init=4` (múltiples inicializaciones), `random_state=seed`, `n_components=2`, modo `metric=True`.
- Complejidad: $O(N^3)$ por iteración (cuello de botella del pipeline para $N$ grandes).


(Nota de santimath: yo creo que como en la propuesta de trabahjo proponemos hacer una exploraicon total y MDS es de los que mejor conservan las distancias originales pero el costo comutacional es tan alto habra que darnos la pela de implemtanr alguna tencia de landsmarks para ese experimento grande donde $N = 100\text{K}$  o no? como lo ves? es facil de sacar ?)

**Preguntas clave:**
- ¿Por qué MDS métrico y no no-métrico? (Métrico preserva magnitudes, no solo rangos; apropiado cuando la escala de $\rho$ es informativa). (Nota de santimath: uy la verdad aca no se que decir es una pregunta tecnica que se debe hacer y que yo no respondi en el repo, la verdad no se si MDS metrico o por que, tampoco sé como averiguarlo)
- ¿Es el stress una medida suficiente o necesitamos métricas complementarias? (Nota de santimath: Excelnte pregunta hay que ver si otros trabajos de reprenstancion de acordes usan MDS y como lo evaluan)

    <!-- REDACTADO: Q-027 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-17 -->
    > **Resolución (Q-027):** **Krumhansl (1990)** argumenta a favor del MDS **no métrico** para datos perceptuales porque los juicios humanos de similitud rara vez cumplen propiedades de escala de intervalo; el MDS no métrico solo requiere monotonicidad entre disimilitud percibida y distancia espacial, siendo más conservador ante ruido subjetivo \cite{Krumhansl1990}. Sin embargo, cuando la métrica de entrada tiene propiedades aritméticas definidas (como $\sqrt{\text{JSD}}$ o euclidiana sobre vectores computados), el MDS **métrico** es apropiado porque preserva las magnitudes absolutas de las diferencias, no solo sus rangos. Milne et al. (2023) demuestran que la magnitud absoluta de rugosidad tiene efecto predictivo cuantificable sobre la percepción de estabilidad \cite{Milne2023}, lo que justifica usar MDS métrico cuando el input es $\rho$ computacional. **Bernardes et al. (2016)** usan MDS no métrico con la librería *smacof* para visualizar el TIS \cite{Bernardes2016}. El **stress de Kruskal** (Fórmula 1) es la métrica estándar: valores $<0.05$ son excelentes, $<0.1$ buenos (Kruskal, 1964). Sin embargo, el stress es **insuficiente** como único indicador: se recomienda complementar con *trustworthiness* y *continuity* (Venna & Kaski), diagrama de Shepard (correlación entre distancias originales y del embedding), y `sklearn.manifold.trustworthiness()`. Para ChordSpace, dado que $\rho$ se computa (no proviene de juicios subjetivos), MDS métrico (`metric=True`) está justificado; se debe reportar stress + diagrama de Shepard + trustworthiness.
    <!-- FIN REDACTADO: Q-027 -->

#### 3.5.2 Métodos Complementarios: UMAP, t-SNE, ISOMAP

**Contenido para cada método:**

(Nota de santimath:  yo creo que hay que decidir cual metodo esocoger decir que nuestrosistem permite rpbar varios qy como hisicmos las ´preurbas para quedarnos con UMAP y MDS y por que etc etc )

**UMAP** (McInnes et al., 2018):
- Aproxima la estructura topológica local del espacio original mediante grafos fuzzy de simplicial sets.
- Hiperparámetros: `n_neighbors = max(2, min(15, N-1))`, `n_epochs=200`, `init="spectral"`.
- Ventaja: preserva estructura local; desventaja: las distancias globales no son interpretables.
(Nota de santimath: Claro por eso MDS y tambien por eso las otras tenicas porque teien valor encontrar cluster eso en si mismo puede resultar ser una nueva forma de explorar la musica, los acordes de sustituior de encontrar vecinos o acordes que suenen similarres, tener una metrica compuesta!!! algo como: "este acorde es vecino de este acorde" en UMAP y en MDS esta a esta distancia entonces un algoritmo de sustitucuin custerizado? algoasi creo que esta idea se puede desaroollar y engorosar el trabajo y la relevancia NOTEBOOKLM revisa estooo!)

**t-SNE** (van der Maaten & Hinton, 2008):
- Preserva probabilidades de vecindario local.
- Hiperparámetros: `perplexity` ajustada a $N$ (rango 2–10 en el pipeline), `init="random"`.
- Ventaja: excelente para visualizar clusters; desventaja: sensible a hiperparámetros, no preserva estructura global.

**ISOMAP** (Tenenbaum et al., 2000):
- MDS aplicado sobre distancias geodésicas (camino más corto en grafo de $k$-vecinos).
- Hiperparámetros: `n_neighbors=5`.
- Ventaja: preserva geodésicas en variedades; desventaja: sensible a cortocircuitos y huecos en los datos.

**Contenido adicional:**
- Tabla comparativa: qué preserva cada método (local vs global, métrico vs topológico).
- Justificación de por qué se incluyen múltiples métodos: no hay un reductor universalmente superior; la comparación es parte del diseño experimental.

**Preguntas clave:**
- ¿Los embeddings de diferentes métodos son directamente comparables? (Nota de santimath:  no lo sé , pero me parece una pregunta importante y tecnica y deberiamos pensar en como resolverlo)  
- ¿Cómo interpretar un embedding UMAP donde las distancias globales no son significativas? (Nota de santimath:  no lo sé , pero me parece una pregunta importante y tecnica y deberiamos pensar en como resolverlo)

    <!-- REDACTADO: Q-028 | Fuente: 5 notebooks NotebookLM | Workflow: /redactor-critico | Fecha: 2026-02-17 -->
    > **Resolución (Q-028):** Los embeddings de MDS, UMAP y t-SNE **no son directamente comparables** en sus distancias absolutas porque optimizan funciones de costo diferentes y preservan propiedades distintas. MDS preserva distancias globales por pares (stress); t-SNE minimiza la divergencia KL entre distribuciones de probabilidad locales; UMAP optimiza entropía cruzada difusa sobre conjuntos simpliciales, penalizando tanto la separación de vecinos como la agrupación incorrecta de puntos lejanos \cite{McInnes2018}. Las métricas estándar de comparación son **trustworthiness** (penaliza intrusiones: puntos lejanos que aparecen como vecinos) y **continuity** (penaliza extrusiones: vecinos que se separan), introducidas por Venna y Kaski e implementadas en `sklearn.manifold.trustworthiness()` \cite{Venna2005, Lee2009}. El marco unificador es la **matriz de co-ranking** $Q$, que compara rangos de distancia entre espacio original y embebido. Para alinear embeddings antes de compararlos, se usa **análisis de Procrustes** (rotación + escalado óptimos), disponible en `scipy.spatial.procrustes()` \cite{Wang2021}. Krumhansl advierte que proyecciones 2D pueden ser engañosas: acordes visualmente cercanos en 2D pueden estar lejos en la solución completa 4D \cite{Krumhansl1990}. Métricas recientes como el **coeficiente Saturn** evalúan estructura local+global simultáneamente \cite{Chicco2026}. La inicialización espectral de UMAP (por defecto) captura mejor la estructura global que la inicialización aleatoria de t-SNE. Para ChordSpace: (1) documentar que MDS es el método principal (distancias globales significativas), (2) UMAP/t-SNE son complementarios (estructura local/clusters), (3) reportar trustworthiness+continuity para cada método, (4) usar Procrustes para medir estabilidad entre ejecuciones.
    <!-- FIN REDACTADO: Q-028 -->

**Debilidades a vigilar:**
- Riesgo de sobreinterpretar embeddings de t-SNE/UMAP (las distancias entre clusters no son informativas).
- Documentar explícitamente que MDS es el método principal; UMAP/t-SNE/ISOMAP son complementarios.

#### 3.5.3 Protocolo de Reproducibilidad: Semillas y Determinismo

**Contenido:**
- Modo determinista: `n_jobs=1`, seed fija, `mds_n_init=4`.
- Estudio de selección de semilla: evaluación de múltiples seeds sobre población de referencia; selección de seed 17 por estabilidad. (Nota de santimath:  no lo sé , pero me parece una pregunta importante y tecnica y deberiamos pensar en como resolverlo y tener refernecias cortas sobre el tema de la seleccion de semilla porque no tengo idea de eso)
- Modo paralelo: permite `n_jobs > 1` pero no garantiza reproducibilidad bit-a-bit.

---

### 3.6 Diseño Experimental

*Motivación: Conectar las definiciones formales (§3.1–3.5) con un protocolo de evaluación sistemática que responda a los objetivos de investigación.*

#### 3.6.1 Preguntas Experimentales

Cada pregunta se vincula a un objetivo específico de la tesis:

| Pregunta | Objetivo | Sección de respuesta |
|---|---|---|
| P1: ¿Qué combinación (normalización × métrica × reductor) produce el embedding más fiel? | OE2 + OE3 | §3.7 (métricas cuantitativas) |
| P2: ¿El embedding discrimina entre tipos de acordes (por cardinalidad)? | OE3 | §3.7 (silhouette, Davies-Bouldin) |
| P3: ¿El modelo preserva relaciones perceptuales conocidas (e.g., consonancia < disonancia en rugosidad)? | OE1 | §3.2.5 + §3.7 |
| P4: ¿Cómo afecta la inclusión de inversiones musicales a la estructura del espacio? | OE1 | Experimentos de inversiones |

(Nota de santimath: Yo creo que aca faltan mas preguntas, que pasa si tengo dos acorde scasi iguales pero le agrego una nta arriba o abajo o n el medio que pasa con ese tipo demovimeintos en nuestro espacio, o que apsa is enteinteamos emular otros movimeintos de la teoria calsica de la armonia ennuetros espcaio cuales expiermtnos hacer ? esto es una nueva familai de expeirmtinacion? donde quedan los experitnos sobre el barroco que escirbismo jaja eso fue graciosos lo propuesse sin saber mucho y ahora sigo sin saber nada del barroco imagino que pensaaba enque  ciertos tipos de intervalos musicales se usabvan con muy poca freucenai por asuntos culkturales porejemplo acorde napolitnano y otros que no se suabancomo expeirmtnar en eso ?)

**Preguntas clave:**
- ¿Las preguntas son falsificables? ¿Qué resultado descartaría la hipótesis?
- Si TODAS las normalizaciones producen embeddings de baja calidad → la representación $\Phi$ en $\mathbb{R}^{12}$ podría ser inadecuada.

(Nota de santimath:  como antes quiza aca faltan preguntas y hay que dejar claro donde se le va a explciar el lenguaje muscial al matemático aca. en este capiutlo? en otro? que seria lo correccto segun las rubricas de modelamiento y los estandares en tesiss de posgrado ?)

#### 3.6.2 Factores Controlables y sus Niveles

**Tabla de factores:**

| Factor | Niveles | Cantidad |
|---|---|---|
| Normalización | identity, simplex, simplex_sqrt, simplex_smooth, perclass_{0.25, 0.5, 0.75, 1.0}, global_pairs, divide_mminus1 | 10 |
| Métrica | cosine, euclidean, manhattan, js, hellinger | 5 |
| Reductor | MDS, UMAP, TSNE, ISOMAP | 4 |
| Semilla | Lista variable (estudio: múltiples; producción: seed 17) | 1–N |
| Población | Definida por consulta SQL o generación combinatoria | Variable |

**Escenario** = (normalización, métrica). **Configuración completa** = (escenario, reductor, seed, población).

**Observación:** El número total de combinaciones es $10 \times 5 \times 4 = 200$ escenarios por población y seed. No todos son significativos (e.g., JSD solo aplica a vectores en el simplex).

(Nota de santimath: naaah 200 escenarios?? no creo que haya que hacer todo eso en el codigo hay que arguemtanr de alguna manera para no recisar todo eso.)

**Preguntas clave:**
- ¿Hay combinaciones inválidas? Sí: métricas de distribución (js, hellinger) con normalización `identity` requieren normalización L1 previa (documentar cómo el pipeline lo maneja). (Nota de santimath: hay que robustecer estos criterios para descartar)
- ¿Es necesario corregir por comparaciones múltiples al comparar 200 escenarios?

#### 3.6.3 Poblaciones de Experimentación

**Contenido:**
- **Poblaciones de referencia (controladas):** Díadas canónicas ($N = 13$), tríadas core ($N \approx 21$), catálogos nombrados.
(Nota de santimath: porque 13? no son 12 diadas? que es la 13ava que se incluye?)
- **Poblaciones intermedias:** Subconjuntos aleatorios controlados por consulta SQL ($N \in [100, 5000]$).
- **Poblaciones masivas:** ChordCodex completo o submuestras grandes ($N \sim 10^4$–$10^5$).
- Justificación de la progresión: validar primero en casos conocidos → escalar gradualmente.

(Nota de santimath: realmente el enfoque de la base de datos no es tan buena nose, pero ultimamnete las pruebas que hice las hice con generacion combinatorial a punta de filtros y demás cosas, hay una huerisitica para identifica acordes comunes dentro de la teoria armonica clasica, basada en leer sus intervalos de formacion, no se si hablamos de eso pero bueno basado en los iinteralos uno podia ponerle nombre al acorde, por lo menos de manera basica sabemos que hay toda una linea en la musc tech al rededor de la nomencalcura de acordes pero eso no nos parece tan relevante hoy dia mas cuando tenemos articulos que dicen que pues esa teoria es subjetiva un poco al rededor de lo cultural, el entrenamiento musical, edad epoca, entre otros notebook deberia ayudar a redactar aca ya refernciar)

#### 3.6.4 Experimentos de Referencia Implementados

**Contenido:**
1. **Golden Master:** Pipeline completo en caso pequeño determinista (díadas, simplex + identity, MDS, seed 42). Propósito: verificación de corrección durante refactorización.
2. **Estudio de selección de semilla:** Múltiples seeds sobre población extendida; selección de seed 17.
3. **Estudios de inversiones:** Poblaciones A/B/C (original, inversiones musicales, inversiones estructurales) comparando MDS vs UMAP.

( Nota de santimath: acá definitivamente nos falta revisar la bibliogarfia proponer experimentos basados en teoria de la armonia clasica ver donde quedan esos acordes experimetnos creativos y sigificativos para el usuario musico asi como metodologia para la evaluacion, aca siento que falta trabajo)

**Preguntas clave:**
- ¿Los experimentos implementados cubren suficientemente los objetivos de la tesis?
(Nota de santimath: si, esa es una buena pregunta tampoco nos podemos poner locos hay que revisar la literatura y escoger buenos pocos y representativos)
- ¿Hay gaps experimentales? (El Exp1–Exp6 planteado originalmente no está implementado como tal; documentar honestamente). 

---

### 3.7 Marco de Evaluación y Métricas de Calidad

*Motivación: ¿Cómo sabemos si el embedding es "bueno"? Se necesitan criterios cuantitativos vinculados a las preguntas experimentales.*

#### 3.7.1 Métricas de Preservación de Vecindarios

**Definición 3.18 (Trustworthiness — Kaski et al., 2003):** Mide cuántos puntos aparecen como vecinos en el embedding pero NO lo eran en el espacio original (falsos vecinos):
$$T(k) = 1 - \frac{2}{Nk(2N - 3k - 1)} \sum_{i=1}^{N} \sum_{j \in U_k(i)} (\hat{r}(i,j) - k)$$
donde $U_k(i)$ son los $k$-vecinos en el embedding que no están entre los $k$-vecinos originales, y $\hat{r}(i,j)$ es el rango de $j$ respecto a $i$ en el espacio original.

(Nota de santimath: aca hay que revisar la notacion y que sea consistentente y ver si nos faltan definciones previas)

**Definición 3.19 (Continuity):** Mide cuántos vecinos originales se "pierden" en el embedding (análogo dual de trustworthiness).
(Nota de santimath:Y? como se define? no esta escrito, además todas estas hay que justificar su eleccion con base en teoria quiza en trabajos que no son de musucia pero si de ingenieria de caracterisitcas no? aca toca que revises bien)

**Definición 3.20 (kNN Recall):** Fracción de los $k$ vecinos más cercanos en el espacio original que permanecen como $k$ vecinos en el embedding. 
(Nota de santimath: falta detallee y cosas aca tambien)

**Parámetro:** $k = 3$ (fijado en `config.py` como `EVAL_N_NEIGHBORS`).

**Preguntas clave:**
- ¿Por qué $k = 3$? ¿Es sensible la evaluación a este parámetro? (Discutir en §3.8).
- ¿Trustworthiness y continuity son redundantes o complementarios?
(Nota de santimath: acaso no hay mas ? revisar repo y bilbiografia)

#### 3.7.2 Métricas de Correlación Global

**Definición 3.21 (Correlación de rangos de Spearman):** Mide la correlación monótona entre las distancias originales $D_{ij}$ y las distancias en el embedding $\hat{D}_{ij}$. 
(Nota de santimath: creo que como en las de arriba falta detalle y deficnion aca recuerda que el lector es matemático)

**Definición 3.22 (Diagrama de Shepard):** Gráfica de dispersión $(D_{ij}, \hat{D}_{ij})$ para todos los pares. Se ajusta una regresión lineal y se reportan: pendiente, intercepto, $R^2$.

**Contenido:**
- Límite de muestreo: `MAX_SHEPARD_PAIRS = 20000` para evitar costo cuadrático.
- Interpretación: $R^2$ alto indica buena preservación global de distancias.

(Nota de santimath: en esta sub seccion faltan muchas cosas )

#### 3.7.3 Métricas de Clustering (sobre etiquetas de cardinalidad)

**Definición 3.23 (Silhouette Score):** Mide cohesión intra-cluster vs separación inter-cluster usando la cardinalidad del acorde como etiqueta.

**Definición 3.24 (Davies-Bouldin Index):** Promedio de la ratio entre dispersión intra-cluster y distancia inter-centroide. Menor es mejor.

**Contenido:**
- Justificación: si el embedding es bueno, acordes de la misma cardinalidad deberían formar grupos cohesivos (hipótesis testeable).
- Limitación: la cardinalidad es una etiqueta proxy, no necesariamente la estructura más relevante.

#### 3.7.4 Stress de Kruskal (ya definido en §3.5.1)

Referencia cruzada a la definición 3.17. Aquí se usa como métrica de evaluación post-hoc, no solo como función objetivo del MDS.

#### 3.7.5 Protocolo de Agregación Multi-semilla

**Contenido:**
- Para cada escenario, se ejecutan múltiples seeds y se reporta media $\pm$ desviación estándar de cada métrica.
- Esto permite distinguir variabilidad intrínseca del método (e.g., UMAP es más variable que MDS) de la calidad real del escenario.

(Nota de santimath: aca tambien falta mas detalle

una de las metricas que teniamos muy vista era la "varianza" digamos que como forte y otros enfoques hacen relaciones de equivlanetencia entre acordes por razones de la teoria armonica entonces pues digamos que esos acordes son mapeados por esa metodologia al mismo punto, a su clase, pero en nueentro engoque estrucutural y perecptual,pues las inversiones musicales, se difenrician un poco y tienen disnaicia positiva o variniaza psitiva la verdad no se cual esocger ahi, en el repo creo que eso no esta implementado pero havra que ver como lo hacemos rapido pero eso es importante reportar que nuestra metodlogia separa acordes que aunque familires la oreja detecta como disintitos eso es un arugmento fuerte en mi intiuicicon y me gustaria matemámticamente mostrarlo en la expeirmeitnacion y en la defincion de esos criterios de evaluacion)
---

### 3.8 Supuestos, Límites y Reflexión Metodológica

*Motivación: Cierre de la rúbrica de modelamiento. El lector debe saber exactamente qué se asume, qué no se puede hacer, y qué decisiones podrían cambiar los resultados.*

#### 3.8.1 Supuestos del Modelo Psicoacústico

- Timbre genérico (espectro armónico con decaimiento exponencial; no específico de instrumento).
- Sistema 12-TET (excluye microtonalidad y afinación justa).
- Parametrización fija de Sethares ($H=6$, $\delta=0.88$, constantes numéricas). No se realiza análisis de sensibilidad paramétrica en este trabajo.
( Nota de santimath: creo que si se hizo algo de esto en el repo, y si no, creo que podriamos hacerlo rapido tambien asi que no descartarlo y programar esa tarea para el repo)

- Rugosidad como proxy principal de "similitud sonora" (omite brightness, spectral centroid, attack transients, etc.).

(Nota de santimath, no se si esas deficiones se den en este capitulo, yo las domino muscalmente muy por encima pero si las mencionamos habra que explicarlas en algun lugar, ojala muy rapido)

#### 3.8.2 Supuestos de la Representación

- Acorde como tupla discreta de notas MIDI (no frecuencias continuas; no bending).
- Representación en 12 bins sin colapsar complementarios (decisión justificada en §3.1.3–3.1.4).
- Poblaciones ancladas en raíz `0` atenúan efectos de registro pero no los eliminan.

(Nota de santimath: decir que si es posible explorar con variacion en la altura, pero no es tan relevante, nos dimos cuenta que lo relebante son las estructuras y la rugsidad de los intervalos internos , es decir fijando una nota bajo )

#### 3.8.3 Límites Computacionales

- Matriz de distancias: $O(N^2)$ → límite práctico $N \approx 10^4$ en hardware estándar.
( Nota de santimath: hay que mostrar como justificamos esto)
- MDS-SMACOF: $O(N^3)$ → cuello de botella para $N > 5000$.
- Plan documentado (no implementado): Landmark-MDS, k-NN aproximado para UMAP.

( Nota de santimath: defitincamente a esta subsección le falta detalle y considraciones de la bibliografia asi como protocolos de experimentacion)

#### 3.8.4 Decisiones Críticas de Diseño

1. **Convención de bins:** Intervalo 0 → índice 11. Afecta comparabilidad con IC de Forte. (Nota de santimath: Es que el intervalo cero para nosotors es la distanci octava porque como no hay unsisono, entonces la primera posiscion del vector correpsonde al tipo de intervalo de 2m segnda menor, es decir 1 semitono, y la posiicion 12 del vector indice 11 pues correposnde a la distacia 12 semitono es edecir octava aun asi peela con FORTE? bueno y eso es grave? lo podemos justificar de alguna manera? que dicen las rerenrencias? )
2. **Modo determinista:** `n_jobs=1`, `mds_n_init=4`, seed fija. Prioriza reproducibilidad sobre velocidad. (Nota de santimath: la verdad nose mucho sobre esto pero siento que falta detalle y definciones para poder hablar de esto, en realdiad es relevante mencionarlo? que dice la literatura?)
3. **Muestreo Shepard:** Máximo 20,000 pares. Suficiente estadísticamente pero no exhaustivo.

#### 3.8.5 Amenazas a la Validez

**Validez interna:**
- Correlación entre propuestas: ¿propuestas similares (e.g., simplex y simplex_sqrt) inflan artificialmente el número de "buenos" escenarios?
- Sensibilidad a $k=3$ en métricas de vecindario.
( Nota de santimath: yo creo que aca faltan cosas, revisar repo y refenrecias)
**Validez externa:**
- No se valida con juicio humano (músicos evaluando similitud percibida). El modelo es intrínseco.
- Generalización a otros sistemas de afinación o timbres no está cubierta.

**Validez de constructo:**
- ¿La rugosidad de Sethares realmente captura "similitud sonora"? Es un proxy, no una medida directa.( Nota de santimath: hay mcuhas refernecia para jsutificar estas cosas)

---

### 3.9 Estrategia de Reproducibilidad

**Contenido breve:**
- Repositorio versionado con código, configuración y dependencias.
- Salidas con timestamp.
- Protocolo: preparar ChordCodex → seleccionar población → configurar escenario → modo determinista → recolectar artefactos.
- GUI como canal alternativo con trazabilidad equivalente (exporta `config.json`).
( Nota de santimath relamente es muuuuy poco lo que tenemos que habar de chrcodex, es limitado, preferible quitar esas meenciones quiza mencioar las colcumnas que tomamos al extraer caracteriristcas del acorde y la posibilidad en el flujo de tener una basde dedatos de los experimentos de exploracion pero la GUI es lo estandar ahora de manera combinatorial)
---

## EVALUACIÓN ADVERSARIAL (Scientific-Critical-Thinking)

### Ronda 1: Detección de Debilidades

| # | Debilidad detectada | Severidad | Sección afectada | Acción |
|---|---|---|---|---|
| W1 | La sección 3.3 (normalización) no discute si las propuestas son algebraicamente independientes o si hay redundancias. | Media | §3.3.2 | Añadir Observación sobre correlación potencial entre propuestas y cómo el diseño experimental lo aborda. |
| W2 | No se menciona corrección por comparaciones múltiples al evaluar 200 escenarios. | Alta | §3.6.2 | Añadir nota en §3.6.2 sobre el problema y la estrategia (ranking relativo + multi-seed como mitigación). |
| W3 | La Proposición 3.1 afirma "no equivalencia con PC-sets" pero no da una demostración formal. | Baja | §3.1.3 | Es una observación constructiva, no un teorema; aclarar que es por diseño, no por demostración. |
| W4 | El paso de $\Phi_{\text{raw}}$ a distancias omite la justificación de POR QUÉ la rugosidad por clase de intervalo es una buena representación para medir similitud. | Alta | §3.2–3.3 | Añadir un párrafo de motivación al inicio de §3.3 conectando la hipótesis de trabajo con la representación. |
| W5 | §3.7 no define un criterio de "éxito" numérico (e.g., "trustworthiness > 0.9 se considera aceptable"). | Media | §3.7 | Añadir umbrales de referencia de la literatura (Venna & Kaski, 2006) como guía, no como criterios rígidos. |
| W6 | No hay discusión de la estabilidad de los embeddings respecto a perturbaciones en la población (robustez). | Media | §3.8.5 | Añadir como amenaza a la validez; el estudio de seeds mitiga parcialmente pero no aborda perturbación de datos. |
| W7 | El capítulo no conecta explícitamente los resultados esperados del embedding con el concepto de "sustitución armónica" del título de la tesis. | Alta | §3.0 / §3.6.1 | Añadir nota: la sustitución se operacionaliza como vecindad en el espacio métrico; la calidad del embedding determina la utilidad del espacio para esta tarea. |


(Nota de santimath: Pero si es cierto hay diseños o al menos ideas de isñeos de formas de buscarsusutoituis o sismilitd armonica pperceptual y eso le da valor a este trabajo  tenemos que revisar eso porque me parece relevante)


### Ronda 2: Evaluación de Hipótesis (Hypothesis-Generation Framework)

**Hipótesis implícita del capítulo:**
> "La representación $\Phi \in \mathbb{R}^{12}$ basada en rugosidad de Sethares, combinada con métricas apropiadas y reducción dimensional, produce un espacio donde acordes perceptualmente similares se ubican cerca."

**Evaluación de calidad de la hipótesis:**
| Criterio | Score (1-5) | Comentario |
|---|---|---|
| Testabilidad | 4 | Sí: se miden métricas cuantitativas sobre el embedding. |
| Falsificabilidad | 3 | Parcial: ¿qué resultado la descarta? → Stress alto + trustworthiness baja para TODAS las combinaciones. Falta hacerlo explícito. |
| Parsimonia | 4 | Un solo modelo psicoacústico, múltiples normalizaciones evaluadas. |
| Poder explicativo | 3 | Explica "estructura" pero no valida percepción humana directamente. |
| Alcance | 3 | Limitado a 12-TET, timbre genérico, sin contexto tonal. |
| Consistencia | 5 | Compatible con Plomp-Levelt, Sethares, teoría de MDS. |
| Novedad | 4 | La representación 12-D sin colapsar complementarios + comparación sistemática de normalizaciones es original. |


( Nota de santimath_: respecto a "poder explicativo" yo creo que en las referenias podremso encontrar ocmo justificar eso en que la rugosidad es un feonomoo fisilogico y no depende de la cultura o el entreameinto msucial o estoy entenididmo mal este criterio?, respecto a "alcance" de nuevo creo que la biliografia nos ayuda, en que la metoodogia podria cambair la funcion de rugosidad  de tal manera que no se limtie a ese sisntema te temperamento, y se podria permiteri robustecer para timbre y otros sinsitrumentos eso hay que darl calro jaja hayq ue revisar el repo si deberdad es posible impemetnar eso sipero yo creo que si)
**Recomendación:** Hacer explícitos los criterios de falsificación en §3.6.1.

### ScholarEval Assessment (Scholar-Evaluation Framework)

**Dimensión 3: Methodology & Research Design**

| Criterio | Score | Justificación |
|---|---|---|
| Alineamiento diseño-preguntas | 4 | Las preguntas P1-P4 se responden con el diseño experimental propuesto. |
| Rigor y validez | 4 | Métricas formales, protocolo reproducible, múltiples seeds. |
| Reproducibilidad | 5 | Semillas fijas, código versionado, artefactos con timestamp. |
| Controles | 3 | Baseline `identity` presente; falta grupo de control perceptual humano. |
| Sesgos reconocidos | 4 | §3.8 documenta amenazas a la validez; W2 (comparaciones múltiples) necesita refuerzo. |
| Limitaciones discutidas | 4 | §3.8 es explícito sobre supuestos y alcance. |

**Score Metodología: 4.0 / 5.0** — Fuerte, con mejoras menores recomendadas.

**Prioridades de mejora (ordenadas por impacto):**
1. Explicitar criterios de falsificación de la hipótesis (W7 + hipótesis framework).
2. Abordar corrección por comparaciones múltiples (W2).
3. Añadir umbrales de referencia para métricas de evaluación (W5).
4. Conectar explícitamente "calidad del embedding" → "utilidad para sustitución" (W7).
