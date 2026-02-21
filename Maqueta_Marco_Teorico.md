# Maqueta del Marco Teórico (Enfoque de Embudo Integrado)

Este documento presenta el esqueleto estructural (maqueta) del Capítulo 2: Marco Teórico. Esta versión integra la "Ley del Embudo" (desde acústica general hasta el modelo específico) fusionada con la **riqueza y el detalle de las subsecciones del marco teórico original (`02Seccion02.tex`)**, logrando la robustez esperada para una tesis de maestría en Matemáticas Aplicadas.

---

## 2. Marco Teórico y Estado del Arte

*(Ilustración sugerida al inicio: Un organigrama o diagrama de flujo que muestre cómo las propiedades físicas del sonido se traducen en representaciones matemáticas en la investigación actual).*

### 2.1 Fundamentos Acústicos y Perceptivos del Sonido Musical (Boca del embudo)
*Objetivo de la sección: Establecer las bases biológicas y físicas de la audición que justifican el uso posterior de la "rugosidad" como métrica central.*

#### 2.1.1 El Hardware Auditivo: Batimientos y Bandas Críticas
*   **Párrafo 1:** Antecedentes históricos: de las proporciones de Pitágoras a la comprensión de las interferencias de ondas `\cite{Helmholtz, 1863}`.
*   **Párrafo 2:** Fisiología básica: La cóclea, la membrana basilar y la definición de anchos de banda crítica `\cite{Plomp y Levelt, 1965}`.
*   **Párrafo 3:** Definición mecánica de la "rugosidad" (sensorial dissonance) cuando dos frecuencias chocan en la misma banda crítica. *(Ilustración sugerida: Curva original de Plomp-Levelt).*

#### 2.1.2 Consonancia, Disonancia y el Enfoque de Sethares
*   **Párrafo 1:** Transición de notas puras a sonidos complejos. El rol del *timbre* y los espectros armónicos/inarmónicos `\cite{Sethares, 1993}`.
*   **Párrafo 2:** Explicación del modelo matemático de Sethares para calcular curvas de disonancia sumando las interferencias parciales `\cite{Sethares, 2005}`.

#### 2.1.3 Procesamiento Cognitivo: Pitch Virtual y Armonicidad
*   **Párrafo 1:** El salto acústica $\to$ cerebro: La teoría del Pitch Virtual y cómo el cerebro infiere "fundamentales perdidas" `\cite{Terhardt, 1974}`.
*   **Párrafo 2:** "Pitch commonality" y la armonicidad de Parncutt: por qué los humanos agrupan un conjunto de frecuencias complejas en un solo percepto de "acorde" `\cite{Parncutt, 1988}`.

#### 2.1.4 Naturaleza vs. Crianza: Rugosidad vs. Preferencia Cultural
*   **Párrafo 1:** Distinción crucial: La aversión a la rugosidad acústica (sensorial dissonance) es un fenómeno fisiológico universal (bottom-up).
*   **Párrafo 2:** En contraste, la "consonancia" entendida como preferencia estética o estabilidad armónica, está fuertemente dictada por la exposición cultural (top-down) a la polifonía occidental, como demuestran los estudios transculturales `\cite{McDermott et al., 2016}`. Esto justifica modelar la "rugosidad" métrica, ya que provee un modelo más objetivo y generalizable que la teoría tonal clásica.

### 2.2 Teoría Musical y la Evaluación del Acorde Aislado (Medio Superior)
*Objetivo de la sección: Aterrizar los fenómenos perceptivos en constructos musicales, separando expresamente la sonoridad intrínseca de los acordes de su función en la sintaxis progresiva.*

#### 2.2.1 Intervalos, Acordes y Conjuntos de Clases de Altura
*   **Párrafo 1:** Definición formal de la teoría matemática de la música (PC-sets, clases de tono, equivalencia de octava y enarmonía).
*   **Párrafo 2:** Impacto numérico y combinatorio de las inversiones y el *voicing* en la estructura del acorde.

#### 2.2.2 Dualidad: Tensión Vertical vs. Sintaxis Horizontal
*   **Párrafo 1:** Diferenciación fundamental entre la "identidad acústica aislada" (vertical) y el "voice leading o progresión armónica" (horizontal).
*   **Párrafo 2:** Teoría de Meyer: La tensión intrínseca vertical (ej. equidistancia en el acorde aumentado) propensión biológica a la resolución horizontal. Aislar el análisis vertical ayuda a explicar el origen causal de las reglas horizontales `\cite{Meyer, 1956}`.
*   **Párrafo 3:** Justificación teórica para modelos abstractos (bottom-up) frente a sesgos occidentales tonales (top-down), habilitando simulaciones compositivas imparciales.

### 2.3 Representación Computacional y Modelos de Aprendizaje Automático (Medio Inferior)
*Objetivo de la sección: Examinar cómo se ha codificado tecnológicamente la armonía, contrastando la tendencia moderna de Machine Learning con la necesidad de rigor matemático explicable.*

#### 2.3.1 Representaciones Simbólicas y Señales
*   **Párrafo 1:** Diferencias entre formatos de entrada: partituras simbólicas, MIDI (matrices limitadas) y señales de audio puro (Limitaciones en la traducción computacional).
*   **Párrafo 2:** Transformaciones de señal (Fourier, STFT, CQT) y cómo filtran el espectro para el análisis `\cite{Muller}`.
*   **Párrafo 3:** Codificación vectorizada de acordes: Vectores booleanos (chroma de 12 dimensiones) vs representaciones densas.

#### 2.3.2 Modelos de Machine Learning: Alcances y Opacidad
*   **Párrafo 1:** Adaptación de embeddings de lenguaje natural a música (Word2Vec $\to$ Chord2Vec) `\cite{Huang, 2016}`.
*   **Párrafo 2:** Fortalezas: Procesamiento de gigantescos corpus de partituras (capturan "quién acompaña a quién").
*   **Párrafo 3:** La "Caja Negra" estadística: Un vector latente es opaco. Dos acordes pueden agruparse por coincidencia distribucional sin tener parentesco acústico subyacente, dificultando la "explicabilidad" metodológica para un análisis topológico riguroso `\cite{InterpretableML_Music}`.

### 2.4 Espacios Geométricos y Exploración Armónica (Fondo del embudo)
*Objetivo de la sección: Presentar la solución analítica: los espacios matemáticos que permiten mapear la cercanía de los acordes transparentemente.*

#### 2.4.1 Geometrías Musicales y Topología
*   **Párrafo 1:** Genealogía de los espacios espaciales paramétricos continuos: del *Tonnetz* de Euler a los hiper-grafos abstractos `\cite{euler1739tentamen, Tymoczko}`.
*   **Párrafo 2:** Ejemplos clásicos: *Tonal Pitch Space* (TPS) de Lerdahl y variables explícitas (distancias auditables) `\cite{Lerdahl, 1988}`.
*   **Párrafo 3:** La Topología de Redes de Marco Buongiorno Nardelli: Unificación de conjuntos combinatorios de clases de tono y distancias en subespacios de invariancia `\cite{Nardelli}`.

#### 2.4.2 Visualización de Tonalidad (Ej. Spiral Array)
*   **Párrafo 1:** El modelo espiral (Spiral Array) de Chew como ejemplo de un marco espacial que jerarquiza tonos, acordes y tonalidades `\cite{chew2014mathematical}`.

#### 2.4.3 El enfoque de Vectores de Clases de Intervalo Dirigidos (DIC)
*   **Párrafo 1:** Uso de perfiles interválicos (contar segundas mayores, terceras menores, etc., dentro y entre acordes) para capturar tanto la afinidad acústica como la parsimonia de conducción de voces.

### 2.5 Análisis Multidimensional y Reducción de Datos (Puente a la Metodología)
*Objetivo de la sección: Introducir formalmente las herramientas matemáticas que se explotarán en ChordSpace para medir y visualizar la semejanza de acordes.*

#### 2.5.1 Reducción de Dimensionalidad Computacional
*   **Párrafo 1:** Por qué la alta dimensionalidad (cientos de acordes evaluados acústicamente) dificulta la interpretación visual humana.
*   **Párrafo 2:** Breve panorama comparativo: MDS (para conservar distancias globales euclídeas), PCA (varianza lineal), t-SNE (vecindarios locales no lineales) y UMAP (preservación métrica topológica) `\cite{McInnes, 2018}`.

#### 2.5.2 Métricas de Validación Armónica
*   **Párrafo 1:** Cuantificación del error computacional (Stress en MDS, Trustworthiness/Continuity en UMAP).
*   **Párrafo 2:** Correlación entre métricas puramente matemáticas (distancia euclidiana) vs métricas perceptivas (curvas empíricas de consonancia).

### 2.6 Síntesis: El Diseño de ChordSpace (Salida del embudo)
*Objetivo de la sección: Conclusión del capítulo que consolida la revisión de la literatura y perfila de forma inequívoca el nacimiento del modelo de la Tesis.*
*   **Párrafo 1:** Resumen del problema principal: Las herramientas actuales o son altamente abstractas (solo mates) o inescrutables empíricamente (Puro ML).
*   **Párrafo 2:** Planteamiento del modelo: ChordSpace cubrirá esta brecha desarrollando una matriz explícita de distancias que integra (2.1) rugosidad de Plomp-Levelt/Sethares aplicado a (2.2) acordes aislados sin sesgo tonal, proyectados algorítmicamente mediante (2.4) topología de intervalo directo y reducido iterativamente vía (2.5) MDS/UMAP, esquivando las cajas negras del ML (2.3).
