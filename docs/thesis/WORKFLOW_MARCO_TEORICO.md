# Flujo de Trabajo (Workflow) - Capítulo Marco Teórico

Este documento establece el plan operativo de **12 iteraciones** usando NotebookLM y herramientas complementarias (habilidades científicas) para construir el Capítulo 2: Marco Teórico y Estado del Arte de la tesis de maestría en matemáticas aplicadas.

**Audiencia Objetivo:** Matemáticos expertos en modelación matemática, sin conocimiento especializado previo en teoría musical o psicoacústica.

**Metodología General:** *Ley del Embudo*. Partimos de verdades psicoacústicas y fisiológicas universales para aterrizar, capa por capa, en las decisiones topológicas y asunciones matemáticas específicas de la representación en `ChordSpace`.

---

## Preparación del Entorno
Antes de comenzar, asegúrate de tener configurado el proyecto y las siguientes áreas preparadas:
1. **Directorio Objetivo:** Todos los fragmentos generados se guardarán temporalmente en `docs/marco_teorico_research/` como archivos individuales `.md` (ej. `iteracion_1_acustica.md`). Las versiones pulidas se integrarán luego a la carpeta de LaTeX de la tesis.
2. **Bibliografía:** El archivo `docs/thesis/referencias_metodologia_vf.bib` actuará como *single source of truth*. Cada cita generada se validará con `citation-management` o `research-lookup`.
3. **NotebookLM:** Crea un nuevo Notebook específico ("Marco Teórico Tesis Matemáticas") y carga:
   - Los documentos antiguos de la tesis (marco teórico, maqueta, estructura matemática detallada).
   - *Papers* clave: Sethares (1993, 2005), Plomp & Levelt (1965), McInnes (2018), Tymoczko (2006, 2011), Harrison & Pearce (2020), Cook (2009).

---

## Fase 1: Bases Físico-Matemáticas (Boca del Embudo)

**Objetivo:** Establecer que la base del trabajo es biológica y física (bottom-up), no una convención cultural.

### Iteración 1: Física Acústica y Series de Fourier
*   **Prompt a NotebookLM:** "Actúa como un matemático escribiendo para otros matemáticos. Explica sucintamente pero con rigor la descomposición de un sonido complejo en componentes espectrales (Series de Fourier) y cómo esto se relaciona con los timbres armónicos. Evita explicaciones para niños. Aporta fórmulas clave si es necesario y enlázalo con referencias del Notebook."
*   **Acción de la IA (Redacción):** Escribir `iteracion_1_fourier_acustica.md` usando `scientific-writing`.

### Iteración 2: Fisiología de la Cóclea y Ancho de Banda Crítico (ERB)
*   **Prompt a NotebookLM:** "A partir del trabajo de Plomp y Levelt, y su base fisiológica en la cóclea, define el concepto de 'Ancho de Banda Crítico' (Critical Bandwidth). Explica matemáticamente por qué percibimos las frecuencias logarítmicamente (cents, semitonos) pero las interferencias auditivas ocurren a nivel frecuencial lineal local. Proporciona citas precisas."
*   **Acción de la IA (Recursos):** Usar `scientific-schematics` para solicitar/generar una gráfica vectorial de la escala ERB o de la respuesta de la membrana basilar.

### Iteración 3: Batimientos y Disonancia Sensorial
*   **Prompt a NotebookLM:** "Sintetiza de forma rigurosa la relación entre la frecuencia de batimiento de dos tonos cercanos y la percepción de 'Rugosidad' (Disonancia Sensorial). Detalla la curva característica donde la máxima disonancia ocurre alrededor del 25% del ancho de banda crítico. Diferencia explícitamente entre 'Disonancia Sensorial' (fisiológica) y 'Disonancia Musical' (cultural)."
*   **Acción de la IA (Recursos):** Integrar la imagen generada previamente (`sethares_curva_gemini.png` o equivalente) y redactar `iteracion_3_batimientos_rugosidad.md`.

---

## Fase 2: El Modelo Computacional de Rugosidad

**Objetivo:** Formalizar la métrica principal que usará ChordSpace.

### Iteración 4: Parametrización de Sethares
*   **Prompt a NotebookLM:** "Describe matemáticamente la función de curva de disonancia paramétrica de Sethares (1993) $d(x) = e^{-ax} - e^{-bx}$. Menciona cómo sus parámetros se ajustaron empíricamente minimizando el error cuadrático medio sobre los datos de Plomp y Levelt. Escribe las ecuaciones completas."
*   **Acción de la IA (Redacción):** Escribir `iteracion_4_modelo_sethares.md`. Asegurar formato LaTeX riguroso de ecuaciones.

### Iteración 5: Extensión a Tonos Complejos y Comparación
*   **Prompt a NotebookLM:** "Explica cómo se extiende el modelo de Sethares iterando sobre la suma par-a-par de todos los parciales de espectros armónicos con decaimiento temporal fijo (por ejemplo $H=6$ y $\delta=0.88$). Elabora una tabla comparativa breve y rigurosa sobre Sethares vs. Otras alternativas del estado del arte (Vassilakis, Hutchinson-Knopoff), detallando pro y contras (e.g. la omisión de factores como la armonicidad/compacidad por Masina 2024)."
*   **Acción de la IA (Edición):** Usar `scientific-writing` para estructurar la tabla en formato LaTeX (paquete `booktabs` preferiblemente). Escribir `iteracion_5_metologias_comparadas.md`.

---

## Fase 3: Formalización Combinatoria y Computacional

**Objetivo:** Justificar la representación matemática adoptada e introducir el problema de las dimensiones.

### Iteración 6: El Espacio Discreto de Acordes $\mathcal{A}$
*   **Prompt a NotebookLM:** "Define formalmente el Acorde como una tupla estrictamente ordenada en nuestro dominio musical discreto, enfatizando matemáticamente nuestra decisión de descartar la equivalencia de octava y de transposición típica de los PC-sets. Justifica esto apoyándote en la sensibilidad al registro del ancho de banda crítico, citando a Harrison & Pearce y a Eerola."
*   **Acción de la IA (Redacción):** Escribir `iteracion_6_espacio_discreto_A.md`.

### Iteración 7: Inyección a $\mathbb{R}^{12}$ (El vector Croma Raw)
*   **Prompt a NotebookLM:** "Explica la construcción del histograma/vector `dic` o vector `raw` de $\mathbb{R}^{12}$ que agrupa la suma total de rugosidades por cada clase de intervalo dirigido. Justifica matemáticamente (y apoyado por la bio-acústica) por qué se rechaza la equivalencia de inversión (OPT vs OPTI space de Tymoczko/Callender), para evitar el colapso de intervalos complementarios (ej. 3st y 9st), contrastando contra el approach de Forte."
*   **Acción de la IA:** Escribir `iteracion_7_inyeccion_12d.md`.

### Iteración 8: Embeddings, Deep Learning y la Crisis de Explicabilidad
*   **Prompt a NotebookLM:** "Haz un balance y contraste crítico entre el enfoque topológico puro desarrollado hasta ahora versus la tendencia dominante de caja negra de NLP aplicada a música (Machine Learning, Word2Vec, Chord2Vec, Transformers). Resalta la debilidad de carecer de una noción isomórfica-perceptual intrínseca en arquitecturas complejas de parámetros latentes, para convencer al matemático de que es más riguroso un modelo de disonancias exactas paramétricas."
*   **Acción de la IA:** Redactar y usar la sintaxis de *alertas* (*warning*) en Markdown, si conviene, para preparar `iteracion_8_estadistica_vs_topologia.md`.

---

## Fase 4: Reducción Dimensional y Disposición Topológica

**Objetivo:** Llevar el vector en 12 dimensiones a un espacio euclídeo o topológico comprensible 2D. 

### Iteración 9: Transformaciones Composicionales (Simplex)
*   **Prompt a NotebookLM:** "Al incrementar drásticamente la dimensión con la cardinalidad (acordes de más notas), la energía de los vectores en $\mathbb{R}^{12}$ explota. Explica metodologías formales para normalizar esto sin afectar las propiedades analíticas. Detalla la justificación de proyectar el volumen al Simplex $\Delta^{11}$ unitario y en qué situaciones conviene suavizado Gaussiano para imitar tolerancias cognitivas a la percepción categórica de intervalos."
*   **Acción de la IA:** Redacción en `iteracion_9_normalizaciones.md`.

### Iteración 10: Métricas Asimétricas sobre Probabilidades
*   **Prompt a NotebookLM:** "Compara y justifica el uso de distancias para variables probabilísticas frente a métricas euclídeas estándar en el contexto del Simplex. Detalla la distancia de Jensen-Shannon (JSD) y la divergencia Hellinger, explicando matemáticamente cómo abordan el problema de entropía y por qué sus raíces generan métricas válidas."
*   **Acción de la IA:** Redactar en `iteracion_10_jsd_hellinger.md`.

### Iteración 11: Mapeo y Reducción Global vs Local (MDS / SMACOF)
*   **Prompt a NotebookLM:** "Sintetiza la adaptación de algoritmos de optimización (SMACOF) para el Escalamiento Multidimensional (MDS). Relaciona esto estrictamente desde una matriz simétrica de disimilitudes, y cómo el Kruskal Stress define formalmente el fracaso o éxito topológico. Aclara bajo qué principios se usaría esta técnica."
*   **Acción de la IA:** Escribir `iteracion_11_smacof_mds.md`.

### Iteración 12: Embeddings No Lineales (UMAP) y Topología Comparada
*   **Prompt a NotebookLM:** "Finalmente, contrasta y debate críticamente el uso del Uniform Manifold Approximation and Projection (UMAP) como contraparte o suplemento. Habla de la confianza (Trustworthiness) como función de la distorsión del manifold local contra el esparcimiento global. Expón sus ventajas algorítmicas al simular los complejos simpliciales latentes que emulan a los vecinos continuos del pitch discreto musical."
*   **Acción de la IA:** Redactar `iteracion_12_umap_topologia_conclusiones.md`. Asegurar cierre que conecte como un puente para el capítulo de Metodología Computacional.

---

## Fin de Iteraciones y Consolidación Tesis (`.tex`)
*   Reunir los 12 fragmentos.
*   Con el perfil de `redactor-critico` activado, estructurar y unificar coherencia, eliminando redundancias inter-secciones para crear un borrador del capítulo.
*   Convertir a fichero `.tex` en `docs/thesis/capitulo_marco_teorico_new.tex`.
*   Asegurar que todas las fórmulas estén alineadas al preámbulo existente de la tesis ChordSpace.
*   Correr scripts locales de chequeo (`verify_citations.py`, etc.) para asegurar control orto-tipográfico del `.bib`.
