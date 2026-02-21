# INFORME DE AVANCE DE ESCRITURA DE TESIS
## Maestría en Matemáticas Aplicadas — Universidad Nacional de Colombia
### Fecha: 18 de febrero de 2026

---

## 1. CONTEXTO GENERAL DEL PROYECTO

### 1.1 De qué trata esta tesis

Esta tesis propone un **modelo computacional para la construcción de un espacio de representación y sustitución armónica** de acordes musicales. La idea central es:

> Dado un acorde musical cualquiera, ¿es posible construir un espacio matemático donde acordes que "suenan similar" queden ubicados cerca, y así descubrir sustitutos armónicos que la teoría musical tradicional no contempla?

El trabajo vive en la intersección de tres campos: **teoría de grupos y espacios métricos** (matemática), **psicoacústica** (modelo de rugosidad de Sethares/Plomp-Levelt), y **reducción dimensional** (MDS, UMAP). NO es una tesis de ingeniería de software: el repositorio ChordSpace es la realización computacional del modelo, pero el documento de tesis debe presentar la formalización matemática con rigor de posgrado.

### 1.2 Pregunta de investigación

> ¿Es posible construir un espacio de representación para acordes que permita ubicar cerca aquellos posibles sustitutos por sonoridad similar?

### 1.3 Hipótesis de trabajo

Una representación computacional que preserve la estructura interválica real (12 clases de intervalo, sin colapsar complementarios como hace Forte) e incorpore métricas perceptuales como la rugosidad de Sethares, produce un espacio donde acordes perceptualmente similares se ubican cerca — superando las limitaciones de la teoría de PC-sets.

### 1.4 Objetivo general

Desarrollar un modelo computacional para la construcción de un espacio de representación y sustitución armónica.

### 1.5 Objetivos específicos

- **OE1:** Modelar matemáticamente reglas y características para la generación y caracterización de acordes (descriptores estructurales + rugosidad).
- **OE2:** Implementar técnicas de reducción de dimensionalidad (MDS, UMAP) que preserven relaciones de similitud.
- **OE3:** Evaluar cuantitativamente la calidad del espacio resultante (trustworthiness, stress, silhouette, etc.).

---

## 2. ESTRUCTURA DEL DOCUMENTO DE TESIS

La tesis sigue la plantilla UNAL con la siguiente estructura:

| Capítulo | Título | Estado |
|----------|--------|--------|
| 1 | Planteamiento del Problema | **Borrador avanzado** (`00HipotesisPlanteamiento.tex`) |
| 2 | Marco Teórico / Estado del Arte | **Borrador previo extenso** (`02Seccion02.tex`, ~86KB) — requiere revisión y alineación con Cap. 3 |
| 3 | Metodología y Modelado Matemático | **En escritura activa** — documento maestro en `.md` (~39K tokens) + versión LaTeX previa (`03Seccion03.tex` 73KB, `metodologia.tex` 42KB) |
| 4 | Resultados | **Borrador parcial** (`04Seccion04.tex`, ~26KB) — requiere actualización con experimentos formales |
| 5 | Discusión | **Esqueleto** (`05Seccion05.tex`, ~2.6KB) |
| 6 | Conclusiones | **Pendiente** |
| 7 | Anexos | **Pendiente** |
|   | Bibliografía | **Avanzada** (`Tesis MSc UNAL.bib`, ~111KB) + 5 notebooks NotebookLM con ~243 fuentes PDF |
|   | Resumen/Abstract | **Vacío** |

---

## 3. AVANCE DETALLADO POR CAPÍTULO

### 3.1 Capítulo 1: Planteamiento del Problema (70% completado)

**Archivo:** `docs/Tesis MsC/Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024/00HipotesisPlanteamiento.tex`

**Lo que existe:**
- Planteamiento del problema y justificación redactados
- Preguntas de investigación formuladas (3 subproblemas)
- Hipótesis de trabajo escrita
- Justificación con citas (Sethares, Forte, Krumhansl, etc.)

**Lo que falta:**
- Revisión de redacción académica y coherencia
- Limpieza de artefactos de generación automática (`:contentReference` markers)
- Revisión por el asesor para validar el alcance declarado

### 3.2 Capítulo 2: Marco Teórico / Estado del Arte (40-50% completado — requiere diagnóstico)

**Lo que existe:**
- `02Seccion02.tex` — **86KB de contenido** (borrador previo extenso, fecha y estado de revisión por determinar)
- Notas de Obsidian aisladas
- Implícitamente, muchos conceptos del marco teórico están desarrollados *dentro* del capítulo de Metodología (por necesidad de la escritura bottom-up)

**ACCIÓN URGENTE:** Revisar `02Seccion02.tex` para determinar qué ya está escrito y qué falta. Es posible que muchos fundamentos ya estén cubiertos allí.

**Lo que debe estar en el marco teórico (verificar contra el .tex existente):**
El marco teórico debe preceder y fundamentar la metodología. De la escritura del Cap. 3 se identifican estos temas mínimos que DEBEN estar en el marco teórico antes de usarlos en metodología:

1. **Teoría musical básica para el lector matemático:** notas, intervalos, acordes, escalas, sistema 12-TET, protocolo MIDI
2. **Psicoacústica:** banda crítica auditiva, batimientos, rugosidad, disonancia sensorial vs. musical, modelo de Plomp-Levelt (1965)
3. **Teoría de conjuntos de clases de altura (PC-sets):** Forte (1973), vector IC, equivalencias Tn/TnI — para luego contrastar con nuestro enfoque
4. **Geometría de espacios de acordes:** marco OPTIC de Callender-Quinn-Tymoczko, orbifolds, pitch chords vs. pitch-class sets
5. **Reducción de dimensionalidad:** MDS (formulación de Kruskal), UMAP, t-SNE, métricas de calidad de embeddings
6. **Métricas de distancia en espacios probabilísticos:** JSD, Hellinger, coseno, propiedades métricas en el simplex

**Estrategia sugerida:** Extraer las definiciones y motivaciones que ya están escritas en ESTRUCTURA_MATEMATICA_DETALLADA.md y moverlas al marco teórico, dejando en el Cap. 3 solo la aplicación específica al modelo.

### 3.3 Capítulo 3: Metodología y Modelado Matemático (50-60% del contenido generado, ~25% en prosa publicable)

**Este es el foco principal de los últimos esfuerzos.** El trabajo se ha concentrado en el archivo `metodologia_temporal/ESTRUCTURA_MATEMATICA_DETALLADA.md`, que es un borrador extenso (~39,000 tokens) del capítulo de metodología siguiendo el enfoque **"Opción C+ (Híbrido Científico)"** recomendado por la evaluación de estructura.

#### Estado por sección:

| Sección | Título | Estado de contenido | Estado de redacción |
|---------|--------|--------------------|--------------------|
| §3.0 | Introducción y flujo metodológico | Esqueleto | Sin redactar |
| §3.1 | El Acorde como Objeto Matemático | **Completo** (Def 3.1–3.7, Prop 3.1) | **Borradores Q-003 a Q-011 insertados** |
| §3.2 | Modelo Psicoacústico de Rugosidad | **Completo** (Ec 3.1–3.2, Def 3.8–3.10) | **Borradores Q-012 a Q-021 insertados** |
| §3.3 | Transformaciones del Vector (Normalizaciones) | **Completo** (Tabla de 10 propuestas, Prop 3.3–3.4) | **Borradores Q-022 a Q-024 insertados** |
| §3.4 | Geometría: Métricas de Disimilitud | **Completo** (Def 3.12–3.16) | **Borradores Q-025 a Q-029 insertados** |
| §3.5 | Reducción Dimensional | **Parcial** (MDS definido, UMAP/t-SNE/ISOMAP descritos) | **Borradores Q-030 a Q-033 insertados** |
| §3.6 | Diseño Experimental | **Parcial** (factores y niveles definidos, poblaciones esbozadas) | **Borradores Q-035, Q-036 insertados** |
| §3.7 | Marco de Evaluación | **Parcial** (Def 3.18–3.24, métricas listadas) | **Borradores Q-039, Q-040 insertados** |
| §3.8 | Supuestos, Límites y Reflexión | **Parcial** (supuestos listados, amenazas esbozadas) | **Borradores Q-041, Q-042 insertados** |
| §3.9 | Estrategia de Reproducibilidad | **Esqueleto** | Sin redactar |

**Total de preguntas resueltas via workflow NotebookLM:** Q-003 a Q-042 (~30 resoluciones insertadas como bloques `<!-- REDACTADO -->` en el documento).

#### Evaluación adversarial (ya realizada sobre el borrador):

| Debilidad | Severidad | Estado |
|-----------|-----------|--------|
| W1: Independencia algebraica de normalizaciones no discutida | Media | Pendiente |
| W2: Corrección por comparaciones múltiples (200 escenarios) | Alta | **Parcialmente resuelta** (Q-036) |
| W3: Prop 3.1 sin demostración formal | Baja | Decidido: es por diseño, no teorema |
| W4: Falta justificación de por qué Φ es buena representación | Alta | Pendiente |
| W5: Sin umbrales numéricos de "éxito" en métricas | Media | Pendiente |
| W6: Estabilidad ante perturbaciones de población | Media | Pendiente |
| W7: Conexión explícita embedding → sustitución | Alta | **Parcialmente resuelta** |

#### ScholarEval Score actual del capítulo: **4.0/5.0**

---

## 4. WORKFLOW DE TRABAJO ACTUAL

### 4.1 Flujo de escritura asistida (operativo)

Se ha establecido un workflow de escritura científica con las siguientes fases:

```
ESTRUCTURA_MATEMATICA_DETALLADA.md  ←  Documento maestro (borrador del Cap. 3)
         ↑
    /redactor-critico (workflow v2)
         ↑
    respuestas_notebooklm.json  ←  Respuestas de 5 notebooks especializados
         ↑
    preguntas_CONTEXTUALIZADAS.json  ←  Preguntas extraídas del documento
         ↑
    NotebookLM MCP  ←  5 notebooks con ~243 PDFs de referencia
```

**Notebooks NotebookLM configurados:**
1. **Psicoacústica** (39 fuentes): Plomp-Levelt, Sethares, Vassilakis, Helmholtz, ERB, rugosidad
2. **Armonía** (70 fuentes): Tymoczko, Forte, Harrison-Pearce, Cambouropoulos, Lerdahl, Chew
3. **Math/Visualización** (42 fuentes): Callender-Quinn-Tymoczko, orbifolds, espacios geométricos
4. **Computación/ML** (52 fuentes): MIR, embeddings, features, algoritmos
5. **Reducción dimensional** (40 fuentes): MDS, UMAP, t-SNE, métricas de calidad

**Workflow /redactor-critico v2:** Proceso de 5 fases (Contexto Dinámico → Gate de Relevancia → Formalización → Síntesis → Redacción) con checklist de calidad y contexto acumulativo para evitar degradación en lotes.

### 4.2 Lo que ha producido este workflow

- **~30 resoluciones** tipo "bloque redactado" insertadas en el documento maestro
- Cada resolución tiene: claim central + 2-3 argumentos convergentes + citas bibliográficas verificadas en fuentes primarias
- Extensión típica: 150-250 palabras de prosa científica en español académico
- Las resoluciones cubren desde cuestiones fundamentales (¿por qué 12-TET?) hasta técnicas (¿por qué MDS métrico y no no-métrico?)

---

## 5. DIAGNÓSTICO HONESTO: QUÉ FALTA PARA TERMINAR

### 5.1 Ruta crítica (ordenada por prioridad)

#### BLOQUE A — Sin esto no hay tesis (4-6 semanas de trabajo intenso)

| # | Tarea | Depende de | Estimación esfuerzo |
|---|-------|-----------|---------------------|
| A1 | **Convertir ESTRUCTURA_MATEMATICA_DETALLADA.md a LaTeX** del Cap. 3 | Nada | Alto — requiere unificar las ~30 resoluciones, eliminar notas de autor, pulir notación, agregar ejemplos numéricos, generar figuras |
| A2 | **Escribir Cap. 2 (Marco Teórico)** con los fundamentos identificados en §3.2 de este informe | A1 (para saber qué precede) | Alto — es el capítulo que da contexto al lector matemático |
| A3 | **Diseñar y ejecutar los experimentos** del §3.6 sobre el repositorio | A1 | Alto — seleccionar 3-5 experimentos representativos, correrlos, generar figuras y tablas |
| A4 | **Escribir Cap. 4 (Resultados)** con las salidas de A3 | A3 | Medio — describir figuras, tablas, métricas |
| A5 | **Escribir Cap. 5 (Discusión)** conectando resultados con hipótesis | A4 | Medio |
| A6 | **Escribir Cap. 6 (Conclusiones)** | A5 | Bajo |
| A7 | **Escribir Resumen/Abstract** | A6 | Bajo |

#### BLOQUE B — Mejora la calidad significativamente

| # | Tarea | Impacto |
|---|-------|---------|
| B1 | Resolver debilidades W4 y W7 (justificar Φ como representación + conexión embedding→sustitución) | Alto — un evaluador lo preguntará |
| B2 | Agregar ejemplos numéricos concretos al Cap. 3 (tríada mayor C-E-G como running example) | Alto — hace legible el capítulo |
| B3 | Generar figuras clave: curva de disonancia de Sethares, diagrama de flujo metodológico, ejemplo de embedding MDS, diagrama de Shepard | Alto — un capítulo sin figuras pierde impacto |
| B4 | Implementar y documentar al menos un experimento de sustitución armónica (ya existe diseño en `substitution_metrics.md`) | Alto — conecta el modelo teórico con la aplicación declarada |
| B5 | Análisis de sensibilidad: H=4,6,8,10 armónicos; δ variable | Medio — robustece la validez |

#### BLOQUE C — Pulido final

| # | Tarea |
|---|-------|
| C1 | Verificar consistencia de notación a lo largo de todo el documento |
| C2 | Verificar que toda cita del texto tenga entrada en la bibliografía |
| C3 | Revisión de estilo académico (eliminar anglicismos, pasivos innecesarios) |
| C4 | Revisión humana por el asesor |
| C5 | Formato UNAL final (márgenes, portada, dedicatoria, etc.) |

### 5.2 Lo que NO va en la tesis (exploración descartada)

Estos elementos del repositorio fueron exploratorios y **no deben incluirse** en el documento final:

- **ChordCodex / base de datos SQL:** La generación combinatoria con filtros es el enfoque final, no la DB
- **La GUI como producto de software:** Se menciona como herramienta de exploración, no como contribución
- **Los módulos `substitution/`:** El diseño completo de `substitution_metrics.md` es un plan futuro (líneas de trabajo). Solo el perfil básico `susti_probab(JSD_Jaccard)` podría mencionarse brevemente
- **Archivos de workflow del agente** (`.agent/workflows/`): Son herramientas de proceso, no contenido
- **Scripts utilitarios** (`count_questions.py`, `create_queries_json.py`, etc.): Infraestructura de apoyo

---

## 6. EVALUACIÓN CIENTÍFICA DEL ESTADO ACTUAL

### 6.1 Fortalezas

1. **Formalización matemática sólida:** El Cap. 3 tiene definiciones, proposiciones, y notación consistente con nivel de posgrado en matemáticas
2. **Base bibliográfica extensa y verificada:** Las ~30 resoluciones del workflow NotebookLM están respaldadas por fuentes primarias (Sethares 1993, Plomp-Levelt 1965, Tymoczko 2011, Harrison-Pearce 2020, Callender-Quinn-Tymoczko 2008, etc.)
3. **Decisiones de diseño justificadas:** Cada decisión (12 bins vs 6, pitch chords vs PC-sets, MDS métrico vs no-métrico, Sethares vs alternativas) tiene argumentación con evidencia convergente
4. **Autoevaluación adversarial:** Ya se realizó una ronda de detección de debilidades con severidad y acciones propuestas
5. **Repositorio funcional:** El código implementa el pipeline completo (generación → features → distancias → reducción → evaluación → reporte)

### 6.2 Debilidades y riesgos

1. **El documento no existe en LaTeX como texto continuo:** Todo está en Markdown con bloques `<!-- REDACTADO -->`. La conversión a prosa fluida en LaTeX es un trabajo significativo
2. **Cap. 2 (Marco Teórico) existe pero necesita diagnóstico:** Hay un `02Seccion02.tex` de 86KB cuyo estado de actualización es desconocido. Debe verificarse si cubre los fundamentos que el Cap. 3 necesita (psicoacústica, PC-sets, OPTIC, MDS)
3. **Experimentos no ejecutados formalmente:** Hay código funcional y pruebas exploratorias, pero no un protocolo experimental documentado con resultados reproducibles para el documento
4. **Cap. 4 y 5 dependen completamente de los experimentos**
5. **La conexión "espacio → sustitución" es débil:** La tesis promete sustitución armónica pero lo formalizado es el espacio de representación. Hay un diseño de algoritmo de sustitución (`substitution_metrics.md`) pero no está implementado completamente ni evaluado
6. **Riesgo de alcance excesivo:** 10 normalizaciones × 5 métricas × 4 reductores = 200 combinaciones. Hay que acotar drásticamente para ser viable

### 6.3 Recomendaciones concretas para el asesor

1. **Acotar los experimentos:** Sugerir 3-5 combinaciones (normalización × métrica × reductor) fundamentadas, no las 200. Por ejemplo: `{simplex, identity}` × `{cosine, jsd}` × `{MDS, UMAP}` = 8 combinaciones manejables.

2. **Definir el alcance de "sustitución":** ¿Basta con demostrar que el espacio agrupa acordes similares (validación intrínseca con métricas de embedding), o se requiere un experimento de sustitución explícito (k-NN + evaluación musical)?

3. **Priorizar la escritura del Marco Teórico** para que el Cap. 3 no cargue con definiciones previas.

4. **Decidir sobre el capítulo de resultados:** ¿Qué figuras y tablas mínimas constituyen evidencia suficiente? Propuesta mínima:
   - Tabla comparativa de métricas (stress, trustworthiness, continuity) para las combinaciones seleccionadas
   - 2-3 embeddings MDS y UMAP con coloración por cardinalidad/rugosidad
   - Diagrama de Shepard para la mejor configuración
   - Curva de disonancia de referencia (díadas) como validación del modelo

5. **Establecer fecha límite para la conversión a LaTeX** — mientras el contenido esté en Markdown, no hay tesis.

---

## 7. INVENTARIO DE ARCHIVOS RELEVANTES

### Documentos de contenido (para la tesis)

| Archivo | Contenido | Prioridad |
|---------|-----------|-----------|
| `metodologia_temporal/ESTRUCTURA_MATEMATICA_DETALLADA.md` | Borrador del Cap. 3 (~39K tokens) | **Máxima** |
| `docs/Tesis MsC/.../00HipotesisPlanteamiento.tex` | Cap. 1 en LaTeX | Alta |
| `docs/Tesis MsC/.../00Objetivos.tex` | Objetivos en LaTeX | Alta |
| `metodologia_temporal/PROPUESTA_ESTRUCTURA_FINAL.md` | Decisión de estructura (Opción C+) | Referencia |
| `metodologia_temporal/modelo_matematico_sustitucion_armonica.md` | Diseño de algoritmo de sustitución | Media (líneas futuras) |
| `metodologia_temporal/substitution_metrics.md` | Métricas de sustitución: notación y código | Media |
| `metodologia_temporal/respuestas_notebooklm.json` | Respuestas verificadas de NotebookLM | Referencia para redacción |
| `RESPUESTAS_RECOPILADAS.md` | 85 Q&A para Cap. 3 (~43 pendientes de consulta) | Alta — cerrar preguntas abiertas |
| `docs/Tesis MsC/.../02Seccion02.tex` | Cap. 2 Marco Teórico previo (~86KB) | Alta — diagnosticar estado |
| `docs/Tesis MsC/.../03Seccion03.tex` | Cap. 3 versión anterior (~73KB) | Media — posible base para LaTeX |
| `docs/Tesis MsC/.../04Seccion04.tex` | Cap. 4 Resultados parcial (~26KB) | Media — verificar utilidad |
| `docs/Tesis MsC/.../metodologia.tex` | Metodología borrador reciente (~42KB, ene 2026) | Alta — versión LaTeX más reciente |
| `docs/Tesis MsC/.../Tesis MSc UNAL.bib` | Bibliografía compilada (~111KB) | Alta — ya existe |

### Documentos de proceso (NO van en la tesis)

| Archivo | Rol |
|---------|-----|
| `.agent/workflows/redactor-critico.md` | Workflow de escritura asistida |
| `.agent/workflows/handoff-redactor.md` | Protocolo de transferencia entre agentes |
| `metodologia_temporal/preguntas_CONTEXTUALIZADAS.json` | Input del workflow |
| `metodologia_temporal/prepare_batch_queries.py` | Script de preparación de consultas |
| Todos los `*.py` en raíz y `metodologia_temporal/` | Scripts auxiliares |

---

## 8. PLAN DE ACCIÓN SUGERIDO (para discutir con el asesor)

### Semana 1-2: Cerrar Cap. 3
- [ ] Resolver preguntas pendientes del documento (secciones §3.5-§3.9 tienen menos resoluciones)
- [ ] Agregar ejemplos numéricos (running example: tríada C-E-G)
- [ ] Generar figuras clave (curva de disonancia, flujo metodológico)
- [ ] Convertir a LaTeX como texto continuo

### Semana 3: Escribir Cap. 2 (Marco Teórico)
- [ ] Extraer fundamentos del Cap. 3 que deben ir en Cap. 2
- [ ] Redactar secciones de psicoacústica, PC-sets, geometría de acordes, reducción dimensional
- [ ] Asegurar que Cap. 3 no repita lo que ya está en Cap. 2

### Semana 4-5: Experimentos y Resultados
- [ ] Seleccionar con el asesor 3-5 configuraciones experimentales
- [ ] Ejecutar experimentos en el repositorio con semillas fijas
- [ ] Generar figuras y tablas
- [ ] Escribir Cap. 4

### Semana 5-6: Discusión, Conclusiones, Pulido
- [ ] Escribir Cap. 5 (Discusión) conectando con hipótesis
- [ ] Escribir Cap. 6 (Conclusiones y líneas futuras)
- [ ] Resumen/Abstract
- [ ] Revisión de notación, bibliografía, formato
- [ ] Entrega al asesor para revisión

---

## 9. NOTA FINAL (para el agente/asesor que lea este informe)

Este proyecto tiene **sustancia matemática real**: hay definiciones formales, proposiciones, justificaciones con literatura, y un repositorio funcional que implementa el modelo. Lo que falta es principalmente **escritura y organización**, no ideas ni resultados.

La hipótesis central del trabajo es **testeable y falsificable**: si ninguna combinación de (normalización × métrica × reductor) produce embeddings con trustworthiness > 0.8 y stress razonable, la representación Φ ∈ ℝ¹² basada en rugosidad no captura estructura útil para sustitución. Los resultados preliminares (pruebas en la GUI) sugieren que sí la captura, pero falta la documentación formal.

El mayor riesgo es el tiempo. La conversión de ~39K tokens de Markdown con notas del autor a prosa LaTeX publicable es un trabajo que requiere atención humana y no debe subestimarse. Las notas del autor ("Nota de santimath") son valiosas porque revelan exactamente dónde hay dudas, dónde falta rigor, y dónde hay ideas sin desarrollar — pero también revelan que hay muchas preguntas abiertas que deben cerrarse o declararse explícitamente fuera de alcance.

**La tesis es viable y tiene valor científico.** Necesita foco, decisiones de alcance firmes, y disciplina de escritura.

---

*Documento generado el 18/02/2026 como herramienta de planificación para la escritura de la tesis. No es un documento público.*
