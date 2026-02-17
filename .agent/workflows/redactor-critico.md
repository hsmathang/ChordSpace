---
description: Flujo de redacción crítica para insertar respuestas de NotebookLM en el capítulo de metodología. Usa 4 skills para filtrar, formalizar, sintetizar y redactar.
---

# Redactor Crítico — Flujo de Trabajo

> **Objetivo:** Tomar las respuestas almacenadas en `respuestas_notebooklm.json`, evaluar su relevancia, y producir texto insertable en `ESTRUCTURA_MATEMATICA_DETALLADA.md` (capítulo de metodología de la tesis).
>
> **Principio rector:** Resolver cada pregunta de la manera más sencilla posible sin ahondar en detalles innecesarios que desvíen el foco metodológico y los objetivos comunicativos del capítulo y de la sección en cuestión.

---

## Archivos de entrada/salida

| Archivo | Rol |
|---------|-----|
| `metodologia_temporal/preguntas_CONTEXTUALIZADAS.json` | INPUT: preguntas tipo A con contexto, ubicación y reformulación |
| `metodologia_temporal/respuestas_notebooklm.json` | INPUT: respuestas consolidadas de 5 notebooks con evidencia, BibTeX, notas asesor |
| `metodologia_temporal/ESTRUCTURA_MATEMATICA_DETALLADA.md` | TARGET: capítulo donde se inserta el texto redactado |
| `metodologia_temporal/COMO_CONSULTAR_NOTEBOOKLM_MCP.md` | REF: instrucciones para consultas adicionales si se necesita más info |

---

## Fase -1: Contexto Dinámico Acumulativo (NUEVO — v2)

> **Problema detectado (2026-02-17):** En lotes grandes de queries (10+ preguntas × 5 notebooks), se observó degradación progresiva de los prompts: las últimas queries perdían contexto global, perspectiva por notebook, y riqueza de reformulación. Esto produce respuestas más genéricas y menos útiles.
>
> **Solución:** Contexto que crece con cada query exitosa.

### -1.1 Inicialización del Context Buffer

Antes de iniciar un lote de queries, crear un **Context Buffer** (mental o explícito) con:

```
CONTEXT_BUFFER = {
  "contexto_base": <contenido de COMO_CONSULTAR_NOTEBOOKLM_MCP.md §3.2, líneas 80-86>,
  "hallazgos_previos": [],
  "decisiones_confirmadas": [],
  "terminologia_establecida": []
}
```

El `contexto_base` es el bloque fijo que **SIEMPRE** debe incluirse en cada prompt (nunca omitirlo, nunca abreviarlo):

```
CONTEXTO GLOBAL (ChordSpace / Metodología):
- Objetivo: espacio de representación de acordes para explorar/sugerir sustituciones por similitud sonora.
- Dominio: MIDI n∈{0..127}, 12‑TET (A4=440 Hz), f(n)=440·2^((n-69)/12).
- Acorde: tupla estrictamente creciente (sin unísonos MIDI); identidad sensible a registro/voicing.
- Feature: rugosidad/disonancia sensorial (Plomp–Levelt + Sethares), tonos complejos con parciales armónicos (H=6, δ=0.88).
- Representación: Φ_raw∈R_{≥0}^{12} por clase de intervalo.
- Pipeline: población→Φ_raw→normalización→distancia ρ→matriz D→embedding 2D (MDS/UMAP)→evaluación.
```

### -1.2 Enriquecimiento tras cada query

Después de procesar Q-N (sintetizar sus 5 respuestas), actualizar el Context Buffer:

```python
# Pseudocódigo — el agente hace esto mentalmente o en un buffer
hallazgo = extraer_hallazgo_clave(Q_N)  # 1-2 oraciones
CONTEXT_BUFFER["hallazgos_previos"].append(f"Q-{N}: {hallazgo}")

# Si la respuesta confirmó una decisión metodológica:
if decision_confirmada:
    CONTEXT_BUFFER["decisiones_confirmadas"].append(decision)

# Si se estableció terminología nueva:
if nuevo_termino:
    CONTEXT_BUFFER["terminologia_establecida"].append(termino)
```

**Ejemplo de evolución del buffer después de 3 queries:**

```
hallazgos_previos:
  - "Q-019: La percepción de acordes ≥3 notas no es suma lineal de intervalos; Φ_raw pierde configuración interna (limitación documentada)."
  - "Q-020: Ceros en Φ_raw se manejan con suavizado gaussiano previo + ε-padding para JSD."
  - "Q-021: Ground truth para tríadas: Bowling2018 (66 tríadas cromáticas). Unidad formal: asper."

decisiones_confirmadas:
  - "Φ_raw es simplificación deliberada que pierde configuración interna del acorde."
  - "El suavizado gaussiano tiene justificación perceptual (no heurística)."

terminologia_establecida:
  - "asper (unidad psicofísica de rugosidad)"
  - "trustworthiness, continuity (métricas de evaluación de embeddings)"
```

### -1.3 Inyección en el prompt de Q-N+1

Cada prompt nuevo debe incluir, **después del contexto base y antes de la pregunta**, un bloque:

```
CONTEXTO ACUMULADO (hallazgos previos de esta sesión):
- Q-019: [hallazgo resumido en 1 línea]
- Q-020: [hallazgo resumido en 1 línea]
- ...
(Esto te ayuda a contextualizar la respuesta y evitar repeticiones.)
```

**Límite:** Máximo 5-8 hallazgos en el bloque (los más recientes y relevantes). Si hay más de 8, rotar y quedarse con los más conectados a la pregunta actual.

### -1.4 Regla de estabilidad

> **REGLA INVIOLABLE:** El `contexto_base` (las 6 líneas del bloque CONTEXTO GLOBAL) no puede abreviarse, omitirse ni parafrasearse NUNCA, independientemente del número de queries procesadas. Si la ventana de contexto se satura, reducir los `hallazgos_previos`, no el contexto base.

---

## Fase -0.5: Checklist de Calidad del Prompt (NUEVO — v2)

> **Propósito:** Evitar la degradación observada. Verificar ANTES de enviar cada prompt.

### Checklist obligatoria por prompt (5 notebooks × N preguntas)

| # | Criterio | Falla → |
|---|----------|--------|
| 1 | ¿Incluye el bloque CONTEXTO GLOBAL completo (6 líneas)? | DETENER. Agregar antes de enviar. |
| 2 | ¿El notebook primario recibe el formato completo (ROL + REGLAS + CONTEXTO + FORMATO 4 secciones)? | DETENER. Usar template §3.2. |
| 3 | ¿Los notebooks secundarios tienen perspectiva adaptada (no copia del primario)? | Reformular según §6 Perspectivas. |
| 4 | ¿Se incluyó contexto acumulado de hallazgos previos (si hay)? | Agregar bloque de hallazgos. |
| 5 | ¿La pregunta fue reformulada para la perspectiva del notebook, no copiada textual? | Reformular. |
| 6 | ¿El prompt tiene longitud similar (±20%) al del primer prompt del lote? | Si es significativamente más corto, está degradado. Revisar. |

### Indicador de degradación

Si al llegar al prompt N la longitud promedio es <70% de la longitud del prompt 1, hay degradación activa. Acción correctiva:
1. Pausar el lote
2. Regenerar los prompts restantes usando el template completo + Context Buffer
3. Continuar

---

## Fase 0: Carga y Orientación

1. Leer `respuestas_notebooklm.json` y ubicar la respuesta objetivo por su `id` (e.g., `Q-003`).
2. Leer `preguntas_CONTEXTUALIZADAS.json` para obtener:
   - `ubicacion_original` (sección y línea exacta)
   - `contexto_extraido_del_capitulo` (modelo general, decisiones previas, parámetros)
   - `para_redactor.donde_insertar` (ubicación precisa en el capítulo)
   - `para_redactor.tipo_contenido` (justificación / definición / comparación / decisión)
3. Leer la sección relevante de `ESTRUCTURA_MATEMATICA_DETALLADA.md`:
   - El texto circundante (2-3 párrafos antes y después del punto de inserción)
   - Las "Notas de santimath" en la zona (contienen intención del autor)
   - Las "Preguntas clave" de la sección (dan contexto de lo que se busca resolver)
4. Identificar los objetivos comunicativos de la sección:
   - ¿Qué debe entender el lector al terminar esta sección?
   - ¿Qué definiciones previas ya tiene disponibles?
   - ¿Qué viene después y cómo conecta?

---

## Fase 1: Gate de Relevancia (scientific-critical-thinking)

> **Skill:** `scientific-critical-thinking` → Capability 7: Claim Evaluation + Capability 4: Evidence Quality Assessment
>
> **Propósito:** Decidir QUÉ de la respuesta vale la pena incluir y QUÉ se descarta. Evitar inflar el capítulo con detalles tangenciales.

### 1.1 Clasificar cada afirmación

Para cada afirmación en `evidencia_por_notebook`, asignar un nivel:

| Nivel | Criterio | Acción |
|-------|----------|--------|
| **A — Esencial** | Responde directamente la pregunta. Sin esta info, la justificación queda incompleta. La fuente es de alta calidad (revista indexada, libro de referencia). | **Incluir** en el texto principal |
| **B — Refuerza** | Apoya la respuesta pero no es indispensable. Añade perspectiva útil sin ser la evidencia central. | **Incluir si cabe** sin alargar más de 1 oración; de lo contrario, nota al pie o cita parentética |
| **C — Tangencial** | Interesante pero se desvía del foco metodológico. Podría ir en otro capítulo o en discusión. | **Omitir** del texto. Puede guardarse como nota interna para cap. 4 o 5 |

### 1.2 Evaluar proporcionalidad afirmación–evidencia

Aplicar el principio de proporcionalidad de `scientific-critical-thinking`:
- ¿La fuerza de la afirmación es proporcional a la evidencia disponible?
- ¿Se está haciendo una afirmación causal donde solo hay correlación?
- ¿Se necesita hedging language? ("se ha propuesto que...", "la evidencia sugiere...", "Sethares (1993) argumenta que...")

### 1.3 Identificar información faltante

Si después de clasificar, hay un gap lógico (la justificación no cierra sin un dato), marcar para:
- Follow-up a NotebookLM (usar `COMO_CONSULTAR_NOTEBOOKLM_MCP.md`)
- O aceptar como limitación y declarar hedge

### Formato de salida de Fase 1

```markdown
## Evaluación de Relevancia — Q-XXX

### Afirmaciones Nivel A (incluir obligatoriamente)
- [notebook: math] Tymoczko 2011: espacio lineal ≠ circular → NECESARIO para justificar registro
- [notebook: psicoacustica] Eerola 2022: rugosidad satura bajo C3 → CLAVE para rango

### Afirmaciones Nivel B (incluir si cabe)
- [notebook: armonia] Callender 2008: regiones fundamentales → apoya pero no es central aquí

### Afirmaciones Nivel C (omitir)
- [notebook: computacion] Dalmazzo 2023: tokenización → interesante pero irrelevante para §3.1.1

### Gaps detectados
- Ninguno / [descripción del gap y acción]
```

---

## Fase 2: Formalización de Claims (hypothesis-generation)

> **Skill:** `hypothesis-generation` → Steps 3-5: Sintetizar evidencia, generar hipótesis, evaluar calidad
>
> **Propósito:** Transformar las afirmaciones seleccionadas en claims formalizados con estructura lógica clara, no en prosa suelta.

### 2.1 Formular el claim central

A partir de las afirmaciones Nivel A, formular **un solo claim** que la sección debe comunicar:

```
CLAIM: [Afirmación principal que resuelve la pregunta, en una oración]
PORQUE: [Razón 1 — fuente] + [Razón 2 — fuente] + ...
IMPLICACIÓN PARA EL MODELO: [Qué cambia/se justifica en el modelo ChordSpace]
```

Ejemplo para Q-003:
```
CLAIM: El dominio teórico se define sobre N={0,...,127} pero la experimentación 
       se restringe al subrango MIDI 48–84 (C3–C6).
PORQUE: (1) La rugosidad depende del ancho de banda crítico, que varía con registro 
        (Cubarsí 2019, Sethares 1993); (2) La consonancia perceptual tiene curva U 
        invertida con óptimo en C4–C5 (Eerola 2022); (3) La explosión combinatoria 
        hace inviable el rango completo (Buongiorno Nardelli 2020).
IMPLICACIÓN: Se declara como supuesto metodológico, no como limitación.
```

### 2.2 Preguntas de control

Hacerse las preguntas del framework de hypothesis-generation:

1. **¿El claim es falsificable?** ¿Qué evidencia lo refutaría?
2. **¿Es el claim más parsimonioso posible?** ¿Se puede simplificar sin perder rigor?
3. **¿Hay claims alternativos?** Si otro enfoque es razonable, ¿se reconoce?

Si la respuesta a (2) es "sí, se puede simplificar", **simplificar**. El principio rector es brevedad.

---

## Fase 3: Síntesis Temática (literature-review)

> **Skill:** `literature-review` → Phase 5: Synthesis and Analysis
>
> **Propósito:** Sintetizar las fuentes de múltiples notebooks en una narrativa coherente, NO como lista de citas.

### 3.1 Organizar por argumento, no por notebook

Las afirmaciones ya clasificadas (Fase 1) y formalizadas (Fase 2) deben organizarse temáticamente:

```
ARGUMENTO 1: La restricción de rango tiene justificación perceptual
  → Eerola 2022 (psicoacustica) + Cubarsí 2019 (math) + Sethares 1993

ARGUMENTO 2: La restricción tiene justificación computacional  
  → Buongiorno Nardelli 2020 (armonia) + Yang 2006 (dimred) + Quick & Hudak 2011 (math)

ARGUMENTO 3: Es práctica estándar en la literatura
  → Chew 2014 (math) + Callender 2008 (armonia)
```

### 3.2 Regla del triángulo de evidencia

Para que un argumento se incluya en el texto, debe tener al menos **2 fuentes independientes** (preferiblemente de notebooks distintos). Si solo tiene 1 fuente, se incluye como referencia pero no como argumento central.

### 3.3 Seleccionar BibTeX

De la lista en `respuestas_notebooklm.json[bibtex]`, incluir SOLO las entradas que aparecen en las afirmaciones Nivel A y B seleccionadas. No incluir BibTeX de afirmaciones Nivel C.

---

## Fase 4: Redacción Estructurada (scientific-writing)

> **Skill:** `scientific-writing` → Capabilities 7 (Writing Process) + 2 (Section-Specific Writing) + 6 (Writing Principles)
>
> **Propósito:** Producir el texto final insertable.

### 4.1 Seguir el proceso de dos etapas

**Etapa 1 — Outline con key points** (ya hecho en Fases 1-3):
- Claims formalizados
- Argumentos con fuentes
- Orden lógico

**Etapa 2 — Convertir a prosa fluida:**

Reglas de redacción para este capítulo específico:

1. **Idioma:** Español académico (la tesis está en español)
2. **Prosa, nunca bullets** en el texto final: Todo en párrafos con transiciones
3. **Citas integradas**: `\cite{claveBib}` dentro de las oraciones, no al final como lista
4. **Longitud objetivo por respuesta:** 150–250 palabras (máx. 300 para preguntas complejas)
5. **Registro:** Formal pero no pomposo. El lector tiene doctorado en matemáticas
6. **Notación consistente:** Usar la misma notación del capítulo ($\mathcal{N}$, $\Phi_{\text{raw}}$, etc.)
7. **Conectores con el texto circundante:** El párrafo debe fluir naturalmente desde lo que viene antes y hacia lo que viene después. No empezar con "En respuesta a..." ni "Con respecto a..."
8. **Priorizar:** Definiciones > Justificaciones > Comparaciones > Discusión abierta
9. **Comillas:** Usar `"texto"` estándar. **NUNCA** usar comillas LaTeX (` `` ` ... `''`) en archivos `.md` — los backticks rompen el renderizado Markdown. La conversión a LaTeX se hará al compilar.

### 4.2 Templates según tipo de contenido

**Si `para_redactor.tipo_contenido` = "justificacion":**
```
[Afirmación directa de la decisión tomada]. Esta elección se sustenta en [N] 
consideraciones. En primer lugar, [argumento perceptual/teórico] \cite{fuente1, fuente2}. 
[Desarrollo breve, 1-2 oraciones]. Además, [argumento computacional/práctico] 
\cite{fuente3}. [Si aplica: conexión con el modelo ChordSpace]. Por consiguiente, 
[restatement de la decisión como supuesto formal del modelo].
```

**Si `para_redactor.tipo_contenido` = "definicion":**
```
[Contexto que motiva la definición, 1 oración]. [Definición formal en LaTeX]. 
[Interpretación en lenguaje natural, 1-2 oraciones]. [Conexión con la decisión 
del modelo, si aplica] \cite{fuente}.
```

**Si `para_redactor.tipo_contenido` = "comparacion":**
```
Existen [N] enfoques principales para [tema]: [lista breve]. [Enfoque elegido] 
se selecciona porque [razón 1] \cite{fuente1} y [razón 2] \cite{fuente2}. 
[Los enfoques alternativos] se descartan dado que [razón breve]. [Nota sobre 
extensibilidad si aplica].
```

### 4.3 Checklist de calidad pre-entrega

Antes de considerar terminado el texto:

- [ ] ¿El texto responde la pregunta original sin rodeos?
- [ ] ¿Se puede entender sin leer la respuesta completa de NotebookLM?
- [ ] ¿La notación es consistente con el resto del capítulo?
- [ ] ¿Las citas son integrables al `.bib` existente?
- [ ] ¿El texto NO introduce conceptos que se definen más adelante?
- [ ] ¿Se mantiene dentro de 150–250 palabras?
- [ ] ¿Un lector con doctorado en matemáticas lo leería sin fricción?
- [ ] ¿NO hay bullets, listas numeradas, ni headers internos?

---

## Fase 5: Revisión Crítica Final (scientific-critical-thinking)

> **Skill:** `scientific-critical-thinking` → Capability 1: Methodology Critique
>
> **Propósito:** Última revisión antes de insertar. Detectar si se introdujeron problemas.

### 5.1 Checklist de rigor

| Pregunta | Si falla → |
|----------|-----------|
| ¿Alguna afirmación carece de cita? | Añadir cita o eliminar afirmación |
| ¿Se hace un claim más fuerte que la evidencia? | Añadir hedging |
| ¿Se usa "demuestra" o "prueba" cuando debería ser "sugiere" o "indica"? | Corregir verbo |
| ¿El texto introduce una tangente no resuelta? | Eliminar o mover a §3.8 (Supuestos) |
| ¿Se promete algo que el capítulo no entrega? | Eliminar promesa |
| ¿El párrafo se siente como respuesta a una pregunta o como prosa natural? | Reescribir para que fluya |

### 5.2 Test del lector hostil

Imaginar que un evaluador del jurado lee este párrafo y pregunta:
- "¿Y la fuente de esto?" → Cada claim debe tener `\cite{}`
- "¿Por qué es relevante aquí?" → Debe conectar con el objetivo de la sección
- "¿No están sobrecomplicando esto?" → Si sí, simplificar

---

## Fase 6: Inserción y Metadata

### 6.1 Insertar en el capítulo

Usar la información de `para_redactor.donde_insertar` para ubicar exactamente dónde va el texto en `ESTRUCTURA_MATEMATICA_DETALLADA.md`.

El texto se inserta como contenido nuevo:
- Marcado con `<!-- REDACTADO: Q-XXX -->` al inicio
- Marcado con `<!-- FIN REDACTADO: Q-XXX -->` al final
- Esto permite trazabilidad y revisión posterior

### 6.2 Actualizar respuestas_notebooklm.json

Después de redactar, actualizar el objeto de la respuesta en el JSON:

```json
{
  "redaccion": {
    "estado": "redactado",
    "fecha_redaccion": "2026-02-16T15:XX:00-05:00",
    "palabras": 215,
    "afirmaciones_usadas": ["Tymoczko2011", "Eerola2022", "Cubarsi2019", "BuongiornoNardelli2020"],
    "afirmaciones_descartadas": ["Dalmazzo2023", "Lazzari2023"],
    "razon_descartes": "tokenización y embeddings no relevantes para §3.1.1 (dominio de notas)"
  }
}
```

---

## Resumen del flujo

```
┌─────────────────────────────────────────────────────────────────┐
│  FASE 0: CARGA                                                  │
│  Leer: pregunta + respuesta + sección del capítulo              │
├─────────────────────────────────────────────────────────────────┤
│  FASE 1: GATE DE RELEVANCIA (scientific-critical-thinking)      │
│  → Clasificar afirmaciones en A (incluir) / B (si cabe) / C    │
│  → Evaluar proporcionalidad claim–evidencia                     │
│  → Detectar gaps                                                │
├─────────────────────────────────────────────────────────────────┤
│  FASE 2: FORMALIZACIÓN (hypothesis-generation)                  │
│  → Formular claim central en 1 oración                          │
│  → Verificar falsificabilidad y parsimonia                      │
│  → Simplificar si es posible                                    │
├─────────────────────────────────────────────────────────────────┤
│  FASE 3: SÍNTESIS (literature-review)                           │
│  → Organizar por argumento, no por notebook                     │
│  → Aplicar regla del triángulo (≥2 fuentes por argumento)       │
│  → Seleccionar BibTeX final                                     │
├─────────────────────────────────────────────────────────────────┤
│  FASE 4: REDACCIÓN (scientific-writing)                         │
│  → Convertir outline a prosa académica en español               │
│  → 150–250 palabras, citas integradas, notación consistente     │
│  → Template según tipo (justificación/definición/comparación)   │
├─────────────────────────────────────────────────────────────────┤
│  FASE 5: REVISIÓN FINAL (scientific-critical-thinking)          │
│  → Checklist de rigor: citas, hedging, tangentes                │
│  → Test del lector hostil                                       │
├─────────────────────────────────────────────────────────────────┤
│  FASE 6: INSERCIÓN + METADATA                                  │
│  → Insertar en ESTRUCTURA_MATEMATICA_DETALLADA.md               │
│  → Actualizar respuestas_notebooklm.json con estado             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Principios a respetar siempre

1. **Brevedad > Exhaustividad.** Si puedes zanjar en 150 palabras, no uses 250.
2. **Metodología > Teoría general.** Solo incluir lo que justifica una decisión del modelo ChordSpace.
3. **Evidencia > Opinión.** Cada afirmación con `\cite{}`. Sin cita → no se incluye.
4. **Flujo narrativo > Completitud.** Mejor omitir una referencia que romper el flujo del párrafo.
5. **El lector es matemático.** No explicar lo que un doctor en matemáticas ya sabe. Sí explicar decisiones musicológicas/psicoacústicas que un matemático NO sabría.
6. **No prometer lo que el capítulo no entrega.** Si se menciona un concepto, debe estar definido previamente o en esta misma sección.
7. **Contexto nunca decrece.** El contexto proporcionado a NotebookLM en cada query debe ser al menos tan rico como el de la query anterior. Si hay menos contexto, hay un error de proceso. (Principio añadido v2, 2026-02-17).
8. **Prompt Quality = Response Quality.** La calidad del prompt determina la calidad de la respuesta. Un prompt degradado produce respuestas genéricas. Verificar con Checklist de Calidad del Prompt (Fase -0.5). (Principio añadido v2, 2026-02-17).
