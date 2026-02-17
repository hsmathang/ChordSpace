# Cómo consultar NotebookLM vía MCP — Guía para agentes LLM

> **Propósito:** Este documento explica a cualquier LLM conectado al IDE cómo consultar NotebookLM mediante el servidor MCP, siguiendo el protocolo de asesoría bibliográfica del proyecto ChordSpace.

---

## 1. Qué es el MCP de NotebookLM

El servidor MCP `notebooklm` expone herramientas para interactuar con NotebookLM **sin navegador** (API directa). Las herramientas principales son:

| Herramienta | Función |
|-------------|---------|
| `mcp_notebooklm_notebook_list` | Listar todos los notebooks del usuario |
| `mcp_notebooklm_notebook_query` | Enviar una consulta a un notebook específico |
| `mcp_notebooklm_notebook_get` | Obtener detalles y fuentes de un notebook |
| `mcp_notebooklm_source_describe` | Obtener resumen de una fuente individual |

**No se necesita**: navegador, autenticación manual, scraping. Todo es API pura.

---

## 2. Notebooks del proyecto ChordSpace (FIJOS)

Estos son los **5 notebooks de la tesis** que deben consultarse. Usa estos IDs exactos:

| Clave | Título | ID | Fuentes |
|-------|--------|----|---------|
| `psicoacustica` | PDFs_Tesis_Psicoacustica | `27d02df9-0405-4ae0-b1d5-58675f73cc49` | 39 |
| `armonia` | PDFs_Tesis_Armonia | `8dedc0d4-9af1-482a-b779-e9733609414a` | 70 |
| `math` | PDFs_Tesis_Math_viz | `3f51c34d-3ad5-4fc2-b2d0-2e0f48a144b8` | 42 |
| `computacion` | PDFs_Tesis_Computacion_ML | `14fa63f0-279b-4348-b50e-5d350542b25b` | 52 |
| `dimred` | Reducción de Dimensionalidad (Metodología) | `43913228-e430-45cb-9489-c3b27904f02c` | 40 |

### ⚠️ PASO 0 OBLIGATORIO: Verificar IDs antes de consultar

**Los IDs de notebooks pueden cambiar sin previo aviso.** Antes de cualquier consulta:

1. Ejecutar `mcp_notebooklm_notebook_list(max_results=10)`.
2. Para cada notebook de la tabla, buscar por **título** (no por ID).
3. Si algún ID cambió, usar el nuevo. Si un título no aparece, detenerse y avisar al usuario.
4. **NUNCA** usar un ID cacheado sin verificar primero con `notebook_list`.

**Diagnóstico rápido:** Si una llamada a `notebook_query` devuelve `"answer": ""` (vacío), lo más probable es que el ID sea obsoleto. Verificar con `notebook_list`.

### Política de consulta

- **Mínimo 4 notebooks** por pregunta tipo A.
- El campo `tema_notebook_sugerido` en `preguntas_CONTEXTUALIZADAS.json` indica el **notebook primario** (recibe el prompt completo).
- Los demás notebooks reciben un **prompt adaptado** pidiendo evidencia complementaria desde su perspectiva específica.

### ⚠️ Regla de formato: Markdown ≠ LaTeX

- En archivos `.md`, **NUNCA** usar comillas LaTeX (` `` ` ... `''`). Los backticks abren bloques de código en Markdown y rompen el renderizado.
- Usar comillas normales `"texto"` en Markdown. La conversión a LaTeX (`\enquote{}` o ` `` '' `) se hará al compilar.

---

## 3. Cómo hacer una consulta

### 3.1. Leer la pregunta

Archivo de entrada: `metodologia_temporal/preguntas_CONTEXTUALIZADAS.json`

Solo procesar preguntas con `"tipo": "A"`. Usar el campo `pregunta_reformulada_para_notebooklm` como base del prompt.

### 3.2. Prompt para el notebook PRIMARIO

```
ROL:
Eres mi asesor de tesis (maestría en matemáticas aplicadas) para el proyecto ChordSpace.

REGLAS:
1) Usa SOLO los documentos de ESTE notebook. Prohibido conocimiento externo.
2) No inventes: si no hay evidencia suficiente, dilo y pide el PDF/fuente faltante.
3) Cita fuentes originales (papers/libros) cuando existan en el notebook.
4) Incluye páginas o sección del documento para cada afirmación importante.
5) Entrega BibTeX válido por fuente.
6) Escribe en español académico (breve, denso, orientado a metodología).

CONTEXTO GLOBAL (ChordSpace / Metodología):
- Objetivo: espacio de representación de acordes para explorar/sugerir sustituciones por similitud sonora.
- Dominio: MIDI n∈{0..127}, 12‑TET (A4=440 Hz), f(n)=440·2^((n-69)/12).
- Acorde: tupla estrictamente creciente (sin unísonos MIDI); identidad sensible a registro/voicing (no PC-sets).
- Feature: rugosidad/disonancia sensorial (Plomp–Levelt + Sethares), tonos complejos con parciales armónicos (H=6, δ=0.88).
- Representación: Φ_raw∈R_{≥0}^{12} por clase de intervalo, sin colapsar complementarios (intervalo 0→índice 11).
- Pipeline: población→Φ_raw→normalización→distancia ρ→matriz D→embedding 2D (MDS/UMAP/…)→evaluación y supuestos.

CONTEXTO/PREGUNTA:
<<PEGA AQUÍ `pregunta_reformulada_para_notebooklm`>>

FORMATO DE SALIDA (OBLIGATORIO):
1) RESPUESTA PARA INSERTAR (máx. 180–250 palabras)
   - Lista para pegar en tesis (metodología). Usa citas tipo \cite{claveBib}.

2) AFIRMACIONES + EVIDENCIA (5–10 ítems)
   - Cada ítem: afirmación breve + (Fuente, año, páginas/ubicación).

3) BIBTEX (mínimo 3 fuentes, máximo 7)

4) NOTAS DEL ASESOR (3–6 bullets)
   - Objeciones típicas de jurado + cómo mitigarlas.

5) PREGUNTAS DE SEGUIMIENTO (0–3)
   - Solo si falta info crítica en los documentos.
```

### 3.3. Prompt para notebooks SECUNDARIOS

Adaptar el prompt pidiendo evidencia **complementaria** desde la perspectiva del notebook:

```
ROL:
Eres mi asesor de tesis (maestría en matemáticas aplicadas) para el proyecto ChordSpace.

REGLAS:
1) Usa SOLO los documentos de ESTE notebook. Prohibido conocimiento externo.
2) No inventes: si no hay evidencia suficiente, dilo.
3) Cita fuentes originales con páginas.
4) Entrega BibTeX válido por fuente.
5) Escribe en español académico.

CONTEXTO GLOBAL (ChordSpace):
- Objetivo: espacio de representación de acordes para explorar sustituciones por similitud sonora.
- Dominio: MIDI n∈{0..127}, 12‑TET, f(n)=440·2^((n-69)/12).
<<Agregar 1-2 líneas adicionales de contexto según la perspectiva del notebook>>

PREGUNTA (busco evidencia COMPLEMENTARIA desde perspectiva <<PERSPECTIVA>>):
<<Reformular la pregunta enfocándola al área del notebook>>

QUÉ BUSCAMOS:
<<Especificar qué tipo de evidencia se espera de este notebook>>

FORMATO DE SALIDA (OBLIGATORIO):
1) AFIRMACIONES + EVIDENCIA (5–10 ítems)
2) BIBTEX (mínimo 3, máximo 7)
3) NOTAS DEL ASESOR (2–4 bullets)
```

### 3.4. Llamada MCP

```python
# Ejemplo de llamada (pseudocódigo para el agente)
mcp_notebooklm_notebook_query(
    notebook_id="3f51c34d-3ad5-4fc2-b2d0-2e0f48a144b8",  # ID del notebook
    query="<prompt completo aquí>",
    timeout=120
)
```

**Tip**: Las 5 consultas pueden ejecutarse **en paralelo** para mayor velocidad.

---

## 4. Formato de respuesta (almacenamiento)

Las respuestas se guardan en: `metodologia_temporal/respuestas_notebooklm.json`

### Schema por respuesta

```json
{
  "id": "Q-003",
  "tipo": "A",
  "pregunta_original": "texto...",
  "tema_primario": "math",
  "notebooks_consultados": ["math", "psicoacustica", "armonia", "computacion", "dimred"],
  "quality_gates_passed": true,
  "fecha_consulta": "2026-02-16T14:37:00-05:00",
  
  "respuesta_para_insertar": "Párrafo con \\cite{...} listo para tesis...",
  
  "evidencia_por_notebook": {
    "math": {
      "notebook_title": "PDFs_Tesis_Math_viz",
      "afirmaciones": [
        {
          "afirmacion": "Texto de la afirmación...",
          "fuente": "Autor, año",
          "ubicacion": "pp. X-Y"
        }
      ]
    }
  },
  
  "bibtex": ["@article{...}", "..."],
  
  "notas_asesor": ["Nota 1...", "..."],
  
  "preguntas_seguimiento": ["Pregunta 1...", "..."],
  
  "para_redactor": {
    "donde_insertar": "§3.1.1, después de Definición 3.1",
    "tipo_contenido": "justificacion",
    "palabras_clave": ["MIDI", "dominio", "..."]
  }
}
```

### Campos clave para el agente redactor

| Campo | Para qué sirve |
|-------|----------------|
| `respuesta_para_insertar` | Párrafo listo para copiar a la tesis |
| `evidencia_por_notebook` | Tabla de afirmaciones con fuentes por perspectiva |
| `bibtex` | Entradas BibTeX para agregar a la bibliografía |
| `notas_asesor` | Anticipar objeciones del jurado |
| `para_redactor.donde_insertar` | Ubicación exacta en el capítulo |
| `para_redactor.tipo_contenido` | Tipo de contenido (justificación, definición, comparación) |

---

## 5. Quality gates

Marcar `quality_gates_passed: true` **SOLO** si:

| Criterio | Mínimo |
|----------|--------|
| Fuentes totales (todos los notebooks) | ≥ 8 |
| BibTeX por fuente | 1 por fuente citada |
| Páginas/ubicación | Presente en mayoría de afirmaciones |
| Notebooks consultados | ≥ 4 |
| Respuesta para insertar | 180–250 palabras con `\cite{}` |

### Si falla un quality gate

1. Reenviar la consulta al notebook con un follow-up pidiendo páginas o BibTeX faltantes.
2. Máximo **2 follow-ups** por notebook.
3. Si después de 2 follow-ups sigue fallando, registrar `quality_gates_passed: false` con nota explicativa.

---

## 6. Perspectivas por notebook

Al adaptar el prompt para cada notebook secundario, usar estas perspectivas:

| Notebook | Perspectiva | Qué buscar |
|----------|-------------|------------|
| `psicoacustica` | Percepción auditiva | Rugosidad, ancho de banda crítico, dependencia del registro, invarianza tímbrica |
| `armonia` | Representación de acordes | Voicings, inversiones, espacios geométricos, dominio fundamental, PC-sets vs voicing concreto |
| `math` | Formalización matemática | Espacios métricos, topología, combinatoria, definiciones de dominio |
| `computacion` | ML / Computacional | Tractabilidad, tokenización, embeddings, muestreo, explosión de estados |
| `dimred` | Reducción dimensional | MDS, UMAP, t-SNE, escalabilidad, crowding problem, calidad de embedding |

---

## 7. Archivos de referencia

| Archivo | Función |
|---------|---------|
| `preguntas_CONTEXTUALIZADAS.json` | **INPUT**: preguntas tipo A con contexto y reformulación |
| `respuestas_notebooklm.json` | **OUTPUT**: respuestas consolidadas para el redactor |
| `COMO_CONSULTAR_NOTEBOOKLM_MCP.md` | **ESTE ARCHIVO**: instrucciones para el agente |
| `PROTOCOLO_CONSULTA_NOTEBOOKLM_MCP.md` | Protocolo original (referencia histórica) |
| `PROMPT_NOTEBOOKLM_ASESOR_MCP.md` | Prompt original (referencia histórica) |
