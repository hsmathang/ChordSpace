# Protocolo de consulta a NotebookLM (vía MCP) — “Asesor de tesis” para ChordSpace

Este protocolo define **cómo consultar NotebookLM** (con notebooks tipo `PDF_thesis_*` / `PDFs_Tesis_*`) para obtener respuestas **útiles para la tesis**: rigurosas, accionables, **con citas rastreables** y **BibTeX** listo para insertar.

**Entrada principal:** `metodologia_temporal/preguntas_CONTEXTUALIZADAS.json`  
**Salida recomendada:** `metodologia_temporal/respuestas_notebooklm_contextualizadas.json`

---

## 0) Qué problema resuelve este protocolo

Evita respuestas genéricas del tipo “definición de manual”, forzando:

1) **Contexto completo del modelo** (tus decisiones/variables reales: MIDI, 12‑TET, Φ_raw 12 bins, Sethares, MDS/UMAP, etc.).  
2) **Respuesta estructurada** orientada a escritura académica.  
3) **Citación fuerte**: fuentes del notebook + páginas/ubicación + **BibTeX**.

---

## 0.1) Contexto general (fijo) — propuesta, capítulo y repo

Este bloque es el “marco común” para que NotebookLM responda con precisión **en el contexto real de tu tesis**.

**Regla de oro:** en **todas** las consultas tipo A se envían dos capas de contexto:
1) **Contexto global fijo** (este bloque, en versión compacta).
2) **Contexto específico de la pregunta** (`pregunta_reformulada_para_notebooklm` del JSON).

### Propuesta / tesis (qué estás tratando de probar/construir)

- **Problema:** facilitar la exploración de sonoridades y la búsqueda de acordes “sustitutos” más allá de las limitaciones de la teoría armónica tradicional, mediante un modelo computacional interpretable.
- **Pregunta central:** construir un **espacio de representación** donde acordes con sonoridad similar queden cerca (y por tanto se puedan explorar como vecinos/sustitutos).
- **Hipótesis de trabajo:** en un contexto armónico dado existen acordes no explorados que satisfacen criterios de similitud sonora y pueden funcionar como sustitutos.

### Capítulo 3 (metodología) — qué objetos/decisiones ya están fijadas

- **Dominio:** notas como MIDI $n\\in\\{0,\\dots,127\\}$ y afinación 12‑TET con $A4=440$ Hz, usando $f(n)=440\\cdot 2^{(n-69)/12}$.
- **Objeto acorde:** acorde como tupla estrictamente creciente (sin unísonos exactos MIDI); el modelo es **sensible al registro/voicing** (no colapsa por PC‑sets).
- **Feature principal:** rugosidad/disonancia sensorial como proxy (Plomp–Levelt + parametrización de Sethares), extendida a tonos complejos por suma entre parciales; en el repo se usa $H=6$ y $\\delta=0.88$.
- **Vectorización:** $\\Phi_{raw}\\in\\mathbb{R}_{\\ge 0}^{12}$ (bins por clase de intervalo) sin colapsar complementarios; convención de implementación: intervalo mod 12 = 0 (octava) → índice 11.
- **Geometría/embedding:** normalizaciones + métricas → matriz $D$ → reducción a 2D (MDS como principal; UMAP/t‑SNE/ISOMAP como complementarios) + evaluación cuantitativa (stress, vecindarios, Shepard, etc.) + supuestos/amenazas.

### Repo (operacionalización) — qué significa “implementar” en tu trabajo

- El repositorio implementa un pipeline GUI/CLI para: construir poblaciones (DB o combinatorial), filtrar/deduplicar/transponer, ejecutar escenarios (normalización×métrica×reductor×seed) y generar **reportes reproducibles** (artefactos/metadata + HTML).
- Por diseño, el capítulo **no es un manual de software**: la implementación se describe como operacionalización reproducible del modelo (parámetros, supuestos, artefactos).

### Contexto global compacto (pegar en cada consulta)

Pegar este bloque **antes** de `pregunta_reformulada_para_notebooklm`:

```
CONTEXTO GLOBAL (ChordSpace / Metodología):
- Objetivo: espacio de representación de acordes para explorar/sugerir sustituciones por similitud sonora.
- Dominio: MIDI n∈{0..127}, 12‑TET (A4=440 Hz), f(n)=440·2^((n-69)/12).
- Acorde: tupla estrictamente creciente (sin unísonos MIDI); identidad sensible a registro/voicing (no PC-sets).
- Feature: rugosidad/disonancia sensorial (Plomp–Levelt + Sethares), tonos complejos con parciales armónicos (H=6, δ=0.88).
- Representación: Φ_raw∈R_{≥0}^{12} por clase de intervalo, sin colapsar complementarios (intervalo 0→índice 11).
- Pipeline: población→Φ_raw→normalización→distancia ρ→matriz D→embedding 2D (MDS/UMAP/…)→evaluación y supuestos.
```

---

## 1) Precondiciones (antes de correr)

1. MCP conectado y el agente puede **listar notebooks**.
2. Existen notebooks con prefijo (o equivalente):
   - `PDF_thesis_*` (tu convención nueva), o
   - `PDFs_Tesis_*` (convención usada en docs del repo).
3. El agente puede enviar una consulta y recibir texto (con citas) de NotebookLM.

> Nota: si el MCP es community/automation, asume límites de chunking/longitud; este protocolo está diseñado para consultas **medianas** (≈300–900 tokens por prompt) y respuestas **controladas**.

---

## 2) Mapeo tema → notebook (regla simple)

Cada pregunta tipo A en `preguntas_CONTEXTUALIZADAS.json` tiene `tema_notebook_sugerido`:

- `psicoacustica` → notebook cuyo nombre contenga `Psicoacustica` (o `Psychoacoustics`, `Roughness`, `Sethares`, `Plomp`).
- `math` → notebook con `Math`, `Matem`, `Topology`, `Metric`, `ChordSpace`.
- `armonia` → notebook con `Armonia`, `Harmony`, `Chord`, `Voicing`, `Forte`.
- `reduccion_dimensionalidad` → notebook con `Dimensionalidad`, `DimRed`, `UMAP`, `MDS`, `t-SNE`, `Manifold`.

Si hay varios candidatos, elige el que tenga más coincidencias por palabras clave.

---

## 3) Persona (instrucción persistente recomendada)

Si el sistema permite “Goal/Persona” permanente, usar esto:

**PERSONA / GOAL (pegar tal cual):**

> Actúa como mi asesor de tesis de maestría en matemáticas aplicadas y tecnología musical (ChordSpace).  
> Responde **únicamente** con evidencia de los documentos del notebook. **Prohibido** usar conocimiento externo.  
> Mantén rigor: define términos, explica supuestos, distingue hechos vs inferencias, y señala limitaciones.  
> Entrega siempre: (1) respuesta para insertar en la tesis, (2) lista de afirmaciones + evidencia con páginas, (3) BibTeX válido para cada fuente, (4) preguntas de seguimiento si faltan datos.

Si no existe persona persistente, incluir ese bloque al inicio de **cada** consulta (ver plantilla).

---

## 4) Plantilla de consulta (1 llamada, salida rica)

Para cada pregunta tipo A, el agente debe enviar un prompt con esta estructura:

### 4.1 Encabezado fijo (rol + restricciones)

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
```

### 4.2 Bloque variable (contexto + pregunta)

Pegar SIEMPRE dos bloques:

1) **Contexto global compacto** (sección 0.1).
2) El campo completo `pregunta_reformulada_para_notebooklm`.

> `pregunta_reformulada_para_notebooklm` ya trae: **CONTEXTO**, **DECISIÓN**, **PREGUNTA**, **QUÉ BUSCAMOS**. El contexto global asegura que NotebookLM no pierda el “marco” (tu propuesta + tu pipeline) entre preguntas.

### 4.3 Contrato de salida (obligatorio)

```
FORMATO DE SALIDA (OBLIGATORIO):

1) RESPUESTA PARA INSERTAR (máx. 180–250 palabras)
   - Redacción lista para pegar en tesis (metodología).
   - Usa citas tipo \\cite{claveBib}.

2) AFIRMACIONES + EVIDENCIA (5–10 ítems)
   - Cada ítem: afirmación breve + (Fuente, año, páginas/ubicación).

3) BIBTEX (mínimo 3 fuentes, máximo 7)
   - Para cada fuente: entrega un bloque BibTeX.
   - Incluye pages cuando aplique; si no hay páginas, explica por qué.

4) NOTAS DEL ASESOR (3–6 bullets)
   - Objeciones típicas de jurado + cómo mitigarlas en el texto.

5) PREGUNTAS DE SEGUIMIENTO (0–3)
   - Solo si falta info crítica en los documentos.
```

---

## 5) “Quality gates” (validación automática/manual)

El agente solo marca una consulta como **exitosa** si:

1) Hay **≥3 fuentes** y cada una tiene **BibTeX**.
2) Las afirmaciones importantes incluyen **páginas/ubicación** (cuando el PDF lo permite).
3) La “Respuesta para insertar” usa `\\cite{...}` con claves consistentes.

### Si falla un gate → follow-up inmediato (máx. 2 follow-ups)

**Follow-up A (faltan páginas):**
> “Repite SOLO la sección ‘Afirmaciones + evidencia’ agregando páginas exactas o sección/figura. Si no es posible, indica la limitación del documento (sin paginación, etc.).”

**Follow-up B (BibTeX incompleto):**
> “Genera BibTeX válido para estas fuentes: [lista]. Incluye author, title, year y journal/publisher; añade DOI/URL si está en el documento.”

---

## 6) Estructura de salida recomendada (JSON)

Guardar resultados en un único archivo agregable:

```json
{
  "metadata": {
    "mcp_status": "conectado",
    "notebooks_encontrados": ["PDF_thesis_Psicoacustica", "..."],
    "archivo_preguntas": "metodologia_temporal/preguntas_CONTEXTUALIZADAS.json",
    "total_tipo_a": 37,
    "total_tipo_b": 5,
    "consultas_exitosas": 0,
    "consultas_fallidas": 0,
    "fecha": "YYYY-MM-DD"
  },
  "respuestas": [
    {
      "id": "Q-012",
      "tipo": "A",
      "tema": "psicoacustica",
      "notebook_usado": "PDF_thesis_Psicoacustica",
      "ubicacion_original": "Sección 3.2.1..., línea 111",
      "seccion_completa": "3.2.1 ...",
      "pregunta_reformulada": "....",
      "respuesta_notebooklm_raw": "....",
      "fuentes_bibliograficas": [
        {
          "key": "plompLevelt1965tonal",
          "cita": "Plomp, R. & Levelt, W. (1965)...",
          "pages": "pp. 12–15",
          "bibtex": "@article{...}\n..."
        }
      ],
      "para_redactor": { "donde_insertar": "...", "tipo_contenido": "definicion" }
    }
  ]
}
```

---

## 7) Consejos para exprimir “oro” (sin inflar tokens)

- Pedir **pocas fuentes pero fuertes** (3–7) con páginas, no listas infinitas.
- Forzar “Respuesta para insertar” corta + “Evidencia” separada (evita verborrea).
- Pedir que distinga **hecho en fuente** vs **inferencia** (reduce alucinación).
- Si hay debate (p.ej., “roughness como proxy”), pedir **2 posturas** + evidencia.
- Si el notebook no contiene una referencia clave, registrar eso como “gap” y sugerir subir el PDF.
