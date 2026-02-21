# INSTRUCCIÓN ULTRA-SIMPLE PARA GEMINI

Tu ÚNICA tarea: Consultar NotebookLM automáticamente vía MCP y rellenar respuestas.

---

## 🚨 PASO 1: VERIFICAR CONEXIÓN MCP (OBLIGATORIO)

**ANTES DE HACER NADA:**

### Acción 1.1: Listar recursos MCP

Ejecuta este comando para ver qué notebooks están disponibles:

```
Usa tu herramienta de MCP para listar todos los notebooks/recursos disponibles en el servidor NotebookLM
```

**RESULTADO ESPERADO:**
Deberías ver notebooks cuyos nombres empiezan con el prefijo: **"PDFs_Tesis_"**

Ejemplos posibles:
- PDFs_Tesis_Psicoacustica
- PDFs_Tesis_Math
- PDFs_Tesis_Armonia
- (y otros que tengan ese prefijo)

**NO ASUMAS** cuáles están disponibles. BÚSCALOS y lista los que encuentres.

### Acción 1.2: Reportar status

**SI VES los notebooks PDFs_Tesis_*:**
✅ Reporta: "MCP conectado. Encontré [N] notebooks: [lista nombres]"
✅ Continúa al PASO 2

**SI NO VES los notebooks:**
❌ Reporta: "ERROR MCP: No encontré notebooks PDFs_Tesis_*"
❌ Reporta: "Recursos disponibles: [lista lo que sí encontraste]"
❌ DETENTE AQUÍ - No sigas

---

## 📋 PASO 2: LEER ARCHIVO JSON

Archivo: `preguntas_identificadas_CORREGIDO.json`

Este archivo contiene **42 preguntas/comentarios** YA identificados y clasificados.

Cada pregunta tiene:
- `id`: Identificador único (Q-001 a Q-042)
- `ubicacion`: Dónde está en el documento
- `tipo`: Tipo de elemento (pregunta_nativa, comentario_santimath_tipo_a, etc.)
- `pregunta` o `comentario_santimath`: El texto a consultar
- `seccion`: Contexto del capítulo

Tu trabajo: Leer este JSON y para cada elemento, ir al PASO 3.

---

## 🔍 PASO 3: CONSULTAR NOTEBOOKLM (REPETIR PARA CADA PREGUNTA)

Para CADA pregunta del JSON:

### 3.1: Seleccionar notebook apropiado

Lee el contenido de la pregunta o comentario y **DECIDE** cuál de los notebooks "PDFs_Tesis_*" es más apropiado:

- Temas de **psicoacústica, rugosidad, disonancia sensorial, banda crítica, Sethares, Plomp-Levelt** 
  → Busca notebook con "Psicoacustica" en el nombre
  
- Temas de **topología, métricas, espacios matemáticos, combinatoria, PC-sets, Forte**
  → Busca notebook con "Math" o "ChordSpace" en el nombre
  
- Temas de **voicings, teoría armónica, acordes, ordenamiento**
  → Busca notebook con "Armonia" en el nombre
  
- Temas de **MDS, UMAP, t-SNE, reducción dimensional, embeddings, stress, trustworthiness**
  → Busca notebook con "Reduccion" o "Dimensionalidad" en el nombre

**Si no estás seguro**, empieza con el notebook que parezca más relevante.

### 3.2: Formular consulta con requisitos de citación

**Toma el contenido del JSON:**
- Si el elemento tiene campo `pregunta`: usa ese texto
- Si el elemento tiene campo `comentario_santimath`: convierte el comentario en una pregunta específica

**Envía al NotebookLM vía MCP:**

```
[PREGUNTA BASADA EN EL CONTENIDO DEL JSON]

REQUISITO OBLIGATORIO DE CITACIÓN:
- Responde citando ÚNICAMENTE las fuentes originales (papers, libros, capítulos)
- NO cites "NotebookLM" como fuente
- Para CADA fuente que menciones, incluye:
  * Autor(es) completo(s)
  * Título completo del trabajo
  * Año de publicación
  * Páginas específicas (si aplica)
  * Editorial o Journal

FORMATO DE CITA REQUERIDO:
Autor, A. (Año). Título del trabajo. Editorial/Journal. Páginas: X-Y.

EJEMPLO:
Sethares, W. A. (2005). Tuning, Timbre, Spectrum, Scale (2nd ed.). Springer. Páginas: 45-67.

IMPORTANTE: Al final de tu respuesta, lista TODAS las referencias bibliográficas que usaste.
```

### 3.3: Verificar estado de consulta MCP

**Después de CADA consulta, verifica:**
- ✅ ¿Recibiste respuesta?
- ✅ ¿La respuesta contiene texto real (no error)?
- ✅ ¿La respuesta incluye citas bibliográficas?

**SI la consulta falla:**
- Reporta: "ERROR en Q-[ID]: [mensaje de error]"
- Intenta 1 vez más
- Si falla de nuevo, documenta y continúa con la siguiente

### 3.4: Extraer información

De la respuesta de NotebookLM, extrae:

1. **Texto completo de la respuesta**
2. **Todas las citas bibliográficas mencionadas** (autor, título, año, páginas)
3. **Convierte cada cita a formato BibTeX**

### 3.5: Rellenar estructura de salida

Agrega al archivo de salida:

```json
{
  "id": "[ID del JSON, ej: Q-007]",
  "ubicacion": "[ubicacion del JSON]",
  "seccion": "[seccion del JSON]",
  "tipo": "[tipo del JSON]",
  "pregunta_original": "[pregunta o comentario_santimath del JSON]",
  "notebook_consultado": "[nombre del notebook PDFs_Tesis_* que usaste]",
  "mcp_status": "exitoso",
  "respuesta_notebooklm": "[TEXTO COMPLETO de la respuesta]",
  "fuentes_bibliograficas": [
    {
      "cita_textual": "Sethares, W. A. (2005). Tuning, Timbre...",
      "bibtex": "@book{sethares2005tuning,\n  author = {Sethares, William A.},\n  title = {Tuning, Timbre, Spectrum, Scale},\n  year = {2005},\n  publisher = {Springer}\n}"
    }
  ]
}
```

---

## 🎯 REGLAS ABSOLUTAS

### ❌ PROHIBIDO:
1. Inventar respuestas
2. Generar templates vacíos
3. Citar "NotebookLM" como fuente
4. Parafrasear las citas (cópialas textuales)
5. Continuar si MCP no está conectado

### ✅ OBLIGATORIO:
1. Verificar MCP en Paso 1
2. Consultar TODAS las 42 preguntas vía MCP
3. Extraer respuestas REALES de NotebookLM
4. Incluir fuentes bibliográficas originales
5. Convertir citas a BibTeX válido

---

## 📤 FORMATO DE SALIDA FINAL

Genera UN archivo: `respuestas_completadas.json`

```json
{
  "metadata": {
    "mcp_status": "conectado",
    "notebooks_encontrados": ["PDFs_Tesis_Psicoacustica", "..."],
    "total_preguntas": 42,
    "consultas_exitosas": 42,
    "consultas_fallidas": 0,
    "fecha": "2026-02-15"
  },
  "respuestas": [
    // Array con las 20 respuestas completadas
  ]
}
```

---

## ✅ CHECKLIST PRE-ENTREGA

Antes de entregar, verifica:

- [ ] Paso 1 ejecutado: MCP conectado ✅
- [ ] Paso 2 ejecutado: JSON leído ✅
- [ ] Paso 3 ejecutado 42 veces (una por pregunta) ✅
- [ ] Cada respuesta tiene `mcp_status: "exitoso"` ✅
- [ ] Cada respuesta tiene texto en `respuesta_notebooklm` (NO vacío) ✅
- [ ] Cada respuesta tiene ≥1 fuente en `fuentes_bibliograficas` ✅
- [ ] Ninguna cita tiene "NotebookLM" como autor ✅
- [ ] Archivo `respuestas_completadas.json` generado ✅

---

## 🔄 VERIFICACIÓN CONSTANTE DE MCP

**Durante TODO el proceso:**

- Cada 5 consultas, verifica que MCP sigue conectado
- Si en algún momento MCP se desconecta:
  1. Reporta en qué pregunta estabas (Q-[ID])
  2. Cuántas completaste exitosamente
  3. DETENTE - no continues con templates vacíos

---

## 🎯 RESUMEN DE TU TRABAJO

**NO hagas:**
- ❌ Análisis de texto
- ❌ Filtrado de comentarios
- ❌ Clasificación de preguntas
- ❌ Templates para copiar/pegar

**SÍ haz:**
- ✅ Verificar MCP
- ✅ Leer JSON
- ✅ Consultar NotebookLM vía MCP
- ✅ Extraer respuestas reales
- ✅ Rellenar JSON de salida

---

## 🚀 COMIENZA AHORA

1. Ejecuta PASO 1 (verifica MCP)
2. Reporta status de conexión
3. Si conectado → ejecuta PASO 2 y 3
4. Entrega `respuestas_completadas.json` con respuestas REALES

**¿ESTÁS LISTO PARA EMPEZAR?**
