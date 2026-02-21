# GUÍA VISUAL ULTRA-SIMPLE

## 🎯 LO QUE VAS A HACER (3 pasos)

```
┌─────────────────────────────────────────────────────────┐
│ PASO 1: VERIFICAR MCP                                   │
│                                                          │
│ Acción: Lista recursos MCP disponibles                  │
│                                                          │
│ ¿Ves "PDFs_Tesis_*"?                                    │
│    ├─ SÍ  → ✅ Continúa a PASO 2                        │
│    └─ NO  → ❌ DETENTE y reporta error                  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ PASO 2: LEER JSON                                       │
│                                                          │
│ Archivo: preguntas_identificadas.json                   │
│ Contiene: 42 preguntas ya listas                        │
│                                                          │
│ Para cada pregunta del JSON → PASO 3                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ PASO 3: CONSULTAR NOTEBOOKLM (Repetir 42 veces)        │
│                                                          │
│ Para pregunta Q-001:                                     │
│   1. Toma "notebook_sugerido"                           │
│   2. Toma "pregunta_formulada"                          │
│   3. Consulta vía MCP                                    │
│   4. Extrae respuesta REAL de NotebookLM                │
│   5. Extrae citas bibliográficas                        │
│   6. Rellena JSON de salida                             │
│                                                          │
│ Repite para Q-002, Q-003... hasta Q-042                 │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ RESULTADO FINAL                                         │
│                                                          │
│ Archivo: respuestas_completadas.json                    │
│ Contiene: 42 respuestas REALES con citas               │
└─────────────────────────────────────────────────────────┘
```

---

## ❌ LO QUE NO DEBES HACER

```
❌ NO hagas análisis del documento ESTRUCTURA_MATEMATICA_DETALLADA.md
   → Ya está hecho. Usa el JSON.

❌ NO filtres comentarios tipo A vs tipo B
   → Ya está filtrado. Usa el JSON.

❌ NO busques preguntas en el texto
   → Ya están identificadas. Usa el JSON.

❌ NO generes templates vacíos
   → Consulta MCP y rellena con respuestas REALES.

❌ NO cites "NotebookLM" como fuente
   → Extrae las fuentes ORIGINALES (papers, libros).
```

---

## ✅ LO QUE SÍ DEBES HACER

```
✅ Verificar MCP primero
✅ Leer preguntas_identificadas.json
✅ Para CADA pregunta:
   → Consultar NotebookLM vía MCP
   → Copiar respuesta textual completa
   → Extraer citas bibliográficas
   → Rellenar JSON de salida
✅ Entregar respuestas_completadas.json con 42 respuestas REALES
```

---

## 🔍 EJEMPLO DE UNA ITERACIÓN

### Input (del JSON):
```json
{
  "id": "Q-007",
  "pregunta_formulada": "¿Qué es la banda crítica auditiva? ¿Qué es la disonancia sensorial?",
  "notebook_sugerido": "PDFs_Tesis_Psicoacustica"
}
```

### Acción MCP:
```
1. Conectar a notebook: PDFs_Tesis_Psicoacustica
2. Enviar consulta: "¿Qué es la banda crítica auditiva?... [con requisitos de citación]"
3. Recibir respuesta de NotebookLM (texto + citas)
```

### Output (para JSON final):
```json
{
  "id": "Q-007",
  "mcp_status": "exitoso",
  "respuesta_notebooklm": "La banda crítica es un rango de frecuencias... [texto completo]",
  "fuentes_bibliograficas": [
    {
      "cita_textual": "Plomp, R., & Levelt, W. J. M. (1965). Tonal consonance...",
      "bibtex": "@article{plomp1965tonal,\n  author = {Plomp, R. and Levelt, W. J. M.},\n  ...\n}"
    }
  ]
}
```

### ¡Listo! Repite para Q-008, Q-009... hasta Q-042

---

## 🚨 SEÑALES DE ALERTA

### Si ves esto → DETENTE:
- ❌ Error MCP al listar recursos
- ❌ No encuentras notebooks "PDFs_Tesis_*"
- ❌ Consulta MCP retorna error en >3 intentos
- ❌ Respuesta de NotebookLM está vacía

### Si ves esto → Estás bien:
- ✅ MCP responde con lista de notebooks
- ✅ Consultas retornan texto completo
- ✅ Respuestas incluyen citas bibliográficas
- ✅ JSON de salida se va llenando con respuestas reales

---

## 📊 CHECKLIST DE PROGRESO

```
[Paso 1] MCP verificado
  ├─ [ ] Listado de recursos ejecutado
  ├─ [ ] Notebooks PDFs_Tesis_* encontrados
  └─ [ ] Status reportado

[Paso 2] JSON leído
  ├─ [ ] Archivo preguntas_identificadas.json cargado
  ├─ [ ] 42 preguntas identificadas
  └─ [ ] Estructura entendida

[Paso 3] Consultas MCP (42/42)
  ├─ [ ] Q-001 completada
  ├─ [ ] Q-002 completada
  ├─ [ ] ...
  └─ [ ] Q-042 completada

[Output] Archivo final
  ├─ [ ] respuestas_completadas.json generado
  ├─ [ ] 42 respuestas con texto real
  ├─ [ ] Todas con fuentes bibliográficas
  └─ [ ] Ninguna cita a "NotebookLM"
```

---

## 🎯 TU OBJETIVO FINAL

**Entregar UN archivo:**
- `respuestas_completadas.json`
- Con 42 respuestas COMPLETAS
- Cada una con texto REAL de NotebookLM
- Cada una con citas bibliográficas de fuentes originales
- Metadata mostrando: 42/42 consultas exitosas

**Eso es todo. Nada más.** 🚀
