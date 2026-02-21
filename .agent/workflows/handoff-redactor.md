---
description: Handoff completo (v2.1) para continuar redacción metodológica (Q&A → NotebookLM → redacción → inserción).
---

# 🔄 Handoff: Redactor Crítico de Metodología — ChordSpace

> **Propósito:** Retomar el flujo de redacción del Cap. 3 (Metodología) en cualquier momento, con **calidad consistente** y **sin degradación de contexto**.

---

## 1. ESTADO ACTUAL (Actualizado: 2026-02-17)

- **Capítulo:** `metodologia_temporal/ESTRUCTURA_MATEMATICA_DETALLADA.md` (~85 preguntas anotadas).
- **Progreso:** Q-003 a Q-028 ✅ RESUELTAS e INSERTADAS.
- **Siguiente:** **Q-029** (Definida Positiva / MDS Clásico).

---

## 2. ORDEN DE LECTURA (OBLIGATORIO)

Lee estos archivos **en este orden exacto** antes de operar:

| # | Archivo | Qué obtienes |
|---|---------|--------------|
| 1 | `metodologia_temporal/ESTRUCTURA_MATEMATICA_DETALLADA.md` | El capítulo. Busca `<!-- REDACTADO: Q-XXX -->` para ver el estilo. |
| 2 | `metodologia_temporal/preguntas_CONTEXTUALIZADAS.json` | El INPUT: preguntas, contexto y metadatos. |
| 3 | `metodologia_temporal/COMO_CONSULTAR_NOTEBOOKLM_MCP.md` | PROTOCOLO MCP: IDs, prompts fijos (§3.2). |
| 4 | `.agent/workflows/redactor-critico.md` | **WORKFLOW v2 (MASTER):** Fases -1 a 6 detalladas. |

---

## 3. WORKFLOW CRÍTICO v2 (RESUMEN EJECUTIVO)

**Problema Detectado:** Degradación del prompt en lotes grandes (pérdida de contexto).
**Solución v2:** Fases preventivas **-1** y **-0.5** OBLIGATORIAS.

### Fase -1: Contexto Dinámico Acumulativo
Crea un **Context Buffer** mental que crece con cada query:
```python
CONTEXT_BUFFER = {
  "contexto_base": [BLOQUE_FIJO_6_LINEAS_COMO_CONSULTAR_MCP], # INVIOLABLE
  "hallazgos_previos": ["Q-028: Topología depende de métrica...", ...], # Máx 8 últimos
  "decisiones": [],
  "terminos": []
}
```
**Acción:** Al terminar Q-N, extrae 1 hallazgo clave → Inyéctalo en el prompt de Q-N+1.

### Fase -0.5: Checklist de Calidad del Prompt
**Verificar ANTES de llamar a `notebook_query`:**
1. [ ] ¿Está el **CONTEXTO GLOBAL** (6 líneas) completo?
2. [ ] ¿El notebook primario tiene el template completo (§3.2)?
3. [ ] ¿Los secundarios tienen perspectiva adaptada?
4. [ ] ¿Se incluyó el **CONTEXTO ACUMULADO** (hallazgos previos)?
5. [ ] ¿Longitud del prompt OK?

### Fase 0: Verificación de IDs
Ejecutar `notebook_list` y confirmar IDs por TÍTULO.

### Fase 1: Consultas Paralelas (5 notebooks)
- Usar prompt enriquecido (Base + Contexto Acumulado).
- `timeout=180`.

### Fases 2-5: Redacción Crítica (ver `redactor-critico.md`)
- **Gate:** Filtrar evidencia (Nivel A/B/C).
- **Formalización:** Claim + Evidencia + Implicación.
- **Síntesis:** Argumentos convergentes (Triángulo de evidencia).
- **Redacción:** 150-250 palabras, prosa académica, citas `\cite{}`.

### Fase 6: Inserción
- Intercalar `<!-- REDACTADO: Q-XXX -->` en el `.md`.
- Actualizar `respuestas_notebooklm.json`.

---

## 4. NOTEBOOKS DE NOTEBOOKLM (IDs Verificados 2026-02-17)

| Clave | Título (BUSCAR SIEMPRE) | ID (Referencia) |
|-------|--------------------------|-----------------|
| `math` | PDFs_Tesis_Math_viz | `3f51c34d...` |
| `psico` | PDFs_Tesis_Psicoacustica | `27d02df9...` |
| `armonia` | PDFs_Tesis_Armonia | `8dedc0d4...` |
| `comp` | PDFs_Tesis_Computacion_ML | `14fa63f0...` |
| `dimred` | Reducción de Dimensionalidad | `43913228...` |

---

## 5. SIGUIENTE PREGUNTA: Q-029

```
ID: Q-029
Pregunta: "¿Es la matriz D definida positiva / semidefinida? (Relevante para MDS clásico; no necesariamente para SMACOF)."
Sección: §3.4.3
Tipo: definición/justificación
Tema sugerido: math
Insertar: §3.4.3 o §3.5.1
```

**Contexto Clave:**
- Disimilitudes no métricas (coseno) o no euclidianas (JSD, Manhattan) generan D que no es PSD.
- MDS Clásico (Strain) requiere D euclidiana exacta.
- SMACOF (Stress) permite D cualquiera, pero el stress no será cero.

---

## 6. PROMPT DE ARRANQUE

> Lee `.agent/workflows/handoff-redactor.md`. La siguiente pregunta es **Q-029**. Ejecuta el workflow completo `/redactor-critico` (v2) asegurando Fase -1 y -0.5 para prevenir degradación.

---

## 7. TIPS
- **Paralelismo:** Lanza las 5 queries juntas.
- **Gate:** Hazlo mentalmente, no gastes tokens.
- **Límite:** Si un archivo (e.g. `redactor-critico.md`) supera 12k chars, resume mentalmente sus instrucciones; no intentes leerlo todo si falla la carga. Prioriza la lógica del workflow v2.