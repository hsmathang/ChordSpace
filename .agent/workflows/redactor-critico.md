---
description: Flujo de redacción crítica (v2-optimized) para metodología. Incluye fases de contexto dinámico y checklist anti-degradación.
---

# Redactor Crítico — Flujo de Trabajo (Optimizado v2)

> **Objetivo:** Convertir respuestas de NotebookLM en texto insertable para el Cap. 3 (Metodología), evaluando relevancia y evitando degradación de contexto.
> **Principio:** Resolver cada pregunta con sencillez y rigor metodológico.

---

## Archivos I/O

| Archivo | Rol |
|---|---|
| `preguntas_CONTEXTUALIZADAS.json` | INPUT: Preguntas + Contexto |
| `respuestas_notebooklm.json` | INPUT: Respuestas + BibTeX |
| `ESTRUCTURA_MATEMATICA_DETALLADA.md` | TARGET: Capítulo |
| `COMO_CONSULTAR_NOTEBOOKLM_MCP.md` | REF: Protocolo MCP |

---

## FASE -1: CONTEXTO DINÁMICO ACUMULATIVO (CRÍTICO v2)

> **Problema:** Degradación del prompt en lotes.
> **Solución:** `CONTEXT_BUFFER` creciente.

Antes de procesar, inicializar (mentalmente):
```python
buffer = {
  "base": [BLOQUE_FIJO_6_LINEAS], # Inviolable
  "hallazgos": [], # De Q anteriores
  "decisiones": [],
  "terminos": []
}
```

**Acción:** 
1. Al terminar Q-N, extraer **1 hallazgo clave** y añadirlo al buffer.
2. Al iniciar Q-N+1, inyectar `hallazgos` (máx 8) en el prompt.

---

## FASE -0.5: CHECKLIST DE CALIDAD (CRÍTICO v2)

**Verificar ANTES de enviar query:**
1. [ ] ¿Contexto Global (6 líneas) completo?
2. [ ] ¿Notebook primario con template completo?
3. [ ] ¿Notebooks secundarios con perspectiva adaptada?
4. [ ] ¿Contexto Acumulado incluido?
5. [ ] ¿Longitud del prompt OK?

**Si falla algo → CORREGIR.**

---

## FASE 0: Carga y Orientación

1. Leer `respuestas_notebooklm.json` (ID objetivo).
2. Leer `preguntas_CONTEXTUALIZADAS.json` (ubicación, contexto, tipo).
3. Leer sección en `ESTRUCTURA_MATEMATICA_DETALLADA.md` (texto vecino, notas).
4. Identificar objetivo comunicativo de la sección.

---

## FASE 1: Gate de Relevancia `scientific-critical-thinking`

Clasificar afirmaciones de los 5 notebooks:
- **Nivel A (Esencial):** Responde directo + fuente de calidad. → **INCLUIR**
- **Nivel B (Refuerzo):** Apoya útilmente. → **Reserva**
- **Nivel C (Tangencial):** Se desvía. → **DESCARTAR**

Evaluar proporcionalidad: ¿La fuerza del claim coincide con la evidencia?

---

## FASE 2: Formalización `hypothesis-generation`

Formular **un solo claim** central:
```
CLAIM: [Afirmación que resuelve la pregunta]
PORQUE: [Razón 1 - fuente] + [Razón 2 - fuente]
IMPLICACIÓN: [Efecto en el modelo ChordSpace]
```
Verificar: ¿Es falsificable? ¿Es parsimonioso? (Si se puede simplificar, hazlo).

---

## FASE 3: Síntesis `literature-review`

Organizar por **ARGUMENTO** (no por notebook):
- Arg 1: [Tema] → Fuentes X, Y
- Arg 2: [Tema] → Fuentes A, B

**Regla:** Triángulo de evidencia (mínimo 2 fuentes para argumento central).
Seleccionar BibTeX SOLO de fuentes usadas.

---

## FASE 4: Redacción `scientific-writing`

**Reglas:**
1. **Español académico.**
2. **Prosa fluida** (CERO bullets).
3. **Citas integradas:** `\cite{Clave}`.
4. **Longitud:** 150-250 palabras.
5. **Notación consistente:** ($\mathcal{N}, \Phi, etc$).
6. **Sin comillas LaTeX:** Usar `"texto"` (no backticks).

**Templates:**
- *Justificación:* "[Decisión] se sustenta en [razón perceptual] \cite{A} y [razón computacional] \cite{B}. Por tanto, [implicación]."
- *Definición:* "[Contexto]. [Definición formal]. [Interpretación]. \cite{C}."
- *Comparación:* "Existen [X, Y]. Se elige [X] por [razón 1] \cite{D}. [Y] se descarta por [razón 2]."

**Checklist Calidad:**
- [ ] ¿Responde sin rodeos?
- [ ] ¿Notación consistente?
- [ ] ¿Dentro del límite de palabras?
- [ ] ¿Legible para matemático?

---

## FASE 5: Inserción y Registro

1. **Insertar:** En `ESTRUCTURA_MATEMATICA_DETALLADA.md` tras la pregunta.
   - Usar bloque `<!-- REDACTADO: Q-XXX ... -->`.
   - Citas tipo `>`.
2. **Registrar:** Actualizar `respuestas_notebooklm.json` con metadata (palabras, fuentes usadas/descartadas).

---

## Principios
1. **Brevedad > Exhaustividad.**
2. **Metodología > Teoría.**
3. **Evidencia > Opinión.**
4. **Contexto nunca decrece (v2).**
5. **Calidad Prompt = Calidad Respuesta (v2).**
