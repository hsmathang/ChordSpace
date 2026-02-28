---
description: Revisión y reescritura de tesis con orientación correcta — sección por sección
---

# Workflow: Revisión y Reescritura de Tesis

## Contexto obligatorio (leer antes de comenzar)
- Thesis file: `docs/Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024 (1)/03Seccion03.tex`
- Orientation document: `docs/Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024 (1)/aclaraciones.md`
- TrackER de avance: artifact `task.md`

## Reglas del juego (invariantes globales)
1. El acorde es un **evento vertical aislado** — no hay voice-leading.
2. "Sustitución" = vecindad métrica d(Φ(c₁), Φ(c₂)) ≤ r — no función tonal.
3. EB va al **apéndice** — no es el núcleo del aporte.
4. Tymoczko/orbifolds/topología = **contexto del Estado del Arte**, no núcleo del modelo.
5. "Barroco" = conjunto de semillas de referencia, no dominio del modelo.

---

## Flujo por sección (ejecutar UNO A LA VEZ, en orden)

### Paso 1 — Preparación (una sola vez al inicio)
- Leer `aclaraciones.md` completo para tener orientación correcta.
- Leer la sección completa del `.tex` a trabajar.
- Marcar la sección como `[/]` en `task.md`.

### Paso 2 — Evaluación con scientific-writing skill (ANTES de proponer)
Aplicar la skill `scientific-writing` para evaluar el texto actual:
- ¿El texto promete algo que el trabajo no entrega? (Frases peligrosas)
- ¿Hay afirmaciones sin respaldo que necesiten cita? → Consultar NotebookLM vía MCP: preguntar autor + año que valide la afirmación.
- ¿Hay figuras/diagramas faltantes? La skill exige al menos 1 visual por sección relevante.
- ¿El flujo del párrafo es: Definición → Mapa/Operador → Propiedad/Justificación → Uso en pipeline?
- ¿Las frases son prosa fluida (no bullet points disfrazados)?

### Paso 3 — Redactar el plan de corrección
Escribir artifact `plan_seccionN.md` con:
- Tabla de problemas encontrados (gravedad 🔴/🟡/🟢, subsección, descripción).
- Texto LaTeX exacto propuesto para cada corrección.
- Propuesta de diagrama nuevo (si aplica) con descripción del visual.
- Preguntas para NotebookLM (si se necesitan citas).

### Paso 4 — Consulta a NotebookLM (si hay citas pendientes)
```
mcp_notebooklm_ask_question(
  question="¿Hay papers que respalden [afirmación concreta]? Citar autor y año.",
  session_id=<reusar sesión si existe>
)
```
Insertar resultado en el plan como `(Autor, año)` entre paréntesis en el texto propuesto.

### Paso 5 — Presentar plan al usuario para aprobación
Usar `notify_user` con:
- Tabla de problemas y gravedad.
- Resumen de cambios propuestos.
- Lista de citas buscadas en NotebookLM.
- Pregunta: "¿Apruebas? ¿Ajustas algo?"

⚠️ **NO EDITAR EL .TEX HASTA RECIBIR APROBACIÓN EXPLÍCITA**

### Paso 5b — Política de diagramas nuevos
- El documento usa **TikZ nativo** → todos los diagramas nuevos propuestos van en TikZ.
- Si el diagrama es muy complejo (redes neuronales, pathways biológicos): usar `generate_image` para una versión PNG complementaria.
- NO usar `scientific-schematics` a menos que se pida explícitamente (requiere API key externa).
- Todo diagrama nuevo debe tener: `\caption{...}` completa + `\label{fig:...}` + referencia desde el texto.

### Paso 6 — Edición del .tex (solo tras aprobación)
- Aplicar cambios con `multi_replace_file_content`.
- Si hay un diagrama nuevo: crear en TikZ dentro del `.tex`.
- Si había un resumen general faltante: agregarlo al inicio del capítulo (Paso 0 especial).

### Paso 7 — Verificación post-edición con scientific-writing skill
- Releer el fragmento editado.
- Verificar que ninguna corrección introdujo nuevas afirmaciones peligrosas.
- Verificar que el diagrama (si se añadió) tiene caption completa y referencia `\label`.
- Verificar que los axiomas/definiciones no se contradicen con otras secciones ya revisadas.

### Paso 8 — Cierre de sección
- Marcar sección como `[x]` en `task.md`.
- Notificar al usuario: resumen de lo que cambió + pregunta "¿avanzamos a la Sección N+1?"

---

## Orden de las secciones

| # | Sección | Líneas aprox. | Estado actual |
|---|---------|--------------|---------------|
| 0 | Introducción del capítulo + figura del pipeline | 1-53 | Pendiente |
| 1 | Conceptualización y Supuestos | 55-135 | ✅ Corregida (parcialmente — falta diagrama) |
| 2 | Formalización del Objeto Acorde | 79-121 | Pendiente |
| 3 | Modelo Psicoacústico de Rugosidad | 123-204 | Pendiente |
| 4 | Espacio Métrico y Transformaciones | 205-310 | Pendiente |
| 5 | Reducción de Dimensionalidad | 311-379 | Pendiente |
| 6 | Diseño Experimental y Validación | 380-614 | Pendiente |
| 7 | Protocolos de Validación Específicos | 615-772 | Pendiente |
| 8 | Reproducibilidad y Consideraciones Finales | 773-844 | Pendiente |

---

## Diagrama pendiente (deuda técnica de Sección 0 / introducción del capítulo)
La introducción del capítulo necesita una **figura de resumen del proceso metodológico** (pipeline de pasos, no de supuestos).
Los supuestos y axiomas quedan como párrafo de prosa — no necesitan diagrama propio.
El diagrama de proceso iría en la **Sección 0 / introducción del capítulo** para orientar al lector antes de entrar al detalle.

---

## Frases prohibidas (revisar en cada sección)
```
❌ "resolvemos el problema topológico de..."
❌ "la topología intrínseca del espacio armónico"
❌ "orbifold" (en contexto de nuestro modelo — solo en Estado del Arte)
❌ "sustitución barroca" (sin aclaración fuerte de que es referencia, no dominio)
❌ "métrica EB" como aporte central
❌ "el modelo es universal/invariante"
```

## Frases permitidas (refuerzan la narrativa correcta)
```
✅ "el acorde se analiza como evento vertical aislado"
✅ "definimos sustitución como vecindad en el espacio de características"
✅ "la reducción dimensional se usa para exploración visual"
✅ "la métrica de sustitución opera en 12D"
✅ "nuestro universo está restringido a 12-TET (supuesto)"
✅ "el modelo es condicionado al espectro/timbre asumido (supuesto)"
```
