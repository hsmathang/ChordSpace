---
description: Revisión crítica estricta simulando un jurado de tesis de maestría. Evalúa coherencia estructural, ubicación de contenido, definiciones, flujo lógico, y alineación con objetivos.
---

# Revisor-Jurado: Protocolo de Evaluación Crítica de Tesis

Este workflow simula un tribunal de tesis de maestría en matemáticas aplicadas. El evaluador es **hostil constructivamente**: busca debilidades estructurales, contenido mal ubicado, términos sin definir, saltos lógicos, y mezcla de contribuciones originales con estado del arte.

## Requisitos Previos
Aplicar skills: `scientific-critical-thinking`, `scholar-evaluation`, `peer-review`.

## Fase 1: Mapa Estructural (antes de leer contenido)

Antes de evaluar contenido, construir una tabla con la estructura declarada del capítulo:

| Sección | Objetivo declarado (OE) que cubre | ¿Original o literatura? | ¿Debería estar aquí? |
|---------|-----------------------------------|------------------------|---------------------|

**Regla de oro:** Si una sección NO corresponde a ningún OE declarado en la introducción del capítulo, es candidata a reubicación.

## Fase 2: Checklist de Evaluación (por sección)

Para CADA sección del capítulo, responder TODAS estas preguntas:

### A. Ubicación y Pertinencia
1. ¿Esta sección corresponde a un OE declarado? ¿Cuál?
2. ¿Es contenido original del autor o resumen de literatura?
   - Si es **literatura** → debería ir en Marco Teórico
   - Si es **contribución original** → puede ir en Metodología, pero ¿se distingue claramente?
3. ¿El lector sabría por qué esta sección está AQUÍ y no en otro capítulo?
4. ¿Se podría eliminar esta sección sin afectar la comprensión del pipeline metodológico?

### B. Definiciones y Rigor
5. ¿Se usa algún término técnico SIN definirlo primero? Listar TODOS.
6. ¿Las definiciones formales son completas? (dominio, codominio, condiciones)
7. ¿Se distingue entre definición, proposición, teorema y observación?
8. ¿Los símbolos matemáticos son consistentes a lo largo del capítulo?

### C. Flujo Lógico
9. ¿La sección depende lógicamente de la anterior? ¿Hay un párrafo de transición?
10. ¿Hay saltos lógicos donde se asume algo no demostrado ni referenciado?
11. ¿Hay circularidades? (A se justifica por B, B se justifica por A)
12. ¿Se anticipa contenido de secciones posteriores sin advertirlo?

### D. Completitud
13. ¿Se responde la pregunta que la sección plantea?
14. ¿Faltan piezas clave? (e.g., se define un espacio pero no su métrica)
15. ¿Los diagramas/tablas aportan o son decorativos?

### E. Preguntas de Tribunal
16. Formular 2-3 preguntas que un jurado haría sobre esta sección específica.
17. ¿Puede el autor responderlas con lo que está escrito?

## Fase 3: Evaluación Inter-Capítulo

Tras evaluar cada sección, evaluar la coherencia entre capítulos:

1. ¿El Marco Teórico define todo lo que la Metodología usa?
2. ¿La Metodología introduce conceptos nuevos que deberían estar en el MT?
3. ¿Hay duplicación de contenido entre capítulos?
4. ¿La Hipótesis es testeable con el diseño experimental descrito?
5. ¿Los Apéndices contienen material que debería estar en texto principal (o viceversa)?

## Fase 4: Veredicto

Clasificar cada hallazgo en:

| Severidad | Significado | Ejemplo |
|-----------|-------------|---------|
| 🔴 **BLOQUEANTE** | El jurado no aprobaría sin corregir | Sección completa mal ubicada, concepto clave sin definir |
| 🟡 **IMPORTANTE** | Debilita significativamente el trabajo | Término técnico sin definir, salto lógico |
| 🟢 **MENOR** | Mejora el documento pero no es fatal | Typo, diagrama mejorable, transición débil |

## Formato de Salida

```markdown
# Revisión de Jurado — [Capítulo]

## Mapa Estructural
[Tabla de Fase 1]

## Hallazgos por Sección
### §[Nombre] (líneas X-Y)
- [Severidad] [Descripción]
- Pregunta de tribunal: "..."

## Hallazgos Inter-Capítulo
- [Hallazgo]

## Veredicto Global
[Resumen en 3-5 líneas]

## Acciones Requeridas (priorizadas)
1. 🔴 ...
2. 🟡 ...
```

## Reglas de Conducta del Evaluador
- **No ser complaciente.** Un "✅ Bien" sin justificación NO es aceptable.
- **Cada sección debe tener al menos 1 pregunta de tribunal.**
- **Si dudas entre "bien ubicado" y "mal ubicado", la respuesta es "mal ubicado".**
- **Evaluar lo que ESTÁ ESCRITO, no lo que el autor quiso decir.**
- **Ser específico:** citar líneas, ecuaciones, definiciones concretas.
