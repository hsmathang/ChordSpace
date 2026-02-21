---
description: Flujo de redacción académica humana para evadir detección de IA (Control de Perplejidad y Ráfaga)
---

# Humanizador Académico: Protocolo Anti-Detección IA

Este workflow está diseñado para procesar textos académicos generados por inteligencia artificial y reescribirlos garantizando un tono natural, humano y científico, minimizando al 0% la probabilidad de detección algorítmica.

## Requisitos Previos
Debes aplicar las skills: `scientific-writing`, `scholar-evaluation`, y `scientific-critical-thinking`.
**Regla de Oro:** NO pierdas NINGUNA etiqueta de citación o formato de LaTeX (ej. `\cite{}`, `\label{}`, `\textbf{}`, `\textit{}`, entornos de ecuaciones). Mantenlos intactos.

## Flujo de Trabajo Párrafo a Párrafo

Para cada párrafo del documento fuente, ejecuta EXACTAMENTE los siguientes 5 pasos:

### Paso 1: Lectura, Aislamiento y Poda (Comprensión)
- Lee el párrafo generado por la IA para entender el mensaje central subyacente.
- Identifica y ELIMINA el "relleno" semántico: información redundante, agrupaciones de ideas que suenan lógicas pero carecen de profundidad real, y cláusulas introductorias vacías.

### Paso 2: Reescritura a Ciegas (Eludir el anclaje sintáctico)
- NO edites palabra por palabra sobre la estructura original.
- Reescribe la idea central desde cero, utilizando tus propios patrones lingüísticos (simulando a un humano de carne y hueso). Esto destruye automáticamente la estructura predictiva de la IA.

### Paso 3: Romper la Monotonía Estructural (Estrategia de Ráfaga / Burstiness)
- **Variación de Longitud:** Intercala oraciones muy cortas (de impacto o declarativas) con oraciones largas y subordinadas.
- **Sintaxis Invertida:** Evita que todas las oraciones sigan el patrón Sujeto + Verbo + Predicado. Usa cláusulas dependientes al inicio (ej. "Aunque los resultados varían, este modelo...").
- **Destrucción de Listas Robóticas:** La IA ama la estructura "A y B" o listas de tres elementos. Rompe este patrón. En lugar de "El modelo es rápido, eficiente y preciso", elabora: "La velocidad del modelo es notable. Esto, sumado a su precisión, lo hace altamente eficiente."
- **Voz Activa:** Pasa construcciones pasivas a activas, o usa la primera persona del plural (ej. "Evaluamos" en lugar de "Fue evaluado").

### Paso 4: Poda Léxica y Factor Humano (Estrategia de Perplejidad y Tono)
- **Cero Vocabulario IA (Lista Negra):** ELIMINA y proscribe el uso de palabras como: *ahondar, aprovechar, robusto, tapiz, panorama, crucial, optimizar, transformación*. Usa alternativas más sencillas (explorar, usar, sólido, sistema, clave).
- **Transiciones Conversacionales:** Evita conectores robóticos al inicio de párrafo ("Además", "Por lo tanto", "En conclusión", "Es importante destacar que"). Usa conectores fluidos ("pero", "así que", "por eso", "sin embargo" -pero en medio de la oración-).
- **Duda Intelectual (Hedging):** Baja el tono de las afirmaciones categóricas. Usa matices propios del investigador humano: *parece que, sugiere, existe la posibilidad de que, es probable que, tendería a*.
- **Crítica Sutil e Imperfecciones:** Introduce pequeñas reflexiones críticas o señala inconsistencias del modelo. Si el contexto lo permite, admite limitaciones ("este enfoque presenta desafíos evidentes").

### Paso 5: Prueba de Metrónomo y Ensamblaje LaTeX
- Verifica el ritmo mentalmente (simula lectura en voz alta). Si el ritmo es excesivamente regular y monótono (como un metrónomo), une o divide oraciones urgentes.
- Reinserta todos los comandos LaTeX (`\cite{}`, `\citep{}`, `\dots`, ecuaciones) exactamente donde corresponden semánticamente en el nuevo texto.

---
**Instrucción de Ejecución:**
Para aplicar este workflow, procesa el documento en bloques lógicos (ej. por secciones). Imprime la versión final formateada en `Markdown` (o `.tex` según sea solicitado), garantizando que el texto resultante sea denso, académicamente riguroso, pero con un "acento" 100% humano.
