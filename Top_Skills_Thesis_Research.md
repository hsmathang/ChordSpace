# Top Skills para Tesis e Investigación Científica

Este documento recopila las "Joyas Ocultas" y herramientas más potentes instaladas en tu agente para potenciar tu tesis de maestría.
 además algunos archivos md
## 1. El "Cerebro": Metodología y Rigor Científico

Estas skills actúan como un tutor metodológico, ayudándote a pensar y estructurar antes de escribir.

### 🧠 `hypothesis-generation`
*   **Función Clave:** Formalización de Hipótesis.
*   **Por qué usarla:** No solo genera ideas. Te obliga a estructurar tu "Planteamiento del Problema" definiendo:
    *   **Mecanismos:** ¿Por qué ocurre X?
    *   **Predicciones Testables:** Si mi modelo funciona, ¿qué debería observar exactamente?
    *   **Explicaciones Rivales:** ¿Qué otras causas podrían explicar los mismos resultados?
*   **Comando sugerido:** "Ayúdame a formalizar la hipótesis de que mi modelo de sustitución armónica preserva la función tonal mejor que el azar."

### 🧐 `scientific-critical-thinking`
*   **Función Clave:** El "Abogado del Diablo".
*   **Por qué usarla:** Evalúa la solidez de tus propios argumentos. Antes de enviarle un capítulo a tu director, pásalo por esta skill.
*   **Capacidades:**
    *   Detecta **falacias lógicas** en tus conclusiones.
    *   Evalúa la **validez interna y externa** de tu diseño experimental.
    *   Identifica **confounding variables** que podrías estar ignorando.

### 🎓 `scholar-evaluation`
*   **Función Clave:** Simulación de Jurado de Tesis.
*   **Por qué usarla:** Utiliza el framework *ScholarEval* para calificar tu trabajo en dimensiones académicas:
    *   Formulación del problema.
    *   Rigor metodológico.
    *   Calidad de la escritura.
*   **Uso:** "Evalúa la sección de Metodología que acabo de escribir bajo los criterios de ScholarEval."

---

## 2. La "Pluma": Escritura Académica de Alto Nivel

Herramientas para transformar ideas y datos en prosa académica pulida.

### ✍️ `scientific-writing`
*   **Función Clave:** Redacción Estructurada (IMRaD).
*   **Joyas Ocultas:**
    *   **Conversion de Esquemas:** Le das un punteo de ideas (bullet points) y genera párrafos académicos fluidos con conectores lógicos.
    *   **Estilos de Revista:** Adapta el tono automáticamnte (ej. "Nature" vs "IEEE Technical Report").
    *   **Reporting Guidelines:** Sabe aplicar normas CONSORT, PRISMA, etc., asegurando que no olvides detalles obligatorios en tu reporte.

### 📚 `literature-review`
*   **Función Clave:** Síntesis Sistemática.
*   **Diferencia clave:** No solo "busca papers". Su fuerza está en la **síntesis temática**. En lugar de resumir paper por paper ("X dijo esto, Y dijo esto"), agrupa hallazgos por temas ("Tres estudios coinciden en A, mientras que B sugiere C...").
*   **Salida:** Genera reportes con citas verificadas y estructura lista para el capítulo de "Estado del Arte".

### 🏷️ `citation-management`
*   **Función Clave:** Higiene Bibliográfica.
*   **Por qué usarla:** Para auditar tu archivo `.bib`.
    *   Verifica que los DOIs resuelvan.
    *   Normaliza nombres de autores y revistas.
    *   Convierte formatos (APA, IEEE, Vancouver) sin errores.

---

## 3. La "Lente": Visualización y Comunicación

Una tesis sin buenos gráficos es ilegible.

### 🎨 `scientific-schematics`
*   **Función Clave:** Generación de Diagramas de Arquitectura y Flujo.
*   **USO OBLIGATORIO PARA TU TESIS:** Úsala para generar el diagrama de flujo de tu **Algoritmo de Sustitución** y la arquitectura de tu **Modelo Computacional**.
*   **Cómo funciona:** Describes el sistema en texto plano y genera el código (Mermaid/Graphviz) y la imagen renderizada profesionalmente.

---

## 📂 Recursos Listos en `metodologia_temporal/`

Se ha preparado una carpeta de "contexto rápido" con los documentos esenciales de tu tesis para alimentar estas skills:

1.  **Capítulos de Tesis (LaTeX):**
    *   `03Seccion03.tex`: (Probablemente tu Marco Teórico o Metodología preliminar).
    *   `04Seccion04.tex`: (Resultados o Desarrollo del Modelo).
    *   `metodologia.tex`: Archivo central de metodología.
2.  **Documentación Técnica del Repositorio:**
    *   `modelo_computacional_de_generacion_y_tratamiento_de_acordes_v_1.md`: Explicación detallada del algoritmo.
    *   `modelo_matematico_sustitucion_armonica.md`: Formalización matemática.
    *   `reporting_pipeline.md` y `FLUJO_DATOS_GUI.md`: Arquitectura técnica.
    *   `substitution_metrics.md`: Cómo mides el éxito.

**Sugerencia de Flujo de Trabajo Inmediato:**
1.  Usar `scientific-critical-thinking` para leer `metodologia.tex` y `modelo_computacional...md`.
2.  Pedirle que identifique "Gaps" o debilidades en la explicación del algoritmo.
3.  Usar `scientific-schematics` para proponer un diagrama que clarifique esos puntos confusos.
