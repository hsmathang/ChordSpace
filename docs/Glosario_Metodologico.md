# Glosario Metodológico Explicativo: Requisitos por Sección

Este documento detalla, sección por sección, los conceptos que deben estar cubiertos en el Marco Teórico para que el lector (o el jurado) comprenda a cabalidad las decisiones tomadas en el capítulo de Materiales y Métodos (y posteriormente Resultados).

---

## Estructura de Subsecciones a Cubrir (Capítulo 3: Materiales y Métodos)
1. **Introducción del Capítulo**
2. **3.1 Problema, Alcance y Supuestos** (Justificación Dimensional, Supuestos Psicoacústicos)
3. **3.2 Definición Matemática del Acorde** (Espacio de notas, Pitch-Chord, Vectores)
4. **3.3 Modelo Psicoacústico de Rugosidad** (Plomp-Levelt, Sethares, Espectro)
5. **3.4 Representación 12D, Normalización y Distancias** (Simplex, D. Euclidiana, D. Coseno)
6. **3.5 Reducción Dimensional para Visualización** (MDS, UMAP, Stress)
7. **3.6 Diseño Experimental** (Combinatoria, Filtros)
8. **3.7 Protocolos de Validación** (Validación Cruzada OOS, Regresión Ridge, Pearson, $R^2$)
9. **3.8 Reproducibilidad Computacional y Límites** (Complejidad Algorítmica, Big O)

---

## 1. Introducción general del Capítulo 03

**Recomendación analítica:** Para leer la introducción del capítulo de metodología, el lector necesita entender el cambio de paradigma conceptual del modelo. Se abandona la conducción de voces por la evaluación acústica vertical.

**Conceptos que el lector DEBE tener claros en el marco teórico previo:**
1.  **Similitud Sonora vs. Similitud Tonal Clásica:** El lector debe entender que tradicionalmente un acorde se clasifica por su "función" (Tónica, Dominante) y "conducción de voces" (*voice-leading*). El marco teórico DEBE explicar que la similitud sonora abordada aquí es un fenómeno puramente físico y perceptual, ignorando el contexto circundante.
    *   *Razón:* Porque desde el inicio de la sección 3 se dice que el acorde se tratará como "conjunto de notas independientes del contexto armónico".
2.  **Harmonicidad y Rugosidad (A nivel general):** Conceptos globales. La rugosidad como la interferencia en bandas críticas que causa disonancia sensorial.
    *   *Razón:* Se enuncia como la justificación principal para construir la hipótesis de la representabilidad espacial del acorde desde la percepción de disonancia física.
3.  **Reducción Dimensional (A nivel de propósito visual):** ¿Para qué proyectar grandes datos en 2D?
    *   *Razón:* Mencionado en la figura del flujo conceptual. El lector debe saber empíricamente qué es $\mathbb{R}^{12}$ (dimensión alta no dibujable) y qué es proyectar a $\mathbb{R}^2$ (plano bidimensional dibujable) para exploración cartográfica musical.

---

## 2. Sección 3.1: Problema, Alcance y Supuestos

Esta sección aterriza el alcance: se trata acústica y no contrapunto; se asume 12-TET; se trabaja con timbres sintéticos.

**Conceptos que el lector DEBE tener claros en el marco teórico previo:**
1.  **Métrica Espacial (Topología Básica):** ¿Qué significa que "la sustitución armónica sea una vecindad métrica"?
    *   *Razón:* La sección postula formalmente que $c_2 \in \mathrm{Subs}_r(c_1) \iff d(\Phi(c_1),\Phi(c_2)) \leq r$. El lector debe entender que la disimilitud musical se evalúa como una *distancia matemática* ($d$) dentro de una burbuja de radio $r$. Esto debe estar soportado en el marco estadístico/matemático.
2.  **Temperamento Igual de 12 Tonos (12-TET):** 
    *   *Razón:* Se declara como supuesto técnico. El lector no puede adivinar qué es esto. El marco teórico debe definirlo como el sistema que divide la octava en 12 semitonos idénticos, a diferencia de las afinaciones justas. Es el suelo fundamental que discretiza el modelo.
3.  **Espectro Armónico y Decaimiento Exponencial (Timbre):** 
    *   *Razón:* La sección 3.1 menciona un "Timbre sintético con espectro armónico idealizado ($A_n = \delta^{n-1}$)". El lector, en particular si es matemático o ingeniero, necesita que el marco físico defina qué es la Serie de Fourier de un sonido, qué es un fundamental y qué es un sobretono, y que la amplitud decrece exponencialmente conformando el timbre del instrumento.
