# Propuesta de Estructura Final: Capítulo de Metodología

**Contexto:** Tesis de Maestría en Matemáticas Aplicadas.
**Objetivo:** Definir la estructura óptima para el capítulo de Metodología, integrando el rigor matemático, la implementación computacional (repo) y la validación experimental.

---

## Opción A: Enfoque Puramente Formal (Matemático)
*Centrado en definiciones, conjuntos y funciones. El software es solo un detalle de implementación.*

1.  **Fundamentación Teórica del Espacio de Acordes**
    *   Definición de nota, pitch class y acorde como conjunto ordenado.
    *   El espacio métrico $\mathcal{C}$: distancias y topología.
2.  **Modelo de Disonancia Sensorial (Formalización)**
    *   Modelo de Plomp-Levelt y parametrización de Sethares.
    *   Función de Rugosidad $R: \mathcal{C} \to \mathbb{R}$.
    *   Vectorización de la rugosidad sobre clases de intervalo.
3.  **Métricas de Similitud y Sustitución**
    *   Definición de métricas en el simplex de rugosidad (JSD, Coseno).
    *   Métrica compuesta para sustitución armónica (Definición formal $D_w$).
4.  **Reducción de Dimensionalidad**
    *   Formulación del problema de minimización de Stress (MDS).
    *   Fundamentos de UMAP (proyección de variedades).
5.  **Diseño Experimental Abstracto**
    *   Definición de los conjuntos de prueba $E_1 \dots E_6$.

*   **Evaluación ScholarEval:** Alta en "Formulación del Problema", baja en "Reproducibilidad" e "Ingeniería". Puede parecer desconectada del repositorio real.

---

## Opción B: Enfoque de Ingeniería de Software (Descriptivo)
*Centrado en la arquitectura, el pipeline y el código. Similar a `metodologia_version_repo.tex`.*

1.  **Arquitectura del Sistema ChordSpace**
    *   Módulos principales: Generador, filtros, procesador.
2.  **Pipeline de Generación de Datos**
    *   Generación combinatoria vs. Base de datos.
    *   Estructura de datos (JSON) y normalización.
3.  **Implementación del Modelo Psicoacústico**
    *   Algoritmo vectorizado de Sethares.
    *   Optimización y caché.
4.  **Subsistema de Visualización y Métricas**
    *   Librerías de reducción (sklearn, umap-learn).
    *   Generación de reportes HTML.
    *   Algoritmo k-NN para sustitución (implementación).
5.  **Verificación y Reproducibilidad**
    *   Tests unitarios.
    *   Instrucciones de ejecución (`run_lab`).

*   **Evaluación ScholarEval:** Alta en "Reproducibilidad", pero baja en "Rigor Científico" para una tesis de matemáticas. Parece un reporte técnico o manual de usuario.

---

## Opción C: Enfoque Híbrido Científico (Recomendado)
*Integra la definición formal con la operacionalización computacional y el diseño experimental. Es el estándar en Matemática Aplicada y Computacional.*

1.  **Introducción y Flujo Metodológico**
    *   Visión general: del objeto matemático al artefacto visual.
2.  **El Objeto Computacional: Definición y Generación**
    *   Formalización matemática del acorde ($n$-tupla MIDI).
    *   *Operacionalización:* Generación combinatoria y filtrado (referencia al código).
3.  **Modelado de Características: Rugosidad y Estructura**
    *   Modelo matemático de Sethares (fórmulas).
    *   *Implementación:* Vectorización y histogramas de intervalo.
4.  **Espacios Métricos y Sustitución Armónica**
    *   Definición de distancias (JSD, Coseno, Euclídea).
    *   **Algoritmo de Sustitución:** Definición de la métrica compuesta (Sensorial + Estructural) y perfiles de búsqueda.
5.  **Proyección y Análisis Exploratorio**
    *   Técnicas de reducción: MDS (preservación de distancia) y UMAP (topología).
    *   Métricas de calidad de embedding (Trustworthiness, Stress).
6.  **Diseño Experimental (Roadmap)**
    *   Descripción sistemática de los Experimentos 1-6 (la "ruta de validación").
    *   Justificación de cada experimento (control, estrés, escala masiva).

*   **Evaluación ScholarEval:** Equilibrada. Satisface el rigor matemático (definiciones), la reproducibilidad (código/operacionalización) y la validez científica (diseño experimental).

---

## Recomendación Final: Opción C+

La **Opción C** es la ganadora indiscutible. Para implementarla, se debe reestructurar el archivo `metodologia_version_repo.tex` para que se convierta en este híbrido.

### Estructura Detallada (Tabla de Contenidos Propuesta)

**Capítulo 3: Metodología**

**3.1 Definición y Operacionalización del Objeto de Estudio**
*   3.1.1 Formalización Matemática (Tuplas, PC-sets, Intervalos).
*   3.1.2 Generación Combinatoria del Espacio de Búsqueda (Algoritmo y Filtros).

**3.2 Modelado Psicoacústico de la Rugosidad**
*   3.2.1 Modelo de Interacción de Parciales (Plomp-Levelt/Sethares).
*   3.2.2 Vectorización por Clases de Intervalo (La "Firma" del acorde).

**3.3 Definición de Distancias y Modelo de Sustitución**
*   3.3.1 Medidas de Disimilitud Probabilística (JSD, Hellinger).
*   3.3.2 Modelo de Sustitución Armónica (Definición del algoritmo k-NN ponderado).
    *   *Aquí se integra la lógica matemática de `substitution_metrics.md`.*

**3.4 Reducción de Dimensionalidad y Calidad del Embedding**
*   3.4.1 Multidimensional Scaling (MDS): Enfoque métrico.
*   3.4.2 UMAP: Aproximación y Proyección de Variedades.
*   3.4.3 Métricas de Evaluación de la Proyección (Trustworthiness, Continuity).

**3.5 Diseño Experimental (Roadmap de Validación)**
*   3.5.1 Exp 1: Validación con Tríadas Diatónicas (N=21).
*   3.5.2 Exp 2: Estructuras Conocidas y Control (N=23).
*   3.5.3 Exp 3 y 4: Stress-test con Rugosidad Extrema y Repeticiones.
*   3.5.4 Exp 5 y 6: Escalamiento a Poblaciones Masivas (1k - 100k).

**3.6 Estrategia de Reproducibilidad**
*   3.6.1 Entorno Computacional y Semillas.
*   3.6.2 Pipeline de Ejecución (`run_lab`).
