# Propuesta de Diseño Experimental y Validación: ChordSpace

> **Estatus:** Versión Maximalista (Consolidada).
> **Objetivo:** Validar la robustez topológica y la relevancia musical del espacio $\mathbb{R}^2$ de acordes.

---

## ⚠️ Marco Teórico Crítico: Disonancia Sensorial vs. Sintaxis Tonal

> **Limitación Fundamental:** El modelo de Sethares mide **Disonancia Sensorial** (fenómeno psicoacústico vertical/atemporal). La armonía tonal (especialmente en el Barroco) se rige por la **Sintaxis** (fenómeno gramatical horizontal/temporal), donde la "tensión" depende de la conducción de voces y la expectativa de resolución.
>
> *Implicación Experimental:* Los experimentos aquí propuestos buscan medir la **intersección** entre estos dos dominios. Si el modelo agrupa acordes funcionalmente equivalentes (Exp D), sugiere que la sintaxis tonal *emerge* o se *apoya* en propiedades sensoriales básicas. Si no, confirma su ortogonalidad.

---

## 1. Guía de Interpretación de Resultados Topológicos

Para evaluar la calidad del "mapa musical", analizaremos los compromisos entre **Confianza ($T(k)$)** y **Continuidad ($C(k)$)** *[Venna & Kaski, 2005]*.

| Escenario | Resultado Métrico | Interpretación Musical | Diagnóstico Técnico |
| :--- | :--- | :--- | :--- |
| **A. Rotura** | $T(k) \approx 0.95$ (Alto)<br>$C(k) < 0.80$ (Bajo) | **"Islas Musicales"**. Acordes similares aparecen separados. Se pierden transiciones suaves (ej. progresiones por círculo de quintas rotas). | Embedding confiable pero discontinuo. |
| **B. Alucinación** | $T(k) < 0.80$ (Bajo)<br>$C(k) \approx 0.95$ (Alto) | **"Falsos Vecinos"**. Acordes disonantes aparecen cerca de consonantes. El mapa sugiere relaciones armónicas inexistentes. | Embedding continuo pero poco confiable (intrusiones). |
| **C. Ideal** | $T(k) > 0.90$<br>$C(k) > 0.90$ | **"Isometría Local"**. El mapa captura fielmente tanto la identidad del acorde como sus transiciones posibles. | Preservación topológica excelente. |

---

## 2. Serie Experimental A: Validación Musical Incremental (Exp 1-6)

Esta serie aisla variables musicales para verificar si el modelo aprende propiedades acústicas fundamentales o solo memoriza frecuencias.

### Exp 1: Línea Base (Tríadas Diatónicas)
*   **Datos:** 21 Tríadas (Do Mayor) + Inversiones.
*   **Razón Musical:** Verificar la **Clasificación de Cualidad**. El sistema debe distinguir *Mayor* vs *Menor* vs *Disminuido* sin información de notas explícitas.

### Exp 2: Segregación Estilística (Acordes Extremos)
*   **Datos:** Exp 1 + "Acordes Extremos" (Clusters cromáticos, Polychords).
*   **Hipótesis:** Los acordes de alta complejidad/disonancia deben ocupar una región (manifold) distinta a las tríadas.
*   **Razón Musical:** Evitar el *Crowding Problem*. Si el modelo mezcla tríadas con clusters de ruido, no captura la "complejidad armónica" como dimensión latente.

### Exp 3: Invariancia de Octava (Notas Repetidas)
*   **Datos:** Tríadas con duplicaciones de octava ($C_3-E_3-G_3$ vs $C_4-E_4-G_4$).
*   **Hipótesis:** La distancia entre inversiones o duplicaciones debe ser casi nula ($d \to 0$).
*   **Razón Musical:** Validar que el modelo aprende **Clases de Altura** (Pitch Classes) y no solo frecuencias absolutas. La identidad del acorde debe ser invariante a la octava.

### Exp 4 & 5: Escalabilidad y Robustez Masiva
*   **Datos:** Muestra Aleatoria Grande ($N \approx 1000$) y Masiva ($N=100k$).
*   **Hipótesis:** La estructura global (círculo de quintas, relaciones tonales) no se diluye al aumentar la densidad de muestreo.
*   **Razón Musical:** Validar la utilidad del modelo para *Big Data* musical.

### Exp 6: Resistencia al Ruido
*   **Datos:** Acordes con *jitter* (pequeñas variaciones de frecuencia).
*   **Hipótesis:** Estabilidad topológica ($T(k)$ estable) ante perturbaciones menores.

---

## 3. Experimento B: Validación Topológica Rigurosa

**Objetivo:** Rechazar la hipótesis nula de que la estructura del mapa es aleatoria.

*   **Métricas:**
    *   **Trustworthiness ($T(k)$):** Penaliza falsos positivos. Benchmark: $>0.90$ para $k=5$.
    *   **Continuity ($C(k)$):** Penaliza falsos negativos (roturas).
*   **Hipótesis Nula ($H_0$):** $T(k)$ y $C(k)$ no difieren significativamente de una proyección aleatoria ($\approx 0.5$).
*   **Referencia:** *Venna, J., & Kaski, S. (2005). Local multidimensional scaling...*

---

## 4. Experimento C: Validación Perceptual (Ground Truth Humano)

**Objetivo:** Correlacionar la geometría del modelo con la percepción humana.

*   **Dataset 1:** **Bowling et al. (2018)**. *Vocal pitch interval similarity*.
*   **Dataset 2:** **Harrison & Pearce (2020)**. *Simultaneous consonance perception*.
*   **Métrica:** Correlación de Spearman ($\rho$) y Correlación Parcial (controlando por número de notas comunes).

---

## 5. Experimento D: Validación Funcional (Estilo Barroco / Bach)

**Objetivo:** Validar si la rugosidad predice la **Sustituibilidad Funcional** en el Barroco.

*   **Protocolo:**
    1.  **Corpus:** **JSB Chorales** (389 corales).
    2.  **Definición Funcional:** Dos acordes son sustitutos si comparten el mismo contexto armónico (vecinos $n-1, n+1$).
    3.  **Hipótesis:** Los acordes funcionalmente intercambiables (e.g., $IV$ y $ii^6$) deben ser vecinos cercanos en el espacio geométrico ($C(k)$ alto sobre clases funcionales).
*   **Referencia:** *Burgoyne et al. (2011). An expert ground-truth set...*

---

## 6. Experimento E: Consistencia Estilística (OOD)

**Objetivo:** Cuantificar la separación entre el "Manifold Barroco" y el "Ruido".

*   **Método:**
    *   Proyectar 3 clases: Bach (A), Ruido Uniforme (B), Extremos (C).
    *   **Test:** Divergencia de Kullback-Leibler o **Cross-Entropy Test**.
    *   **Métrica:** Silhouette Score para medir la separabilidad lineal/no lineal.
