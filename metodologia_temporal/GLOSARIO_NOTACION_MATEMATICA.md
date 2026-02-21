# Glosario y Estándares de Notación Matemática (ChordSpace)

**Propósito:** Este documento define la notación estándar para el capítulo de Metodología y subsiguientes, garantizando consistencia, rigor matemático y claridad expositiva. Se alinea con las convenciones de *Scientific Writing* y *Scholar Evaluation*.

---

## 1. Normas Generales de Estilo

| Elemento | Convención Tipográfica | Ejemplo LaTeX | Ejemplo Renderizado |
|---|---|---|---|
| **Conjuntos / Espacios** | Mayúscula Caligráfica o `mathbb` | `\mathcal{N}, \mathbb{Z}` | $\mathcal{N}, \mathbb{Z}$ |
| **Vectores** | Minúscula Negrita | `\mathbf{n}, \mathbf{x}` | $\mathbf{n}, \mathbf{x}$ |
| **Matrices** | Mayúscula Normal | `D, Y` | $D, Y$ |
| **Escalares / Índices** | Minúscula Cursiva | `n, k, m` | $n, k, m$ |
| **Funciones** | Minúscula Cursiva (o `\text{}` si es nombre) | `f, \pi, \text{JSD}` | $f, \pi, \text{JSD}$ |
| **Normas / Distancias** | Doble barra vertical | `\|\cdot\|` | $\|\cdot\|$ |

---

## 2. Definiciones de Símbolos

### 2.0 Lógica, Conjuntos y Aritmética

| Símbolo | Código LaTeX | Definición / Significado | Contexto |
|---|---|---|---|
| $\iff$ | `\iff` | **Si y solo si.** Equivalencia lógica. | Definiciones formales. |
| $\equiv \pmod{n}$ | `\equiv \pmod{n}` | **Congruencia modular.** Equivalencia en $\mathbb{Z}_n$. | Clases de altura. |
| $\binom{n}{k}$ | `\binom{n}{k}` | **Coeficiente Binomial.** Combinaciones de $k$ en $n$. | Cardinalidad de espacios ($\binom{128}{3}$). |
| $\setminus$ | `\setminus` | **Diferencia de Conjuntos.** | Exclusiones (e.g., $\mathbb{N} \setminus \{0\}$). |
| $\approx$ | `\approx` | **Aproximadamente igual.** | Valores numéricos o estimaciones. |
| $\propto$ | `\propto` | **Proporcional a.** | Relaciones de escala (e.g., $\log_2 \propto \ln$). |

### 2.1 Espacios y Conjuntos Fundamentales

| Símbolo | Código LaTeX | Definición / Significado | Contexto / Rango |
|---|---|---|---|
| $\mathcal{N}$ | `\mathcal{N}` | **Espacio de Notas MIDI.** Conjunto de alturas discretas disponibles. | $\{0, 1, \dots, 127\} \subset \mathbb{N}_0$ |
| $\mathbb{N}_0$ | `\mathbb{N}_0` | **Naturales con cero.** Conjunto de enteros no negativos. | $\{0, 1, 2, \dots\}$ |
| $\mathbb{Z}_{12}$ | `\mathbb{Z}_{12}` | **Grupo de Clases de Altura.** Grupo cíclico de enteros módulo 12. | $\{0, 1, \dots, 11\}$ |
| $\mathcal{A}$ | `\mathcal{A}` | **Espacio Total de Acordes.** Unión de todos los $m$-acordes válidos. | $\bigcup_{m} \mathcal{A}_m$ |
| $\mathcal{A}_m$ | `\mathcal{A}_m` | **Espacio de $m$-Acordes.** Subconjunto de tuplas estrictamente crecientes. | $\subset \mathcal{N}^m, n_1 < \dots < n_m$ |
| $\Delta^{d}$ | `\Delta^{d}` | **Simplex Estándar.** Espacio de distribuciones de probabilidad sobre $d+1$ bins. | $\{x \in \mathbb{R}_{\geq 0}^{d+1} : \sum x_i = 1\}$ |
| $\mathbb{R}^2$ | `\mathbb{R}^2` | **Espacio del Embedding.** Plano cartesiano de visualización. | Salida de MDS/UMAP |
| $\mathbb{T}^n$ | `\mathbb{T}^n` | **n-Toro.** Espacio topológico producto de $n$ círculos (modelo teórico de clases de altura). | Contexto OPTIC (Tymoczko) |
| $S_m$ | `S_m` | **Grupo Simétrico.** Grupo de todas las permutaciones de $m$ elementos. | Contexto de Orbifolds |
| $\mathbb{R}^m/S_m$ | `\mathbb{R}^m/S_m` | **Orbifold de alturas.** Espacio cociente bajo permutación (sin repetición y sin equivalencia de octava). | Modelo geométrico base |

### 2.2 Elementos y Vectores Representativos

| Símbolo | Código LaTeX | Definición / Significado | Contexto / Rango |
|---|---|---|---|
| $n$ | `n` | **Nota MIDI.** Un elemento individual de $\mathcal{N}$. | $n \in [0, 127]$ |
| $\mathbf{n}$ | `\mathbf{n}` | **Acorde (Vector).** Tupla ordenada que representa un acorde. | $\mathbf{n} = (n_1, \dots, n_m)$ |
| $c$ | `c` | **Acorde (Entidad Abstracta).** Usado cuando no se enfatiza la estructura vectorial. | $c \in \mathcal{A}$ |
| $\mathbf{ic}$ | `\mathbf{ic}` | **Vector de Conteo de Intervalos.** Histograma de intervalos (cuentas enteras). | $\mathbf{ic} \in \mathbb{N}_0^{12}$ |
| $\Phi_{\text{raw}}$ | `\Phi_{\text{raw}}` | **Vector de Características (Crudo).** Perfil de rugosidad acumulada por clase de intervalo. | $\mathbb{R}_{\geq 0}^{12}$ |
| $\Phi$ | `\Phi` | **Feature Vector (Normalizado).** Vector procesado listo para reducción dimensional (generalmente en el simplex). | $\in \Delta^{11}$ (típicamente) |
| $Y$ | `Y` | **Configuración del Embedding.** Matriz de coordenadas en el espacio reducido. | $Y \in \mathbb{R}^{N \times 2}$ |
| $\hat{D}_{ij}$ | `\hat{D}_{ij}` | **Distancia en el Embedding.** Distancia euclidiana entre puntos reducidos. | $\|y_i - y_j\|_2$ |

### 2.3 Funciones y Operadores Matemáticos

| Símbolo | Código LaTeX | Definición / Significado | Observaciones |
|---|---|---|---|
| $f(n)$ | `f(n)` | **Frecuencia Fundamental.** Convierte MIDI a Hz (A4=440). | $f(n) = 440 \cdot 2^{(n-69)/12}$ |
| $\pi(n)$ | `\pi(n)` | **Proyección Canónica.** Mapea nota a clase de altura. | $\pi: \mathcal{N} \to \mathbb{Z}_{12}$ |
| $\Delta(\mathbf{n})$ | `\Delta(\mathbf{n})` | **Vector de Intervalos Adyacentes.** Diferencias entre notas consecutivas. | $\Delta \in \mathbb{N}^{m-1}$ (Contexto Acordes) |
| $\text{span}(\mathbf{n})$ | `\text{span}(\mathbf{n})` | **Rango del Acorde.** Diferencia entre nota máxima y mínima. | $n_m - n_1$ |
| $T_n, I$ | `T_n, I` | **Transposición / Inversión.** Operadores de PC-sets (Forte). | Mencionados para distinción. |
| $\Psi$ | `\Psi` | **Función de Reducción Dimensional.** Mapeo del espacio de características al plano. | $\Psi: \mathbb{R}^{12} \to \mathbb{R}^2$ |
| $R(f_a, f_b)$ | `R(f_a, f_b)` | **Rugosidad Binaria.** Disonancia sensorial entre dos frecuencias complejas. | Modelo Sethares (1993) |
| $R_{\text{total}}$ | `R_{\text{total}}` | **Rugosidad Total.** Suma de rugosidades de todos los pares en un acorde. | Escalar $\in \mathbb{R}_{\geq 0}$ |
| $\hat{r}_{ij}$ | `\hat{r}_{ij}` | **Rango.** Posición del objeto $j$ en la lista ordenada de vecinos de $i$. | Para métricas $T(k), C(k)$ (Eq. 3.18) |

### 2.4 Métricas de Distancia y Divergencia

| Símbolo | Código LaTeX | Definición / Significado | Observaciones |
|---|---|---|---|
| $d(\mathbf{x}, \mathbf{y})$ | `d(\mathbf{x}, \mathbf{y})` | **Función de Distancia Genérica.** | Salida $\in [0, \infty)$ |
| $\|\mathbf{v}\|_p$ | `\|\mathbf{v}\|_p` | **Norma $L_p$.** Magnitud del vector. | $p=1$ (Taxicab), $p=2$ (Euclidiana) |
| $d_{\cos}$ | `d_{\cos}` | **Distancia Coseno.** Disimilitud angular. | $1 - \cos(\theta)$ |
| $D_{\text{KL}}$ | `D_{\text{KL}}` | **Divergencia Kullback-Leibler.** Entropía relativa. | Asimétrica. |
| $\text{JSD}$ | `\text{JSD}` | **Jensen-Shannon Divergence** | Disimilitud teórica de la información. Simétrica y acotada. |
| $d_H$ | `d_H` | **Hellinger Distance** | Distancia métrica entre distribuciones. Robusta y acotada en $[0, 1]$. |

### 2.5 Parámetros del Modelo

| Símbolo | Código LaTeX | Descripción | Valor por Defecto |
|---|---|---|---|
| $H$ | `H` | **Número de Armónicos.** Cantidad de parciales por nota en el modelo espectral. | $6$ |
| $\delta$ | `\delta` | **Tasa de Decaimiento.** Factor de atenuación de amplitud de parciales ($a_k = \delta^{k-1}$). | $0.88$ |
| $A_i, C_i, S_i$ | `A_i, C_i, S_i` | **Constantes de Sethares.** Parámetros de la curva de disonancia sensorial. | $A_1=-3.51, C_1=5 \dots$ |
| $\sigma$ | `\sigma` | **Sigma (Kernel Gaussiano).** Desviación estándar para suavizado perceptual de histogramas. | $0.75$ semitonos |
| $\epsilon$ | `\epsilon` | **Epsilon.** Valor pequeño para estabilidad numérica (padding de ceros en logaritmos). | $10^{-12}$ |
| $N$ | `N` | **Tamaño de la Población.** Número total de acordes en el experimento. | Variable (e.g., $10^4$) |
| $m$ | `m` | **Cardinalidad.** Número de notas en un acorde específico. | Variable |
| $k$ | `k` | **Vecinos Cercanos** (en contexto métrico) o **Índice de Intervalo** (en contexto $\mathbf{ic}$). | $3$ (Evaluación) |

### 2.6 Métricas de Evaluación y Calidad

| Símbolo | Código LaTeX | Nombre Completo | Interpretación |
|---|---|---|---|
| $\text{Stress}$ | `\text{Stress}` | **Kruskal's Stress-1** | Error global de MDS (menor es mejor). Ideal $< 0.1$. |
| $T(k)$ | `T(k)` | **Trustworthiness** | Fiabilidad del vecindario visualizado (penaliza intrusiones). Rango $[0, 1]$. |
| $C(k)$ | `C(k)` | **Continuity** | Preservación del manifold original (penaliza extrusiones/téars). Rango $[0, 1]$. |
| $Q_{local}$ | `Q_{local}` | **Calidad Local Agregada.** | Promedio de $T(k)$ y $C(k)$ sobre un rango de $k$. |
| $\text{Sil}$ | `\text{Sil}` | **Silhouette Score** | Cohesión de clusters (basada en cardinalidad u otra etiqueta). |
| $\rho_S$ | `\rho_S` | **Spearman's Rank Correlation** | Correlación de rangos entre distancias originales $D$ y embebidas $\hat{D}$. |
| $R^2$ | `R^2` | **Coeficiente de Determinación.** | Ajuste lineal del diagrama de Shepard. |

---

## 3. Lista de Verificación de Uso (Scholar Evaluation)

Al redactar, verificar:
1.  **¿Está definido?** No usar un símbolo sin haberlo introducido o referenciado a este glosario.
2.  **¿Es consistente?** No mezclar $v$ y $\mathbf{v}$ para el mismo objeto vectorial.
3.  **¿Es necesario?** Evitar proliferación de símbolos si el lenguaje natural es más claro.
4.  **¿Es estándar?** Usar $\mathbb{Z}_{12}$ en lugar de $C_{12}$ (contexto algebraico). Usar $\log_2$ explícito para bits.
5.  **¿Es inequívoco?** Diferenciar $k$ (índice de intervalo) de $k$ (vecinos cercanos) por contexto explícito. Usar $\Delta^d$ para simplex y $\Delta(\mathbf{n})$ para intervalos.

---
*Generado automáticamente a partir del análisis riguroso de `ESTRUCTURA_MATEMATICA_DETALLADA.md`.*
