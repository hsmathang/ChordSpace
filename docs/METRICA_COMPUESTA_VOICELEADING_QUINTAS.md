# Métrica Compuesta: Voice Leading + Círculo de Quintas + Rugosidad

**Archivo de referencia**: `tools/proposals_pipeline/metrics.py` · Identificador: `voiceleading_quintas`  
**Convenciones**: Se sigue el glosario de notación estándar del repositorio (`GLOSARIO_NOTACION_MATEMATICA.md`).

---

## 1. Motivación y Objetivo

Las métricas basadas exclusivamente en rugosidad (JSD, Hellinger, coseno sobre $\Phi_{\text{raw}}$) capturan la similitud **sensorial** entre acordes pero ignoran dos dimensiones perceptual y musicalmente relevantes:

1. **Conducción de voces** (*voice leading*): la eficiencia del movimiento nota-a-nota entre dos acordes, criterio central en la composición y la teoría armónica funcional (Tymoczko, 2011; Callender et al., 2008).
2. **Cercanía tonal** en el ciclo de quintas (*circle of fifths*): la proximidad de las clases de altura de dos acordes sobre el anillo $\mathbb{Z}_{12}$ reordenado por quintas justas, indicador clásico de afinidad tonal (Harte & Sandler, 2006; Bernardes et al., 2016).

La métrica `voiceleading_quintas` integra estas tres perspectivas en una **combinación convexa ponderada** de disimilitudes atómicas, produciendo una distancia compuesta que balancea:

- movimiento eficiente de voces,
- afinidad tonal, y
- similitud en el perfil de rugosidad.

---

## 2. Prerrequisitos y Notación

### 2.1 Objetos de entrada

| Símbolo | Tipo | Definición |
|---|---|---|
| $\mathbf{n}^{(A)} = (n_1^{(A)}, \dots, n_{m_A}^{(A)})$ | $\in \mathcal{A}$ | Acorde $A$, tupla estrictamente creciente de notas MIDI. |
| $\mathbf{n}^{(B)} = (n_1^{(B)}, \dots, n_{m_B}^{(B)})$ | $\in \mathcal{A}$ | Acorde $B$. |
| $\Phi_{\text{raw}}(A) \in \mathbb{R}_{\geq 0}^{12}$ | vector | Histograma de rugosidad por clase de intervalo (Definición 3.10 de la metodología). |
| $\mathbf{p}_A = \Phi_{\text{raw}}(A) / \|\Phi_{\text{raw}}(A)\|_1$ | $\in \Delta^{11}$ | Distribución de probabilidad sobre clases de intervalo (proyección al simplex). |

### 2.2 Constantes y parámetros

| Símbolo | Valor por defecto | Descripción |
|---|---|---|
| $w_{\mathrm{VL}}$ | $0.55$ | Peso de la componente de *voice leading*. |
| $w_{\mathrm{Q5}}$ | $0.25$ | Peso de la componente del círculo de quintas. |
| $w_{\mathrm{JS}}$ | $0.20$ | Peso de la componente de rugosidad (JSD). |
| $\gamma$ | $6.5$ | Penalización por nota sin asignar (*gap penalty*). |
| $\epsilon$ | $10^{-12}$ | Constante de estabilidad numérica. |

Los pesos satisfacen $w_{\mathrm{VL}} + w_{\mathrm{Q5}} + w_{\mathrm{JS}} = 1$ y son configurables por el usuario; la implementación los renormaliza automáticamente:

$$
\hat{w}_k = \frac{w_k}{\sum_{j} w_j}, \quad k \in \{\mathrm{VL}, \mathrm{Q5}, \mathrm{JS}\}.
$$

---

## 3. Componente 1: Distancia de Conducción de Voces $d_{\mathrm{VL}}$

### 3.1 Costo de paso entre dos voces

**Definición (Costo de paso).** Dados $a, b \in \mathcal{N}$, el costo de mover una voz de la nota $a$ a la nota $b$ se define como:

$$
\mathrm{step}(a, b) = d_{12}^{\pm}(a, b) + 0.35 \cdot \frac{\min\!\big(|a - b|,\ 24\big)}{24},
$$

donde $d_{12}^{\pm}$ es la **distancia circular con signo plegado**:

$$
d_{12}^{\pm}(a, b) = \left| \big((a - b + 6) \bmod 12\big) - 6 \right|.
$$

**Interpretación:**
- El primer término $d_{12}^{\pm}(a, b) \in [0, 6]$ mide la distancia mínima en clases de altura sobre $\mathbb{Z}_{12}$, capturando el movimiento cromático más corto.
- El segundo término $\frac{\min(|a - b|, 24)}{24} \in [0, 1]$ penaliza saltos de registro absoluto (en semitonos MIDI), saturando a 24 semitonos (dos octavas). El factor $0.35$ controla la importancia relativa de la penalización de registro frente al movimiento cromático.

> [!NOTE]
> $d_{12}^{\pm}$ difiere de la métrica circular clásica $d_{12}(x, y) = \min_{k \in \mathbb{Z}} |x - y + 12k|$ en que la congruencia se centra en $[-6, 6]$ mediante la operación $\bmod 12$ desplazada. Ambas producen el mismo valor: el mínimo movimiento circular en $\mathbb{Z}_{12}$.

### 3.2 Asignación óptima entre voces (Hungarian)

**Definición (Distancia de voice leading).** Sean $A, B \in \mathcal{A}$ con $m_A = |\mathbf{n}^{(A)}|$ y $m_B = |\mathbf{n}^{(B)}|$ notas respectivamente. Se construye una **matriz de costos** $C \in \mathbb{R}^{M \times M}$ con $M = \max(m_A, m_B)$, definida por:

$$
C_{ij} =
\begin{cases}
\mathrm{step}\!\big(n_i^{(A)},\, n_j^{(B)}\big) & \text{si } i \leq m_A \text{ y } j \leq m_B, \\[4pt]
\gamma & \text{en otro caso (nota sin emparejar).}
\end{cases}
$$

Se resuelve el **problema de asignación lineal** (algoritmo húngaro):

$$
\sigma^* = \arg\min_{\sigma \in S_M} \sum_{i=1}^{M} C_{i,\sigma(i)},
$$

donde $S_M$ es el grupo simétrico de permutaciones de $\{1, \dots, M\}$.

La **distancia de voice leading normalizada** es:

$$
d_{\mathrm{VL}}(A, B) = \operatorname{clip}\!\left(\frac{\displaystyle\sum_{i=1}^{M} C_{i, \sigma^*(i)}}{M \cdot \gamma},\ 0,\ 1\right) \in [0, 1].
$$

**Propiedades:**
- $d_{\mathrm{VL}}(A, A) = 0$ (reflexividad).
- $d_{\mathrm{VL}}(A, B) = d_{\mathrm{VL}}(B, A)$ (simetría, por la simetría de $\mathrm{step}$ bajo valor absoluto y la naturaleza del matching Hungarian).
- $d_{\mathrm{VL}} \in [0, 1]$ por construcción (normalización y clip).
- Para acordes de **igual cardinalidad** ($m_A = m_B$), no se incurre en penalización $\gamma$ y la asignación reduce al matching óptimo clásico.
- Para acordes de **distinta cardinalidad**, las voces sobrantes del acorde más largo se penalizan con $\gamma$, modelando el costo de agregar o eliminar una voz.

> [!IMPORTANT]
> La implementación usa `scipy.optimize.linear_sum_assignment` (algoritmo húngaro, $O(M^3)$). Para triadas y cuatriadas ($M \leq 4$), el costo es despreciable; para poblaciones de $N$ acordes, la distancia se calcula para los $\binom{N}{2}$ pares.

---

## 4. Componente 2: Distancia en el Círculo de Quintas $d_{\mathrm{Q5}}$

### 4.1 Reordenamiento por quintas justas

**Definición (Anillo de quintas).** Se define la permutación $\tau: \mathbb{Z}_{12} \to \{0, 1, \dots, 11\}$ que reordena las clases de altura por el ciclo de quintas justas:

$$
\tau = (0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5),
$$

es decir, $\tau(0) = 0$, $\tau(1) = 7$, $\tau(2) = 2$, $\tau(3) = 9$, etc. La función inversa $\tau^{-1}: \mathbb{Z}_{12} \to \{0, \dots, 11\}$ asigna a cada pitch class su **posición en el anillo de quintas**.

**Justificación musical:** El ciclo de quintas es el generador del grupo cíclico $(\mathbb{Z}_{12}, +)$ por el intervalo de quinta justa (7 semitonos). La distancia entre acordes en este reordenamiento captura la proximidad tonal: acordes con pitch classes cercanas en el ciclo de quintas comparten más notas diatónicas y están más relacionados armónicamente.

### 4.2 Perfil de quintas suavizado

**Definición (Perfil de quintas).** Para un acorde $A$ con notas $\mathbf{n}^{(A)}$, se construye el **vector de activación en el anillo de quintas** $\mathbf{q}_A \in \mathbb{R}_{\geq 0}^{12}$ en tres pasos:

**Paso 1 — Acumulación.** Se calcula el vector de conteo bruto:

$$
\mathbf{q}_A^{(\mathrm{raw})}[k] = \# \big\{ i : \tau^{-1}\!\big(\pi(n_i^{(A)})\big) = k \big\}, \quad k = 0, \dots, 11,
$$

donde $\pi(n) = n \bmod 12$ es la proyección canónica a clases de altura.

**Paso 2 — Suavizado.** Se aplica un filtro de suavizado circular (kernel trinomial):

$$
\tilde{\mathbf{q}}_A[k] = \tfrac{1}{2}\, \mathbf{q}_A^{(\mathrm{raw})}[k] + \tfrac{1}{4}\, \mathbf{q}_A^{(\mathrm{raw})}[(k-1) \bmod 12] + \tfrac{1}{4}\, \mathbf{q}_A^{(\mathrm{raw})}[(k+1) \bmod 12].
$$

**Paso 3 — Normalización.** Se proyecta al simplex:

$$
\mathbf{q}_A = \frac{\tilde{\mathbf{q}}_A}{\|\tilde{\mathbf{q}}_A\|_1}.
$$

Si $\|\tilde{\mathbf{q}}_A\|_1 \leq \epsilon$, se asigna la distribución uniforme $\mathbf{q}_A = (\tfrac{1}{12}, \dots, \tfrac{1}{12})$.

> [!TIP]
> El suavizado trinomial con pesos $(\tfrac{1}{4}, \tfrac{1}{2}, \tfrac{1}{4})$ modela el hecho de que notas contiguas en el ciclo de quintas (e.g., Do y Sol, o Do y Fa) comparten contenido diatónico. Es análogo al suavizado gaussiano $\sigma = 0.75$ usado en `simplex_smooth` (§3.3.2 de la metodología), pero aplicado en el dominio del anillo de quintas en lugar del dominio cromático.

### 4.3 Distancia de Hellinger sobre perfiles de quintas

**Definición (Distancia de quintas).** La disimilitud entre dos acordes en el espacio del círculo de quintas se calcula como la **distancia de Hellinger** entre sus perfiles suavizados:

$$
d_{\mathrm{Q5}}(A, B) = \frac{1}{\sqrt{2}} \left\| \sqrt{\mathbf{q}_A} - \sqrt{\mathbf{q}_B} \right\|_2 \in [0, 1].
$$

**Propiedades:**
- $d_{\mathrm{Q5}}$ es una **métrica** en $\Delta^{11}$ (satisface los cuatro axiomas: no negatividad, identidad de indiscernibles, simetría, desigualdad triangular).
- Está acotada en $[0, 1]$.
- Es robusta a componentes nulas (a diferencia de JSD, no involucra logaritmos).
- Dos acordes que comparten exactamente las mismas clases de altura tendrán $d_{\mathrm{Q5}} = 0$ (tras suavizado y normalización).

---

## 5. Componente 3: Distancia de Rugosidad $d_{\mathrm{JS}}$

**Definición (Distancia JSD de rugosidad).** Se calcula la **divergencia de Jensen-Shannon** en base 2 entre las distribuciones de rugosidad normalizada:

$$
d_{\mathrm{JS}}(A, B) = \text{JSD}_2(\mathbf{p}_A, \mathbf{p}_B)^{1/2},
$$

donde $\text{JSD}_2$ denota la divergencia Jensen-Shannon calculada con $\log_2$:

$$
\text{JSD}_2(\mathbf{p}, \mathbf{q}) = \frac{1}{2}\, D_{\mathrm{KL}}(\mathbf{p} \,\|\, \mathbf{m}) + \frac{1}{2}\, D_{\mathrm{KL}}(\mathbf{q} \,\|\, \mathbf{m}), \quad \mathbf{m} = \frac{\mathbf{p} + \mathbf{q}}{2},
$$

con $D_{\mathrm{KL}}(\mathbf{p} \,\|\, \mathbf{q}) = \sum_{k=0}^{11} p_k \log_2 \frac{p_k}{q_k}$.

**Propiedades:**
- $\sqrt{\text{JSD}_2}$ es una **métrica** en $\Delta^{11}$ (Endres & Schindelin, 2003).
- Está acotada en $[0, 1]$ cuando se usa $\log_2$.
- Compara los **perfiles** de distribución de rugosidad: dos acordes con los mismos intervalos internos pero distinta magnitud global serán considerados idénticos (la normalización al simplex elimina información de magnitud).

> [!NOTE]
> En la implementación, `scipy.spatial.distance.jensenshannon(u, v, base=2.0)` retorna directamente $\sqrt{\text{JSD}_2}$, es decir, el valor ya es métrico sin necesidad de tomar raíz cuadrada adicional.

---

## 6. Composición Final: Métrica Compuesta

### 6.1 Definición

**Definición (Métrica compuesta `voiceleading_quintas`).** Para dos acordes $A, B \in \mathcal{A}$, la distancia compuesta se define como la combinación convexa:

$$
\boxed{
d_{\mathbf{w}}(A, B) = \hat{w}_{\mathrm{VL}}\, d_{\mathrm{VL}}(A, B) + \hat{w}_{\mathrm{Q5}}\, d_{\mathrm{Q5}}(A, B) + \hat{w}_{\mathrm{JS}}\, d_{\mathrm{JS}}(A, B),
}
$$

donde $\hat{w}_k = w_k / (w_{\mathrm{VL}} + w_{\mathrm{Q5}} + w_{\mathrm{JS}})$ son los pesos normalizados.

### 6.2 Propiedades formales

**Proposición (No negatividad y simetría).** $d_{\mathbf{w}}(A, B) \geq 0$ y $d_{\mathbf{w}}(A, B) = d_{\mathbf{w}}(B, A)$ para todo $A, B \in \mathcal{A}$.

*Demostración.* Cada componente $d_{\mathrm{VL}}, d_{\mathrm{Q5}}, d_{\mathrm{JS}}$ es no negativa y simétrica; la combinación convexa con pesos no negativos preserva ambas propiedades. $\square$

**Proposición (Identidad de indiscernibles).** $d_{\mathbf{w}}(A, B) = 0 \iff d_{\mathrm{VL}}(A, B) = d_{\mathrm{Q5}}(A, B) = d_{\mathrm{JS}}(A, B) = 0$, lo cual equivale a que $A$ y $B$ son el mismo acorde (mismas notas MIDI).

*Demostración.* ($\Leftarrow$) Trivial. ($\Rightarrow$) Si $d_{\mathbf{w}} = 0$ y todos los $\hat{w}_k > 0$, entonces cada sumando es cero. $d_{\mathrm{VL}} = 0$ implica que existe un matching perfecto de costo cero, y con $d_{\mathrm{Q5}} = d_{\mathrm{JS}} = 0$, las distribuciones en quintas y en rugosidad coinciden. $\square$

**Observación (Desigualdad triangular).** La combinación convexa de métricas es métrica. Dado que $d_{\mathrm{Q5}}$ y $d_{\mathrm{JS}}$ son métricas y $d_{\mathrm{VL}}$ satisface la desigualdad triangular (como matching óptimo de costos métricos normalizado), la combinación convexa $d_{\mathbf{w}}$ también satisface la desigualdad triangular.

### 6.3 Acotación

$$
d_{\mathbf{w}}(A, B) \in [0, 1],
$$

ya que cada componente $d_k \in [0, 1]$ y los pesos suman 1.

### 6.4 Valores por defecto

Con los pesos por defecto $(\hat{w}_{\mathrm{VL}}, \hat{w}_{\mathrm{Q5}}, \hat{w}_{\mathrm{JS}}) = (0.55, 0.25, 0.20)$:

- La conducción de voces domina ($55\%$), priorizando acordes alcanzables con movimiento mínimo.
- El ciclo de quintas aporta cohesión tonal ($25\%$), favoreciendo acordes dentro de la misma región diatónica.
- La rugosidad preserva la similitud tímbrica ($20\%$), asegurando que el "color" sonoro se mantenga.

---

## 7. Complejidad Computacional

Para una población de $N$ acordes con cardinalidad máxima $m_{\max}$:

| Componente | Complejidad por par | Total ($\binom{N}{2}$ pares) |
|---|---|---|
| $d_{\mathrm{VL}}$ | $O(m_{\max}^3)$ (Hungarian) | $O(N^2 \cdot m_{\max}^3)$ |
| $d_{\mathrm{Q5}}$ | $O(12)$ vectorial | $O(N^2)$ — vía `pdist` vectorizado |
| $d_{\mathrm{JS}}$ | $O(12)$ vectorial | $O(N^2)$ — vía `pdist` vectorizado |

El cuello de botella es $d_{\mathrm{VL}}$ por su componente cúbica en $m_{\max}$, aunque para triadas ($m_{\max} = 3$) y cuatriadas ($m_{\max} = 4$) el costo es despreciable.

---

## 8. Ejemplo Numérico

Sean $A = (60, 64, 67)$ (Do mayor, C4–E4–G4) y $B = (60, 63, 67)$ (Do menor, C4–Eb4–G4).

### Componente VL

$m_A = m_B = 3$, $M = 3$, no hay gaps:

$$
C = \begin{pmatrix}
\mathrm{step}(60, 60) & \mathrm{step}(60, 63) & \mathrm{step}(60, 67) \\
\mathrm{step}(64, 60) & \mathrm{step}(64, 63) & \mathrm{step}(64, 67) \\
\mathrm{step}(67, 60) & \mathrm{step}(67, 63) & \mathrm{step}(67, 67)
\end{pmatrix}.
$$

El matching óptimo es la identidad $\sigma^* = \text{id}$: voz 1 → voz 1 (costo 0), voz 2 → voz 2 (la E4 se mueve 1 semitono a Eb4, costo $\approx 1 + 0.35 \cdot \frac{1}{24} \approx 1.015$), voz 3 → voz 3 (costo 0).

$$
d_{\mathrm{VL}} \approx \frac{1.015}{3 \cdot 6.5} \approx 0.052.
$$

### Componente Q5

Ambos acordes comparten C y G (posiciones 0 y 1 en el anillo de quintas). $A$ tiene E (posición 4), $B$ tiene Eb (posición 8). Tras suavizado y normalización, la diferencia es pequeña pero no nula:

$$
d_{\mathrm{Q5}} \approx 0.18.
$$

### Componente JS

Las distribuciones de rugosidad por clase de intervalo son similares (difieren solo en el bin de la 3ª m vs 3ª M):

$$
d_{\mathrm{JS}} \approx 0.12.
$$

### Distancia compuesta

$$
d_{\mathbf{w}} = 0.55 \times 0.052 + 0.25 \times 0.18 + 0.20 \times 0.12 \approx 0.029 + 0.045 + 0.024 = 0.098.
$$

Este valor bajo ($< 0.1$) refleja correctamente que Do mayor y Do menor son acordes cercanos en las tres dimensiones consideradas.

---

## 9. Relación con otras Métricas del Repositorio

| Métrica | Dimensiones | Dependencia de notas absolutas |
|---|---|---|
| `cosine`, `euclidean` | Solo $\Phi$ (rugosidad) | No (opera sobre vectores procesados) |
| `js`, `hellinger` | Solo $\mathbf{p}$ (simplex) | No |
| `structural_roughness` | Jaccard + perfil + densidad | Parcial |
| **`voiceleading_quintas`** | **VL + quintas + JSD** | **Sí** (requiere `notes_abs`) |

---

## 10. Referencia de Implementación

| Función | Archivo | Responsabilidad |
|---|---|---|
| `_voice_step_cost(a, b)` | `metrics.py` | Costo de paso entre voces individuales. |
| `_voice_leading_distance(notes_a, notes_b, gap)` | `metrics.py` | Matching Hungarian normalizado. |
| `_quintas_profile(notes)` | `metrics.py` | Perfil de quintas suavizado. |
| `_voiceleading_quintas_distance(simplex, entries)` | `metrics.py` | Ensambla las tres componentes y retorna el vector condensado. |
| `_resolve_voiceleading_quintas_params(params)` | `metrics.py` | Resuelve y normaliza los pesos configurables. |

---

*Documento generado como referencia matemática interna para el repositorio ChordSpace.*
