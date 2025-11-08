# Dos formas distintas de invertir acordes
**(inverso musical vs. inverso estructural “a cero”)**

> Documento técnico breve que formaliza dos nociones complementarias de “inversión” para acordes representados como tuplas ordenadas de alturas o clases de altura. La primera opera sobre el **orden de las voces** (rotaciones); la segunda, sobre la **estructura intervalar** tras fijar el primer elemento en 0 (transposición normalizada).

---

## 0. Notación y marco
- Denote por $\mathbb{Z}$ el conjunto de enteros y por $\mathbb{Z}_{12}$ las clases de altura **módulo 12** (12‑TET).
- Un **acorde ordenado** (o *voicing*) de longitud $k$ es una $k$‑tupla $v=(v_1,\dots,v_k)$ con $v_i\in\mathbb{Z}$ (alturas absolutas en semitonos) o su reducción $[v]_{{12}}=(v_1\bmod 12,\dots,v_k\bmod 12)\in\mathbb{Z}_{12}^k$.
- La **transposición** por $t\in\mathbb{Z}$ actúa componente a componente: $T_t(v)=(v_1+t,\dots,v_k+t)$; en clases de altura se entiende $t$ módulo 12.
- La **rotación** (corrimiento cíclico) $\rho:\mathbb{Z}^k\to\mathbb{Z}^k$ está dada por
  $$\rho(v_1,\dots,v_k)=(v_2,\dots,v_k,v_1),\qquad \rho^j=\underbrace{\rho\circ\cdots\circ\rho}_{j\text{ veces}}.$$
- Para $x\in\mathbb{Z}$ se usa $\langle x\rangle_{12}$ para su representante en $\{0,1,\dots,11\}$.

> En este documento **no** usamos la “inversión por reflexión” de la teoría PC‑set (operador $I$) para evitar ambigüedades terminológicas: aquí *inversión* significa dos ideas distintas, ambas ligadas a *rotación* y *transposición normalizada*, respectivamente.

---

## 1. Inverso musical (rotacional)
Sea $v=(v_1,\dots,v_k)\in\mathbb{Z}^k$. Definimos el **conjunto de inversos musicales** de $v$ como la órbita de $v$ bajo la acción del grupo cíclico $C_k=\langle\rho\rangle$:
$$
\mathrm{Inv}_{\mathrm{mus}}(v)\;=\;\big\{\rho^j(v)\;\big|\;j=0,1,\dots,k-1\big\}.
$$

**Comentarios.**
1. Si todas las entradas de $v$ son distintas, entonces $|\mathrm{Inv}_{\mathrm{mus}}(v)|=k$. Si $v$ es periódico bajo rotación (p. ej., contiene repeticiones que hacen que $\rho^d(v)=v$ para algún $d\mid k$), el tamaño de la órbita se reduce a $k/d$.
2. Esta noción **preserva el contenido absoluto** (no altera alturas) y solo reordena voces. Es la formalización del gesto musical “llevar la voz inferior arriba” y sus iteraciones.

**Ejemplo 1.1.** Para $v=[2,7,9]$ (en semitonos absolutos):
$$
\mathrm{Inv}_{\mathrm{mus}}(v)=\big\{[2,7,9],\;[7,9,2],\;[9,2,7]\big\}.
$$

---

## 2. Inverso estructural “a cero” (transposición normalizada por la primera voz)
La **estructura** ignora la altura absoluta y retiene las distancias **desde una voz designada**. Formalizamos la “inversión a cero” como sigue.

Sea $v=(v_1,\dots,v_k)\in\mathbb{Z}^k$. Definimos el **normalizador a cero** $N:\mathbb{Z}^k\to\mathbb{Z}_{12}^k$ por
$$
N(v)\;=\;\big(\,\langle v_1-v_1\rangle_{12},\;\langle v_2-v_1\rangle_{12},\;\dots,\;\langle v_k-v_1\rangle_{12}\,\big)\;=\;(0,\langle v_2-v_1\rangle_{12},\dots,\langle v_k-v_1\rangle_{12}).
$$

El **conjunto de inversos estructurales** (o **inversiones a cero**) de $v$ es la imagen de su órbita rotacional por $N$:
$$
\mathrm{Inv}_{0}(v)\;=\;\big\{\,N(\rho^j(v))\;\big|\;j=0,1,\dots,k-1\,\big\}\;\subseteq\;\mathbb{Z}_{12}^k.
$$

Esto produce **una estructura por cada elección de voz como referencia** (primera del arreglo tras rotar). Opcionalmente, se puede derivar un **representante canónico** aplicando un operador de canonización $C$ (p. ej., ordenar no‑decreciente, o bien la *normal form* de Forte). Definimos dos variantes de uso común:

- **Variante A (orden preservado):** usar $\mathrm{Inv}_0(v)$ tal cual, conservando la secuencia de voces.
- **Variante B (canónica de conjunto):** aplicar $C$ a cada elemento y, si además se desea ignorar el reordenamiento interno de las voces, considerar el multiconjunto resultante $\{\,C(w)\mid w\in\mathrm{Inv}_0(v)\,\}$.

**Ejemplo 2.1.** Con $v=[2,7,9]$,
- $N([2,7,9])=[0,5,7]$,
- $N([7,9,2])=[0,2,7]$,
- $N([9,2,7])=[0,5,10]$.

Por tanto
$$
\mathrm{Inv}_{0}([2,7,9])=\big\{[0,5,7],\;[0,2,7],\;[0,5,10]\big\}.
$$
Si se aplica ordenación interna (Var. B), quedan iguales por ya empezar en 0 y ser no‑decrecientes.

**Ejemplo 2.2 (triadas en posición cerrada).** Para $u=[0,4,7]$ (triada mayor en Do),
$$
\mathrm{Inv}_0(u)=\{[0,4,7],\;[0,3,8],\;[0,5,9]\},
$$
que corresponden a “raíz, 1ª inversión, 2ª inversión” tras *rotar y fijar a cero*.

---

## 3. Propiedades básicas
1. **Invarianza por transposición (estructural).** Para todo $t\in\mathbb{Z}$,
   $$
   \mathrm{Inv}_0\big(T_t(v)\big)=\mathrm{Inv}_0(v).
   $$
   *Prueba:* $N(\rho^j(T_t(v)))=\langle \rho^j(v)+t-(\rho^j(v))_1 - t\rangle_{12}=N(\rho^j(v))$.
2. **Compatibilidad rotacional.** $|\mathrm{Inv}_{\mathrm{mus}}(v)|=|\mathrm{Inv}_{0}(v)|$ cuando todas las rotaciones son distintas (posibles colisiones provienen de periodicidad y/o clases iguales tras normalizar).
3. **Invariante de clase T/rotación.** Si $w=T_t(\rho^j(v))$ para algún $t\in\mathbb{Z},j\in\{0,\dots,k-1\}$, entonces $\mathrm{Inv}_0(w)=\mathrm{Inv}_0(v)$. En particular, $\mathrm{Inv}_0$ clasifica acordes **a nivel estructural** dentro de la clase de equivalencia generada por transposición y rotación.
4. **Distinción respecto de $I$ (reflexión).** La inversión PC‑set $I_p(x)=\langle -x+p\rangle_{12}$ **no** coincide, en general, con ninguna combinación de $N$ y $\rho$. Nuestro uso de “inversión” aquí es **rotacional/estructural**, no reflexión.

---

## 4. Procedimientos (algoritmo)
**Pseudocódigo (Variante A):**
```
INPUT: v = (v1,...,vk) en Z
OUTPUT: Inversos musicales y estructurales

mus = []
str0 = []
para j en 0..k-1:
    w = rotar_izquierda(v, j)      # rho^j(v)
    mus.agregar(w)
    base = w[0]
    str0.agregar( (0, (w[1]-base) mod 12, ..., (w[k]-base) mod 12) )
retornar mus, str0
```

**Python (referencia mínima):**
```python
from typing import List, Tuple

def rot(v: List[int], j: int) -> List[int]:
    k = len(v)
    j %= k
    return v[j:] + v[:j]

def inv_mus(v: List[int]) -> List[List[int]]:
    return [rot(v, j) for j in range(len(v))]

def inv0(v: List[int]) -> List[List[int]]:
    out = []
    for j in range(len(v)):
        w = rot(v, j)
        base = w[0]
        out.append([ (x - base) % 12 for x in w ])
    return out
```

---

## 5. Consecuencias musicales
- **Inverso musical** organiza las **voces**: es la herramienta natural para describir reordenamientos cíclicos del voicing (p. ej., raíz→1ª→2ª inversión en triadas en posición cerrada, o permutaciones en disposiciones abiertas).
- **Inverso estructural “a cero”** organiza la **forma intervalar**: abstrae la altura absoluta y deja visible la huella estructural de cada rotación (una por elección de voz de referencia). Es invariante a transposición y permite comparar acordes por patrón, no por registro.

---

## 6. Resumen compacto
- $\mathrm{Inv}_{\mathrm{mus}}(v)=\{\rho^j(v)\}$ (órbita rotacional en $\mathbb{Z}^k$).
- $\mathrm{Inv}_0(v)=\{N(\rho^j(v))\}$ con $N(v)=(0,\langle v_2-v_1\rangle_{12},\dots)$ (estructura a cero).
- $\mathrm{Inv}_0$ es invariante por transposición y sirve como descriptor estructural; $\mathrm{Inv}_{\mathrm{mus}}$ describe reordenamientos de voces.
