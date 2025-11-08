# Dos formas distintas de invertir acordes

## 0. Preliminares y notación

- Denotaremos por \(\mathbb Z\) el conjunto de enteros (alturas absolutas en semitonos) y por \(\mathbb Z_{12}\) las clases de altura módulo 12.
- Un **voicing** (arreglo ordenado) de un acorde de \(n\) notas es una \(n\)-tupla \(v=(h_1,\dots,h_n)\in\mathbb Z^n\) ordenada de grave a agudo (permitiendo octavas repetidas).
- La proyección a clases de altura se denota por \(\pi(v)=(p_1,\dots,p_n)\in(\mathbb Z_{12})^n\), donde \(p_i\equiv h_i\pmod{12}\).
- Para \(a\in\mathbb Z_{12}\) y \(x\in\mathbb Z_{12}\), la **traslación** (transposición) es \(T_a(x)=x+a\ (\mathrm{mod}\ 12)\). Se extiende componente a componente a tuplas y conjuntos.
- Para una \(n\)-tupla \(w=(q_1,\dots,q_n)\in(\mathbb Z_{12})^n\), definimos la **normalización anclada en 0** como la lista ordenada crecientemente
  \[
  \mathrm{Norm}_0(w)\;=\;\text{orden ascendente de }\{(q_i-q_1)\bmod 12: 1\le i\le n\},
  \]
  que siempre comienza por \([0,\dots]\).

> Comentario. La normalización \(\mathrm{Norm}_0\) “olvida” la altura absoluta y deja sólo la forma estructural relativa al bajo de \(w\).

---

## 1. Inversión musical (rotación de voicing)

**Definición 1 (rotación con elevación de octava).** Sea \(v=(h_1,\dots,h_n)\in\mathbb Z^n\) con \(h_1\le\cdots\le h_n\). Definimos
\[
\rho(v)\;=\;\operatorname{sort}\big(h_2,\dots,h_n,\ h_1+12\big),
\]
la **primera inversión musical** (subir el bajo una octava y reordenar). Recursivamente, \(\rho^k(v)=\rho(\rho^{k-1}(v))\) para \(k=1,\dots,n-1\). El conjunto de **inversiones musicales** de \(v\) es
\[
\mathcal O_{\mathrm{mus}}(v)=\{\rho^k(v):\ k=0,1,\dots,n-1\}.
\]

Si nos interesa sólo el patrón de clases (ignorando octava), aplicamos \(\pi\) y podemos pensar en la rotación cíclica sobre \((\mathbb Z_{12})^n\).

---

## 2. Inversión estructural a cero (derivada de la musical)

**Idea clave (tal como se especifica):** *tomar cada inversión musical y **transponerla** para que su bajo quede en 0; esa lista normalizada es el inverso estructural “a cero”*.

**Definición 2 (inversos estructurales a cero).** Sea \(v\in\mathbb Z^n\) y, para cada \(k\), sea \(w^{(k)}=\pi\big(\rho^k(v)\big)\in(\mathbb Z_{12})^n\). Definimos el **inverso estructural a cero** asociado a \(\rho^k(v)\) como
\[
\sigma^{(k)}(v)\;=\;\mathrm{Norm}_0\big(w^{(k)}\big)\;=\;\text{orden ascendente de }\{\,(w^{(k)}_i-w^{(k)}_1)\bmod12\,\}_{i=1}^n.
\]
El conjunto de **inversos estructurales a cero** de \(v\) es
\[
\mathcal O_{\mathrm{estr},0}(v)=\{\,\sigma^{(k)}(v):\ k=0,1,\dots,n-1\,\}.
\]

> Observación. Por construcción, cada \(\sigma^{(k)}(v)\) es una forma \([0,\alpha_2,\dots,\alpha_n]\) que representa la “estructura” relativa de la inversión musical correspondiente, ignorando altura absoluta y registro.

---

## 3. Ejemplos concretos

### 3.1. Caso que produce \([0,3,7],[0,4,9],[0,5,8]\)
Sea \(v\) con clases \(\pi(v)=(2,7,10)\) (cualquier voicing cuyas clases sean \(\{2,7,10\}\)). Entonces sus rotaciones (como lista cíclica) son
\[(2,7,10),\quad(7,10,2),\quad(10,2,7).\]
Aplicando \(\mathrm{Norm}_0\) a cada una:
- \((2,7,10)\ \to\ [0,5,8]\)
- \((7,10,2)\ \to\ [0,3,7]\)
- \((10,2,7)\ \to\ [0,4,9]\)
Por tanto, \(\mathcal O_{\mathrm{estr},0}(v)=\{[0,3,7],[0,4,9],[0,5,8]\}\).

### 3.2. Qué ocurre con \(\{2,7,9\}\)
Si en cambio \(\pi(v)=\{2,7,9\}\), las rotaciones (mód 12) son \((2,7,9),(7,9,2),(9,2,7)\), y
- \((2,7,9)\ \to\ [0,5,7]\)
- \((7,9,2)\ \to\ [0,2,7]\)
- \((9,2,7)\ \to\ [0,5,10]\)
Esto ilustra que el trío \([0,3,7],[0,4,9],[0,5,8]\) corresponde exactamente a clases \(\{r, r+5, r+8\}\) (por ejemplo \(\{2,7,10\}\)), no a \(\{2,7,9\}\).

---

## 4. Propiedades básicas

1. **Cardinalidad.** Si \(\pi(v)\) no tiene duplicados, \(|\mathcal O_{\mathrm{mus}}(v)|=|\mathcal O_{\mathrm{estr},0}(v)|=n\). Con duplicados pueden colapsar algunos elementos.
2. **Invariancia por transposición.** Si \(v'\) es \(v\) transpuesto por cualquier \(a\in\mathbb Z\), entonces \(\mathcal O_{\mathrm{estr},0}(v')=\mathcal O_{\mathrm{estr},0}(v)\). La estructura a cero es ciega a la altura absoluta.
3. **Equivalencia de clases de conjunto.** Todos los \(\sigma^{(k)}(v)\) se obtienen por rotaciones de \(\pi(v)\) y anclaje a cero; por tanto, describen la misma *set-class* bajo traslaciones.

---

## 5. Algoritmo operativo (pasos)

Dado un voicing \(v\):
1. Ordenar \(v\) de grave a agudo.
2. Para cada \(k=0,\dots,n-1\):
   1. Construir \(\rho^k(v)\) (subir el bajo \(k\) veces y reordenar).
   2. Pasar a clases: \(w^{(k)}=\pi(\rho^k(v))\).
   3. Restar el primer elemento y reducir módulo 12.
   4. Ordenar el resultado crecientemente para obtener \(\sigma^{(k)}(v)=[0,\alpha_2,\dots,\alpha_n]\).
3. Devolver \(\{\sigma^{(k)}(v)\}\).

---

## 6. Comentario final

Conforme a la especificación: **inverso estructural a cero = inversión musical + transposición que pone el bajo en 0 + ordenación**. Los patrones \([0,3,7],[0,4,9],[0,5,8]\) se obtienen precisamente de triadas con diferencias \(5\) y \(8\) respecto al bajo (p. ej., clases \(\{r, r+5, r+8\}\)).

