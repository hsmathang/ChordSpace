# Discusión rigurosa y unificada: Distancia de *Voice-Leading* en espacios estratificados de acordes  
**Fecha:** 22 de febrero de 2026  
**Proyecto:** ChordSpace — métrica/disimilitud para reducción dimensional (UMAP/MDS)  
**Propósito del documento:** unificar, depurar y presentar de forma académica las conclusiones del debate entre las líneas argumentales atribuidas a “Claudio” y “Genri”, incorporando el estado lógico de los axiomas, las refutaciones formales y el plan de implementación.

---

## Resumen (Abstract)
Se estudia el problema de definir una función de distancia de *voice-leading* entre acordes de cardinalidad variable en un espacio geométrico/topológico estratificado (orbifold de Tymoczko; estratificación tipo Whitney en el espacio de todos los acordes). Se identifica un fallo estructural del enfoque clásico de asignación lineal (algoritmo húngaro) con penalización por “voces sobrantes” (*gap penalty*), al violar la continuidad inter-estratos requerida por la identificación topológica de duplicación por unísono (*divisi/merge*). Se evalúan tres familias de propuestas: (i) Transporte Óptimo tipo Wasserstein sobre medidas canónicas uniformes, (ii) formulaciones asintóticas basadas en expansiones a tamaño arbitrario y (iii) la distancia de Expansión Biyectiva (EB) evaluada en la expansión mínima común \(K=\max(|A|,|B|)\). Se concluye que (i) es una métrica en el espacio de medidas pero falla en continuidad estratificada bajo la canonicalización uniforme; (ii) puede degenerar a pseudométrica bajo normalización lineal; (iii) satisface no negatividad, simetría y continuidad estratificada (duplicación costo 0), pero **no** posee demostración general de la desigualdad triangular (M4) cuando \(K\) es dinámico por par, por lo que debe documentarse como **disimilitud** o **cuasi-métrica empírica**. Se propone un marco de documentación y verificación empírica para uso en UMAP con distancia precomputada, y se distinguen de forma explícita las tensiones ontológicas sobre si la multiplicidad de voces debe ser observable geométrico o se trabaja en el cociente por duplicación.

---

## 1. Introducción

### 1.1. Motivación musical y geométrica
La conducción de voces (*voice-leading*) modela el “trabajo” mínimo para transformar un acorde en otro mediante desplazamientos de voces. En modelos geométricos modernos, un acorde de \(n\) notas puede verse como un punto de \(\mathbb{R}^n\), y al factorizar por permutaciones (indistinguibilidad del orden simultáneo) se obtiene el orbifold \(\mathbb{R}^n/S_n\). El espacio de todos los acordes de todas las cardinalidades forma un objeto estratificado: cada estrato corresponde a una cardinalidad, y los estratos se conectan a través de loci de unísonos/duplicaciones (pérdida de rango).

### 1.2. Restricción topológica clave (continuidad inter-estratos)
La geometría estratificada sugiere que duplicar una nota (unísono) no debe introducir salto de distancia. Formalmente, se busca una compatibilidad de la forma:

- **Duplicación costo cero** (o equivalencia en el cociente):
\[
A \sim A\cup\{a\}\quad (a\in A)
\qquad\Rightarrow\qquad
d([A],[A\cup\{a\}])=0.
\]

En términos de perturbaciones:
\[
B(t)=A\cup\{a+t\},\quad t	o 0
\quad\Rightarrow\quad
\lim_{t	o 0} d([A],[B(t)])=0.
\]

### 1.3. Objetivo computacional
Construir una matriz de disimilitud/distancia apta para:
- **UMAP** con `metric='precomputed'` (tolerante a disimilitudes pero sensible a coherencia local),
- **MDS** (más exigente con metricidad si se exige interpretación euclídea clásica).

---

## 2. Formalización del objeto y axiomas

### 2.1. Acordes como multiconjuntos finitos
Sea \(\mathcal{P}\subset\mathbb{R}\) (alturas: MIDI/cents/log-frecuencia). Un acorde es un multiconjunto finito:
\[
A=\{a_1,\dots,a_m\}\subset \mathcal{P}.
\]

### 2.2. Relación de equivalencia estratificada por duplicación
Definimos \(\sim\) por eliminación de duplicados (en el sentido musical/topológico de unísonos):
\[
A\sim B\iff \mathrm{supp}(A)=\mathrm{supp}(B),
\]
donde \(\mathrm{supp}(A)\) es el conjunto de alturas distintas presentes en \(A\).
El espacio geométrico relevante para compatibilidad estratificada es el cociente:
\[
\mathcal{C}/\sim.
\]

> **Nota ontológica:** Si se desea que la multiplicidad sea observable geométrico (contrapunto “físico”), entonces esta identificación no es apropiada como igualdad geométrica. Este conflicto se formaliza en el “No-Go Theorem” (Sección 6).

### 2.3. Axiomas métricos estándar
Para una métrica \(d\) se requieren:
- (M1) \(d\ge 0\).
- (M2) \(d(A,B)=0 \iff A=B\) (o bien \(d([A],[B])=0\iff [A]=[B]\) si se trabaja en el cociente).
- (M3) \(d(A,B)=d(B,A)\).
- (M4) \(d(A,C)\le d(A,B)+d(B,C)\).

### 2.4. Propiedad estratificada adicional
- (E0) continuidad/compatibilidad inter-estratos (duplicación costo 0, límite suave).

---

## 3. Problema del enfoque original: Hungarian + gap penalty \(\gamma\)

### 3.1. Definición operacional previa
Se definía una matriz de costos \(C_{ij}=\mathrm{step}(a_i,b_j)\) y se resolvía asignación biyectiva mínima con el algoritmo húngaro. Para cardinalidades distintas, se añadían “nodos fantasma” con costo fijo \(\gamma\).

### 3.2. Fallo estructural (contraejemplo canónico)
Para:
\[
A=\{60,64,67\},\quad B(t)=\{60,64,67,67+t\},
\]
el emparejamiento perfecto deja una voz sin pareja, pagando \(\gamma\), por lo que:
\[
\lim_{t	o 0} d_{	ext{Hung}}(A,B(t))=\gamma
eq 0.
\]
Esto viola (E0) y, en consecuencia, rompe la continuidad estratificada.

**Conclusión:** el esquema Hungarian+\(\gamma\) queda descartado para el modelo estratificado.

---

## 4. Propuesta 1: Wasserstein/OT sobre medidas canónicas uniformes (WMC)

### 4.1. Idea
Asignar a cada acorde una medida discreta “canónica” (por ejemplo, uniforme sobre soporte) y medir distancia Wasserstein \(W_p\) entre medidas.

Ejemplo de canonicalización uniforme sobre soporte:
\[
\mu_A=rac{1}{|\mathrm{supp}(A)|}\sum_{x\in \mathrm{supp}(A)}\delta_x.
\]

### 4.2. Hecho: \(W_p\) es métrica en el espacio de medidas (bajo hipótesis estándar)
Bajo condiciones usuales, \(W_p\) satisface M1–M4 como distancia entre medidas de masa total fija.

### 4.3. Refutación: discontinuidad estratificada inducida por la canonicalización uniforme
Considerar:
\[
A=\{0,4,7\},\quad B(t)=\{0,4,7,t\},\quad t>0,\ t	o 0.
\]
Entonces:
\[
\mu_A=	frac13(\delta_0+\delta_4+\delta_7),
\quad
\mu_{B(t)}=	frac14(\delta_0+\delta_t+\delta_4+\delta_7).
\]
Al tomar \(t	o 0\), la masa se redistribuye globalmente (de 1/4 a 1/3 por átomo en el soporte reducido), induciendo transporte de masa macroscópico. El límite:
\[
\lim_{t	o 0} W_1(\mu_A,\mu_{B(t)})>0
\]
(es consistente con cálculos explícitos tipo CDF/redistribución).

**Causa raíz:** la aplicación \(\mathrm{supp}\mapsto \mathrm{Uniforme}(\mathrm{supp})\) es discontinua cuando cambia \(|\mathrm{supp}|\).

### 4.4. Conclusión sobre WMC
- ✅ Métrica en el espacio de medidas.
- ❌ No respeta (E0) bajo la canonicalización uniforme discutida.

**Veredicto:** WMC (tal como fue propuesta/criticada) se descarta para continuidad estratificada.

> **Nota:** Esto no “mata” OT como formalismo en general; refuta una elección específica de masas. Reabrir OT requeriría redefinir la representación o usar OT no balanceado, con pruebas adicionales.

---

## 5. Propuesta 2: expansión asintótica \(\inf_N\) (y su degeneración)

### 5.1. Idea
Arreglar la composición/triangularidad usando un tamaño común \(N\) y tomar un ínfimo sobre expansiones a \(N\ge \max(|A|,|B|)\).

### 5.2. Degeneración típica bajo normalización lineal
Si se normaliza el costo de matching por \(1/N\) y se permite \(N	o\infty\), una discrepancia fija \(t\) se diluye:
\[
	ilde d(\{0\},\{0,t\})=\lim_{N	o\infty}rac{t}{N}=0.
\]
En general, basta compartir al menos una nota para hacer tender el costo promedio a 0 replicando coincidencias.

### 5.3. Conclusión
- ✅ Puede restaurar continuidad estratificada y facilitar argumentos de triangularidad en principio.
- ❌ Puede degenerar a pseudométrica poco informativa.

**Veredicto:** descartada en su forma “asintótica ilimitada con normalización \(1/N\)”.

---

## 6. “No-Go Theorem” (tensión ontológica inevitable)

### 6.1. Tríada incompatible
Se intentó exigir simultáneamente:
- (A) **Consciencia de multiplicidad:** \(\{C\}
eq \{C,C\}\) como estados geométricos.
- (B) **Duplicación costo 0:** \(d(A,A\cup\{a\})=0\).
- (C) **Identidad estricta (M2) sobre multiconjuntos crudos:** \(d(X,Y)=0\iff X=Y\).

### 6.2. Demostración de incompatibilidad (esquema)
Si (B) vale, entonces:
\[
d(\{C\},\{C,C\})=0.
\]
Si además (C) vale, entonces \(\{C\}=\{C,C\}\) como objetos geométricos, contradiciendo (A).

### 6.3. Consecuencia metodológica
Para avanzar, debe elegirse explícitamente uno de estos marcos:

1. **Marco cociente (estratificado):** se trabaja en \(\mathcal{C}/\sim\); la multiplicidad no es observable geométrico.
2. **Marco físico (multiplicidad observable):** se abandona el costo cero exacto o se adopta un formalismo más rico (p.ej., OT no balanceado con penalización controlada), aceptando que el objeto no será una métrica clásica sobre el cociente simple.

En el cierre técnico adoptado, se asume el **Marco cociente**, porque se exige continuidad estratificada.

---

## 7. Solución adoptada: Expansión Biyectiva (EB) con \(K=\max(|A|,|B|)\)

### 7.1. Definición
Sea \(K=\max(|A|,|B|)\). Definir el conjunto de expansiones por duplicación:
\[
E_K(A)=\{A' 	ext{ multiconjunto de tamaño }K:\ \mathrm{supp}(A')\subseteq \mathrm{supp}(A)\}.
\]
La disimilitud EB se define como:
\[
d_{\mathrm{EB}}(A,B)=
\min_{A'\in E_K(A),\,B'\in E_K(B)}
\min_{\sigma\in S_K}\frac{1}{K}\sum_{i=1}^{K}\mathrm{step}(A'_i,B'_{\sigma(i)}).
\]

### 7.2. Qué garantiza EB (propiedades que sí se cumplen)
Asumiendo \(\mathrm{step}(x,y)\ge 0\) y simétrica:

- **(M1) No negatividad:** ✅  
  Suma y mínimo de no negativos.
- **(M3) Simetría:** ✅  
  Por simetría de \(\mathrm{step}\) y \(\sigma\mapsto\sigma^{-1}\).
- **(E0) Duplicación costo 0 y límite suave:** ✅  
  Si \(B=A\cup\{a\}\), existe expansión y matching de costo 0.  
  Si \(B(t)=A\cup\{a+t\}\), existe expansión tal que el costo \(\le t/K	o 0\).
- **Identidad en el cociente:** ✅ (si se formula sobre \(\mathcal{C}/\sim\))  
  \(d_{\mathrm{EB}}([A],[B])=0\) cuando difieren solo por duplicaciones.

### 7.3. Qué NO está garantizado
- **(M4) Desigualdad triangular global:** ⚠️ **Conjetura operacional**  
  Al variar \(K\) por par (\(K_{AB}
eq K_{BC}
eq K_{AC}\)), no hay un mecanismo estándar para componer matchings en una dimensión fija sin cambiar la definición. No existe demostración general conocida en el cierre actual.
- **Metricidad de `step`:** ⚠️  
  Si \(\mathrm{step}\) incluye truncamientos del tipo \(\min(|a-b|,24)\) o plegados mod 12, puede violar triangularidad; entonces M4 es imposible incluso si EB fuese “ideal”.

### 7.4. Clasificación formal recomendada
Por lo anterior, el output debe documentarse como:

> **Disimilitud perceptual / cuasi-métrica empírica**, no como “métrica estricta”.

Esto preserva rigor académico y evita afirmaciones falsas si M4 no está demostrada.

---

## 8. Implicaciones para MDS/UMAP

### 8.1. UMAP
UMAP opera sobre un grafo de vecindarios y una construcción difusa (*fuzzy simplicial sets*). En la práctica tolera disimilitudes que no son métricas estrictas, siempre que la estructura local sea consistente. Por tanto, EB es viable si se usa distancia precomputada.

**Recomendación:** ejecutar UMAP con `metric='precomputed'` y validar estabilidad.

### 8.2. MDS
MDS clásico (especialmente el métrico) se beneficia de distancias métricas y/o euclídeas; sin M4 pueden aparecer inconsistencias. Aun así, MDS puede ejecutarse con disimilitudes (p.ej., MDS no métrico), pero deben ajustarse expectativas.

**Recomendación:** si un comité exige “métrica demostrada”, incluir un respaldo alternativo (p.ej., una métrica sobre soportes) como baseline teórica, aunque mida otra noción musical.

---

## 9. Plan de acción (recomendado) para la tesis y el repositorio

### 9.1. Decisiones que deben quedar explícitas
1. **Declarar el marco (cociente):** se trabaja en \(\mathcal{C}/\sim\) donde duplicación es indistinguible.
2. **Nombrar correctamente EB:** disimilitud/cuasi-métrica, no métrica estricta.

### 9.2. Documentación obligatoria
- Probar y escribir M1 y M3 (triviales).
- Probar formalmente (E0): duplicación costo 0 y límite suave.
- Declarar M4 como **conjetura operacional** y acompañar con verificación empírica.

### 9.3. Auditoría empírica de M4 (triangularidad)
En un corpus grande de acordes:
- medir porcentaje de triples \((A,B,C)\) que violan \(d(A,C)\le d(A,B)+d(B,C)\),
- medir magnitud máxima/media de violación,
- reportar resultados en documentación.

### 9.4. Condición sobre `step`
- Si se busca acercarse a metricidad, preferir \(\mathrm{step}(x,y)=|x-y|\) o \(|x-y|^p\).
- Si se mantiene truncamiento/saturación por razones perceptuales, documentar explícitamente que `step` es disimilitud, y por tanto M4 no es esperable.

---

## 10. Implementación computacional (EB)

### 10.1. Esquema
1. \(K\leftarrow\max(|A|,|B|)\).
2. Generar \(E_K(A)\) y \(E_K(B)\) duplicando notas existentes (combinaciones con repetición).
3. Para cada par \((A',B')\in E_K(A)\times E_K(B)\):
   - construir matriz de costos \(K\times K\) con `step`,
   - resolver asignación óptima con `scipy.optimize.linear_sum_assignment`.
4. Retornar el mínimo global, normalizado por \(1/K\).

### 10.2. Complejidad práctica
Para cardinalidades musicales típicas (triadas–tétradas–péntadas), \(K\le 6\) y el número de expansiones es pequeño. El costo es viable.

---

## 11. Conclusión general
- El enfoque Hungarian+\(\gamma\) es incompatible con continuidad estratificada.
- WMC (uniforme canónica) es métrica en medidas pero falla (E0) por discontinuidad inducida por redistribución de masa.
- \(\inf_N\) asintótico con normalización \(1/N\) puede degenerar.
- EB con \(K=\max\) es la solución operacional más razonable: respeta estratos (duplicación 0, límite suave) y es computacionalmente viable, pero **no** tiene demostración general de M4, y además `step` puede romper triangularidad.

**Por tanto:** EB debe presentarse académicamente como **disimilitud/cuasi-métrica empírica en el cociente estratificado**, con auditoría empírica de triangularidad y documentación explícita del “No-Go” ontológico.

---

## Apéndice A. Checklist de “qué se cumple / qué no”
- **EB:** M1 ✅, M3 ✅, E0 ✅, M2 ✅ en cociente, M4 ⚠️ (conjetura), `step` ⚠️ (depende diseño).
- **WMC uniforme:** M1–M4 ✅ (en medidas), E0 ❌ (en frontera de soporte).
- **Hungarian+\(\gamma\):** E0 ❌.

