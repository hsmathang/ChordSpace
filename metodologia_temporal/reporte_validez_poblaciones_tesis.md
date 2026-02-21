# Reporte De Validez De Poblaciones Combinatoriales Para La Tesis ChordSpace

## 1. Objetivo

Documentar, con criterio metodologico explicito, cuando es valido usar poblaciones combinatoriales de acordes en ChordSpace como representacion experimental de estimulos definidos en articulos perceptuales y en corpus barrocos.

---

## 2. Tesis Central

Es valido usar poblaciones combinatoriales para validacion en tesis cuando se cumple esta condicion:

- El dominio de estimulos del estudio externo puede expresarse como subconjunto de acordes definidos por `pitch chord` en MIDI, con restricciones explicitas de clase de altura, registro, cardinalidad y voicing.

Esto aplica a tu pipeline porque:

- El generador trabaja en MIDI absoluto y conserva registro.
- El filtrado permite imponer restricciones musicales concretas.
- Los experimentos de tesis evaluan estructura geometrica y ordenamientos perceptuales, no sintesis acustica identica al paper.

---

## 3. Marco Formal De Equivalencia Representacional

Definimos un universo combinatorial:

\[
U(A, o_{min}, o_{max}, K) = \bigcup_{k \in K} \left\{ (n_1,\dots,n_k) : n_1<\dots<n_k,\ \pi(n_i)\in A,\ o_{min}\leq oct(n_i)\leq o_{max} \right\}
\]

Donde:

- \(A \subseteq \{0,\dots,11\}\) es el alfabeto de clases de altura.
- \(K\) es el conjunto de cardinalidades.
- \(n_i\) son notas MIDI absolutas.

Un estudio externo \(S\) esta bien representado si existe un mapeo \(M\) tal que:

- \(M(S) \subseteq U(A, o_{min}, o_{max}, K)\).
- Las reglas no especificadas por el estudio (por ejemplo voicing cuando solo hay pc-sets) se fijan por politica reproducible unica.
- Las metricas de validacion usan la misma representacion para todos los grupos comparados.

---

## 4. Formalizacion Matematica Del Generador Y Del Control Poblacional

Esta seccion formaliza el comportamiento real del codigo en:

- `services/combinatorial_generator.py`
- `services/population_filter.py`
- `tools/data_access.py` (`ChordFilters`)
- `tools/population_utils.py`

### 4.1 Parametros Primarios Y Notacion

Sea:

- \(A = \{a_1,\dots,a_{|A|}\} \subseteq \{0,\dots,11\}\): pitch classes permitidas (`alphabet`).
- \(o_{min}, o_{max} \in \mathbb{Z}\): limites de octava MIDI (`octave_min`, `octave_max`).
- \(K \subset \mathbb{N}\): cardinalidades solicitadas (`cardinalities`).
- \(\sigma \in \{0,1\}\): indicador de modo estructural (`structural_mode`), con \(\sigma=1\) para catalogo estructural.

Parametros de control posterior (filtros):

- Cardinalidad objetivo \(K_f\).
- Span \(s_{min}, s_{max}\).
- Maximo intervalo interno \(\tau\).
- Reglas de pitch-class \(A_{inc}\), \(A_{exc}\), y modo (`contains_all`, `contains_any`, `subset_of`).
- Reglas sobre secuencias de intervalos \(\mathcal{P}\) y/o valores permitidos \(V_I\).
- Lista de codigos \(C_f\).

### 4.2 Construccion Del Universo MIDI

En codigo, el universo base se construye como:

\[
V_0 = \left\{12(o+1)+a \;:\; o\in[o_{min},o_{max}]\cap\mathbb{Z},\ a\in A \right\}\cap[0,127]
\]

Luego se agrega una nota de borde:

\[
a_{min}=\min(A),\qquad b = 12(o_{max}+2)+a_{min}
\]

\[
V =
\begin{cases}
V_0 \cup \{b\}, & \text{si } 0\le b\le 127 \text{ y } b\notin V_0\\
V_0, & \text{en otro caso}
\end{cases}
\]

Definimos \(m = |V|\). En el regimen usual (sin clipping MIDI por extremos), se aproxima por:

\[
m \approx |A|\,(o_{max}-o_{min}+1) + \beta,\qquad \beta\in\{0,1\}
\]

donde \(\beta\) representa la insercion efectiva de la nota de borde.

### 4.3 Enumeracion Combinatorial Sin Modo Estructural

Para cada \(k\in K\), el generador recorre todas las combinaciones estrictamente crecientes:

\[
\mathcal{C}_k = \{(n_1,\dots,n_k)\in V^k : n_1<\dots<n_k\}
\]

Conteo exacto:

\[
|\mathcal{C}_k| = \binom{m}{k}
\]

Total generado en modo normal (\(\sigma=0\)):

\[
N_{raw}(m,K)=\sum_{k\in K}\binom{m}{k}
\]

Crecimiento marginal al agregar una sola nota MIDI al universo:

\[
N_{raw}(m+1,K)-N_{raw}(m,K)=\sum_{k\in K}\binom{m}{k-1}
\]

Asintoticamente, para \(m\) grande y \(k\) fijo:

\[
\binom{m}{k}\sim \frac{m^k}{k!}
\]

Por tanto, \(k_{max}=\max(K)\) domina el costo y el tamano del catalogo.

### 4.4 Transformaciones Internas Por Acorde

Dado \(c=(n_1,\dots,n_k)\), el pipeline computa:

- Vector intervalar adyacente:
\[
\Delta(c) = (n_2-n_1,\dots,n_k-n_{k-1})
\]
- Span:
\[
s(c)=n_k-n_1
\]
- Pitch classes del acorde:
\[
P(c)=\{n_i \bmod 12\}_{i=1}^k
\]
- Offsets estructurales (anclados al bajo):
\[
\Omega(c)=(0,n_2-n_1,\dots,n_k-n_1)
\]

Ademas guarda metadatos de trazabilidad (`notes_abs_json`, `__root_midi`, `__structure_id`, mascaras absolutas e identificador estable hash).

### 4.5 Modo Estructural: Definicion, Conteo Y Sentido Musical

El modo estructural (\(\sigma=1\)) no devuelve todas las instancias absolutas, sino clases de equivalencia por patron de offsets.

Definimos:

\[
c \sim c' \iff \left(|c|=|c'|\right)\land\left(\Omega(c)=\Omega(c')\right)
\]

Para cada \(k\), el catalogo estructural es el cociente:

\[
\mathcal{S}_k=\mathcal{C}_k/\sim
\]

y su tamano total:

\[
N_{struct}(K)=\sum_{k\in K}|\mathcal{S}_k|,\qquad N_{struct}(K)\le N_{raw}(m,K)
\]

Conteo exacto (expresado como union de patrones posibles segun bajo):

\[
|\mathcal{S}_k|=\left|\bigcup_{r\in V}\left\{\{0\}\cup T:\ T\subseteq S_r,\ |T|=k-1\right\}\right|
\]

con
\[
S_r=\{v-r:\ v\in V,\ v>r\}.
\]

El representante canonico de cada clase se construye en codigo con raiz base
\[
r_0=12(o_{min}+1),
\]
ajustada por octavas para cumplir \(0\le r+\max\Omega\le127\), y luego:
\[
\hat c = (r+\omega_1,\dots,r+\omega_k),\quad \Omega=(\omega_1,\dots,\omega_k).
\]

Funcion musical de `structural_mode`:

- Preserva geometria intervalar absoluta respecto al bajo (densidad/dispersion de voicing).
- Colapsa identidad tonal absoluta (raiz/pitch-class especifica) y registro absoluto.
- Es adecuado para estudiar familias de voicing; no para estudios donde la identidad exacta de pitch-class es variable explicativa central.

### 4.6 Filtros Como Operadores Sobre Conjuntos

Sea \(G\) la poblacion cruda generada. Los filtros implementados actuan como operadores:

\[
G^\star = F_{code}\circ F_{int}\circ F_{pc}\circ F_{maxint}\circ F_{span}\circ F_{card}(G)
\]

En terminos logicos, cada acorde final debe satisfacer simultaneamente todas las restricciones activas.

Condiciones formales:

- Cardinalidad:
\[
|c|\in K_f
\]
- Span:
\[
s_{min}\le s(c)\le s_{max}
\]
- Maximo intervalo interno:
\[
\max(\Delta(c))\le\tau
\]
- Inclusion de pitch classes (`contains_all`):
\[
A_{inc}\subseteq P(c)
\]
- Inclusion (`contains_any`):
\[
A_{inc}\cap P(c)\neq\varnothing
\]
- Inclusion (`subset_of`):
\[
P(c)\subseteq A_{inc}
\]
- Exclusion de pitch classes:
\[
P(c)\cap A_{exc}=\varnothing
\]
- Patrones intervalares exactos:
\[
\Delta(c)\in\mathcal{P}
\]
- Patrones por subsecuencia:
\[
\exists p\in\mathcal{P}: p\sqsubseteq \Delta(c)
\]
- Valores intervalares permitidos (`any_value`):
\[
\Delta(c)\cap V_I\neq\varnothing
\]
- Codigo:
\[
code(c)\in C_f
\]

### 4.7 Dedupe Y Conservacion De Registro

La deduplicacion (`dedupe_population`) define una llave principal por mascara absoluta y raiz:

\[
\kappa(c)=\left(\texttt{abs\_mask\_int}(c),\ \texttt{__root\_midi}(c)\right)
\]

Si faltan esos campos, usa fallback:

\[
\kappa_{fb}(c)=\left(code(c),\Delta(c)\right)
\]

El operador de dedupe selecciona la primera ocurrencia por llave con orden estable por prioridad de fuente. Formalmente:

\[
D(G)=\{c\in G:\ c \text{ es el primer elemento de su clase por } \kappa\}
\]

Consecuencia metodologica:

- Dos acordes con mismo contenido de pitch classes pero en octavas distintas no se colapsan cuando `__root_midi` esta disponible.
- Esto preserva validez para experimentos sensibles a registro.

### 4.8 Sensibilidad Del Espacio A Cada Parametro

Resumen de impacto sobre tamano y semantica:

| Parametro | Notacion | Efecto matematico principal | Efecto musical principal |
|---|---|---|---|
| `alphabet` | \(A\) | \(m\) crece casi lineal con \(|A|\), \(N\) crece combinatoriamente | Amplia/restringe cromaticidad |
| `octave_min`, `octave_max` | \(o_{min},o_{max}\) | \(m\propto(o_{max}-o_{min}+1)\) | Cambia registro y dispersion posible |
| `cardinalities` | \(K\) | Terminos \(\binom{m}{k}\), domina \(k_{max}\) | Controla numero de voces |
| Nota de borde automatica | \(\beta\) | agrega hasta 1 nota al universo | Permite frontera superior coherente |
| `structural_mode` | \(\sigma\) | pasa de instancias \(N_{raw}\) a clases \(N_{struct}\) | Agrupa familias de voicing |
| `span_min/span_max` | \(s_{min},s_{max}\) | poda por desigualdad sobre \(s(c)\) | Controla compacidad/apertura |
| `max_internal_interval` | \(\tau\) | poda por \(\max\Delta(c)\) | Evita saltos internos grandes |
| `include_pitch_classes` + modo | \(A_{inc}\) | restriccion de pertenencia sobre \(P(c)\) | Impone vocabulario tonal/cromatico |
| `exclude_pitch_classes` | \(A_{exc}\) | elimina clases no deseadas | Forza estilo/evita cromas |
| `interval_mode/patterns/values` | \(\mathcal{P},V_I\) | restriccion sobre \(\Delta(c)\) | Controla morfologia intervalar |
| `codes` | \(C_f\) | seleccion exacta discreta | Aisla tipos concretos |

### 4.9 Complejidad Computacional Practica

Sin filtros, el cuello de botella es la enumeracion:

\[
T_{raw}=O\!\left(\sum_{k\in K}\binom{m}{k}\cdot k\right)
\]

`structural_mode=True` reduce tamano de salida, pero no evita recorrer combinaciones en la implementacion actual; por eso reduce memoria de salida mas que costo bruto de enumeracion.

Implicacion operacional:

- Para experimentos con \(k\) altos o rango amplio, conviene combinar:
1. Restriccion temprana de \(A\), \(o_{min},o_{max}\), \(K\).
2. Filtros agresivos (`span`, `max_internal_interval`, reglas intervalares).
3. Muestreo posterior cuando el objetivo estadistico no requiere enumeracion exhaustiva.

---

## 5. Evidencia En Implementacion Del Repositorio

Componentes que soportan el marco anterior:

- Generacion: `services/combinatorial_generator.py`
- Filtros: `services/population_filter.py`
- Contrato de filtros: `tools/data_access.py` (`ChordFilters`)
- Deduplicacion con preservacion de registro: `tools/population_utils.py`
- Expansion por escala: `tools/population_builders.py`
- Flujo GUI y reporte: `ui/launcher/views/app_new.py`, `tools/reporting/report_builder.py`
- Marco experimental: `metodologia_temporal/evaluacion-experimentos.md`

Propiedades clave del pipeline:

- Acorde como `pitch chord` (no colapso por transposicion/octava).
- No unisonos internos (tupla estrictamente creciente).
- Metadatos de estructura (`__structure_id`) y raiz real (`__root_midi`).
- Dedupe que evita colapsar voicings en octavas distintas.

---

## 6. Validez Por Caso De Uso

### 6.1 Estudios Perceptuales Cromaticos (Bowling 2018)

Resultado de validez:

- Valido como equivalencia de dominio simbolico.

Justificacion:

- Bowling trabaja con diadas, triadas y tetradas cromaticas en una octava.
- Eso es representable como \(U(\{0,\dots,11\}, o, o, \{2,3,4\})\).
- Si el paper define subconjuntos por construccion intervalar sobre bajo fijo, ese conjunto sigue siendo subconjunto del universo combinatorial.

Implicacion:

- Puedes evaluar correlaciones perceptuales con tus distancias sin requerir que la sintesis timbrica sea identica al paper, siempre que declares esta diferencia.

### 6.2 Consonancia Simultanea (Harrison Y Pearce 2020)

Resultado de validez:

- Valido para analisis comparativo de ordenamientos y proximidades.

Justificacion:

- El estudio integra interferencia, harmonicidad y familiaridad cultural.
- Tu modelo cubre principalmente componente sensorial/estructural.
- Por tanto la validacion es de alineacion parcial de constructo, no de equivalencia total de modelo cognitivo.

Implicacion:

- Usa Spearman/partials y declara explicitamente alcance: similitud sensorial y geometrica, no consonancia completa multicomponente.

### 6.3 Corpus Barrocos (hcorp/JSB)

Resultado de validez:

- Valido para validar separacion estilistica y proximidad funcional condicionada por contexto.

Justificacion:

- Verticalidades del corpus se traducen directamente a `pitch chord`.
- Puedes construir tres grupos comparables:
1. Poblacion real de corpus.
2. Superpoblacion sintetica con mismas restricciones de rango/cardinalidad.
3. Controles de ruido/extremos con mismo soporte.

Implicacion:

- La comparacion Bach vs ruido/extremos es metodologicamente defendible si controlas rango y cardinalidad entre grupos.

---

## 7. Limites De Validez Y Mitigaciones

Limites:

- No replica automaticamente timbre, dinamica, articulacion ni contexto temporal fino del paper.
- La funcion armonica en Bach no emerge sola; requiere etiquetas o analisis armonico adicional.
- Estudios con estimulos exactos exigen reproducir exactamente esos acordes.

Mitigaciones:

- Mantener un bloque de estimulos exactos como subconjunto ancla.
- Declarar politica de voicing cuando el origen solo trae pc-sets.
- Ejecutar analisis de sensibilidad por rango (registro bajo/medio/alto).
- Reportar siempre parametros de poblacion y preset usado.

---

## 8. Criterio Operativo Para Tesis

Usar esta regla de decision:

1. Si el estudio define estimulos exactos, importar esos estimulos y evaluarlos tal cual.
2. Si define clase de estimulos, generar poblacion combinatorial equivalente con politica fija.
3. Si hay variables no cubiertas por ChordSpace, declarar que la inferencia se limita al subconstructo sensorial-geometrico.

---

## 9. Integracion Con GUI Y Trazabilidad En HTML

Estado recomendado de trabajo en app:

- Construir poblacion en modo combinatorial.
- Guardar configuracion como preset con nombre semantico.
- Reutilizar el preset en ejecuciones posteriores.
- Verificar en reporte HTML seccion de metadatos con estos campos:
- `preset_name`
- `preset_description`
- `alphabet`, `cardinalities`, `octave_min`, `octave_max`, `structural_mode`
- filtros aplicados.

Persistencia de presets:

- Archivo: `outputs/gui_presets/population_presets.json`

---

## 10. Nombres Sugeridos De Presets Para Tesis

- `validacion_bowling_octava4_2_3_4`
- `validacion_hp2020_estimulos`
- `validacion_bach_real_verticalidades`
- `validacion_bach_sintetica_3_4_voces`
- `validacion_ruido_control_3_4_voces`
- `validacion_extremos_clusters_4_8`

---

## 11. Texto Corto Reutilizable Para El Manuscrito

"La estrategia de validacion emplea poblaciones combinatoriales de acordes definidas en el mismo espacio representacional del modelo (pitch chords MIDI con registro absoluto). Para cada estudio externo, se establece un mapeo explicito entre el dominio de estimulos reportado y un subconjunto del universo combinatorial generado por ChordSpace. Cuando los estimulos exactos estan disponibles, se incorporan directamente; cuando el estudio solo define clases de estimulos, se fija una politica de voicing reproducible y se documentan todas las restricciones. Bajo este protocolo, la inferencia resultante es valida para el constructo sensorial-geometrico modelado por ChordSpace."

---

## 12. Referencias

- Bowling, D. L., Purves, D., & Gill, K. Z. (2018). Vocal similarity predicts the relative attraction of musical chords. PNAS.  
  https://pubmed.ncbi.nlm.nih.gov/30030333/

- Harrison, P. M. C., & Pearce, M. T. (2020). Simultaneous consonance in music perception and composition. Psychological Review.  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7006947/

- hrep (representacion de acordes y voicing).  
  https://github.com/pmcharrison/hrep

- incon (modelado de consonancia).  
  https://github.com/pmcharrison/incon

- hcorp (corpora, incluye Bach chorales).  
  https://github.com/pmcharrison/hcorp

- JSB Chorales dataset.  
  https://github.com/czhuang/JSB-Chorales-dataset

- Durham Chord Dataset.  
  https://pmcharrison.github.io/durham-chord-dataset/
