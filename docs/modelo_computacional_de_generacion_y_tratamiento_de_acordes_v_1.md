# Modelo computacional — Generación y tratamiento de poblaciones de acordes

**Versión**: 1.1\
**Autor**: Equipo de I+D (rol: Dr. en Ing. de Software / Dr. en Matemáticas Aplicadas y CC)\
**Enfoque**: generación combinatoria rápida (sin unísonos), tratamiento inmediato, caché ligera opcional, integración con vectorización y reducción ya existentes en el repositorio.

---

## 0) Alcance y principios

- **Alcance**: definición formal y técnica de la primera capa del sistema: **generación** de acordes y su **tratamiento inmediato** (representaciones, firmas, filtros, caché). Vectorización de rugosidad y reducción dimensional **no** se redefinen aquí (ya existen en el repo), pero se acoplan a las salidas de este documento. EL DOCUMENTO BUSCA EXPLORAR LA VIABILIDAD DE CAMBIAR A OTRO GUI DE GENERACION Y EXPLORACION DE ACORDES MÁS LIMPIO YADECUADO ALAS NECESIDADES DE EXPLORACION LA DB Y LA CONSULTA SE HAN MOSTRADO LIMITADOS PARA CIERTOS EJERICIOS DE EXPLORACION COMO EL DE DADO UNA ESCALA CUALQUIERA CONSTRUIUR TODOS LOS ACORDES POSIBLES SOBRE ESA ESCALA, PLANTEAMOS ENTONCES RENOVAR  ESA ETAPA DEL REPO RESCTAR LOQUE NOS SIRVA Y LO NUEO DEJARLO CON LAS MISMAS TIPOS DE SALIDAS QUE ESPERAN LAS ETPAS DONDE SE TRASNOFRMAN LSOA CORDES EN VECOTRES Y SE PROCESAN PARA TEMINAR SIENDO VISUALZIADOS EN 2DIM O PARA ALGORITMOS DE SUSTITUCION ARMONICA QUE PERMITAN DARLE AL USUARIO MUSICO OPCIONES DE SUSTITUCION ENTRE OTRAS, ESTE NUEVO GUI QUIERE SER INNTEGRADO MAS TIPO DASHBOARD.
- **Dominio**: **12‑TET** con A4 configurable (no microtonal en esta fase; dejar notas para futuro).
- **Entrada “musical”**: el usuario define **pitch classes** (PCs que funcionaran como alfabeto) y **rangos de octava**; el motor trabaja en **MIDI** absolutos y proyecta a \(\mathbb{Z}_{12}\) cuando procede.
- **Restricciones**: **no hay unísonos** (no se repite el mismo número MIDI en un acorde). Se permiten PCs repetidas en **octavas distintas**.
- **Persistencia**: **sin DB** obligatoria AUnque ya habiendo una DB podriamos plantear guardar los acordes ahi en todo caso es teantitovo lo sigueinte: Se prioriza velocidad con **caché ligera** conmutables: `none | memory | parquet | sqlite` (append‑only con llave única). Si el proyecto escala, se podrá migrar sin romper API.

---

## 1) Modelo formal mínimo

### 1.1 Universos y representaciones

- **Pitch class**: \(p \in \mathbb{Z}_{12}=\{0,\dots,11\}\).
- **Octava**: \(o \in \mathbb{Z}\) (convención MIDI con \(C4=60\)).
- **MIDI**: \(m = 12\,(o+1)+p\), enteros \([0,127]\) para piano estándar.
- **Afinación**: \(\tau=\text{A4\_Hz}\) (sólo documentada en 12‑TET).
- **Frecuencia** (opcional): \(f(m)= 440\cdot 2^{(m-69)/12}\).

### 1.2 Acorde absoluto y estructural

- **Acorde absoluto**: \(A=(m_1<\dots<m_k)\), \(k\ge 2\), **MIDI distintos**.
- **Acorde estructural**: proyección que olvida altura/transposición: `pc_set(A)`, `canon_0(A)` (PC mínima = 0) y `intervals_mod12(A)`.
- **Span** (apertura): \(\mathrm{span}(A)=m_k-m_1\) (semitonos).

### 1.3 Identificadores para deduplicación y consulta

- \`\` (único por conjunto de alturas absolutas): \(\sum_{m\in A}2^m\) (int arbitrario grande).
- \`\` (12 bits): \(\sum_{p\in pc\_set}2^p\).
- \`\` (canónico estructural): tupla `(pc_mask_canon0, interval_sig)` con firma ordenada de `intervals_mod12(A)`.

---

## 2) Parámetros de generación (gramática de entrada)

### 2.1 Alfabeto y rango

- **Alfabeto**: \(S \subseteq \mathbb{Z}_{12}\) (p. ej., diatónica de Do: \(\{0,2,4,5,7,9,11\}\)).
- **Rango de octavas**: por defecto **cerrado** \([o_{min}, o_{max}]\). Opción `edge_pc0=True` para incluir además la **primera nota** (pc=0) de la octava \(o_{max}+1\) —equivale a “toma hasta la primera nota de la siguiente octava”.

### 2.2 Cardinalidades y restricciones

- **Cardinalidades**: lista \(N \subset \{2,3,\dots\}\).
- **Sin unísonos**: combinaciones sin reposición; si entran duplicados MIDI en eventos externos, se filtran.
- **Reflexión**: puede **salirse de** \(S\) (modo reflexión activo) por su naturaleza dependiente de distancias al eje.

### 2.3 Afinación

- **12‑TET** con A4 variable. Microtonal **fuera de alcance** (documentado en “Futuro”).

---

## 3) Modos de generación

### 3.1 Combinatorial **Total** (absoluta)

- **Universo MIDI**: \(M=\{m: p\in S,\ o\in[o_{min},o_{max}]\}\) + `edge_pc0` opcional.
- Para cada \(k\in N\): recorrer \(\binom{|M|}{k}\) (sin reposición), ordenar y emitir \(A\).
- **Metadatos**: `abs_mask_bigint`, `pc_mask`, `n=k`, `span`, `octave_vector`, `origin=GEN_TOTAL(S,O,N,edge_pc0)`.
- **Complejidad**: combinatoria; requiere **streaming** y **corte temprano** (\§4.3).

### 3.2 Combinatorial **Estructural**

- **E1. Fijando ancla 0**: generar patrones en \(\mathbb{Z}_{12}\) con `max_span_struct\in{12,24}`.
- **E2. Proyección Total→Estructural**: ejecutar **Total**, luego `canon_0` y colapsar por `struct_id`.
- **Uso**: si el objetivo es estudiar **formas**, preferir E1 y (si se desea) elevar luego a absolutos bajo políticas de registro/span.

---

## 4) Tratamiento inmediato tras generación

### 4.1 Representaciones canónicas y vectores derivados

**Entradas**: `midi_list: list[int]` con \(k\ge 2\), ordenada y sin unísonos.

**Salidas**:

1. **Vector MIDI original**: `tuple(midi_list)`.
2. **Representaciones en \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*****\(\mathbb{Z}_{12}\)**:
   - `pc_tuple(m)` → PCs en **orden** del voicing.
   - `pc_set_sorted(m)` → PCs **sin repetición** y **ordenadas**.
   - `pc_tuple_canon0(m)` → PCs con **mínimo = 0**.
3. **Vector de croma**:
   - `chroma01(m)` → \(\mathbb{R}^{12}\) binario (0/1).
   - `chroma_count(m)` → \(\mathbb{R}^{12}\) de conteo (opcional para futuros multiconjuntos de PC).
4. **Distancias internas**:
   - `adjacent_intervals_semitones(m)` → \((m_{i+1}-m_i))_{i=1}^{k-1}\).
   - `pairwise_dist_list_semitones(m)` → todas las parejas \((m_j-m_i))\).
   - `pairwise_dist_hist_semitones(m, max_bins)` → histograma real (hasta `span` o longitud fija).
   - `pairwise_dist_hist_mod12(m)` → histograma **mod 12** de 12 bins (clase 0 excluida por política sin unísonos).
5. **Span y firmas**: `span_of`, `pc_mask_of`, `abs_mask_bigint_of`, `struct_id_of`.

> **Invarianzas**: `pairwise_dist_hist_mod12` es **invariante** por transposición; `pc_mask` rota circularmente.

**Firmas de referencia (archivo ****\`\`****)**

```python
def pc_tuple(midi_list: list[int]) -> tuple[int, ...]: ...
def pc_set_sorted(midi_list: list[int]) -> tuple[int, ...]: ...
def pc_tuple_canon0(midi_list: list[int]) -> tuple[int, ...]: ...

def chroma01(midi_list: list[int]) -> np.ndarray: ...
def chroma_count(midi_list: list[int]) -> np.ndarray: ...

def adjacent_intervals_semitones(midi_list: list[int]) -> tuple[int, ...]: ...
def pairwise_dist_list_semitones(midi_list: list[int]) -> tuple[int, ...]: ...
def pairwise_dist_hist_semitones(midi_list: list[int], max_bins: int|None=None) -> np.ndarray: ...
def pairwise_dist_hist_mod12(midi_list: list[int]) -> np.ndarray: ...

def span_of(midi_list: list[int]) -> int: ...
def pc_mask_of(midi_list: list[int]) -> int: ...
def abs_mask_bigint_of(midi_list: list[int]) -> int: ...
def struct_id_of(midi_list: list[int]) -> tuple[int, tuple[int, ...]]: ...
```

### 4.2 ¿DB o no? — Marco de decisión orientado a **velocidad**

- **Preferencia**: exploración rápida **sin DB**; usar **caché** conmutable:
  - `cache=none` (efímero), `cache=memory` (por defecto), `cache=parquet` o `cache=sqlite` (persistencia ligera, llave única `abs_mask_bigint` y `struct_id` para estructural).
- **Razonamiento**: generar+filtrar con corte temprano es más barato que I/O+index en la mayoría de sesiones interactivas. DB sólo gana si habrá **re‑uso masivo** y **consultas** repetidas en poblaciones muy grandes.
- **Decisión**: implementar primero `memory`; ofrecer `sqlite` como opción.

### 4.3 Auditoría y refactor de **filtros existentes** en el repo

**Instrucciones al desarrollador**:

1. Localizar filtros por **PCs**, **patrones de intervalos**, **span**, **cardinalidad**, y cualquier implementación de `interval_to_ui_bin` o equivalentes.
2. Extraer la lógica común a `filters/` y **centralizar** cálculos de intervalos en `core/encoding.py` (una sola fuente).
3. Añadir **hooks de corte temprano** en `gen_total(...)`:

```python
early_filters = {
  "max_span": Optional[int],
  "must_have_pcs": Optional[set[int]],
  "must_avoid_pcs": Optional[set[int]],
  "interval_pattern": Optional[tuple[int, ...]],  # e.g., (4,3)
}
```

4. Crear “goldens” (tríadas diatónicas) para validar que los resultados refactorizados coinciden con los actuales.

---

## 5) Construcciones complementarias

- Absoluta (en MIDI) y modular (en PCs), con traza en `origin`.

### 5.2 Rotación / “inversión de voicing” **exacta** (sin reordenar más que subir el bajo)

- Definición: para \(A=(m_1,\dots,m_k)\), \(R(A)=\text{ordenar}(m_2,\dots,m_k,m_1+12)\).
- **No** se aplica normalización adicional: se **sube el bajo una octava** y se conserva el orden natural resultante.

**API**

```python
def rotate_bass_up(midi_list: list[int]) -> tuple[int, ...]:
    assert len(midi_list) >= 2
    moved = midi_list[1:] + [midi_list[0] + 12]
    return tuple(sorted(moved))
```

- \(H_a(p)=2a-p\ (\bmod\ 12)\). Puede emitir PCs **fuera de** \(S\) (permitido en modo reflexión). Política de octavas: re‑anclaje en rango por wrap/clip (parámetro).

### 5.4 Neo‑riemannianas (P, L, R)

- **Dominio**: sólo **tríadas mayor/menor** en **estructural** (ganchos documentados para cuatríadas futuro).

---

## 6) Detección vs construcción (heurística)

- **Detectar** si ya existe una población grande (Total/Estructural) y las relaciones son rápidas con índices por `struct_id`/`pc_mask`. ESTO PARA MEJORAR LA FORMA EN QUE SI VISUALZIAN EN SCATTERS, ES DECIR ESA FUNCION DE OPACIDAD Y TAMAÑO PARA DESTACAR CIERTOS ACORDES SE PUEDE EXPLORTAR DE MANERAS DISITNITAS Y UTILES AL USUAARIO MUSICO.
- **Construir** si el universo es pequeño o si la regla genera pocos candidatos por acorde.
- Comenzar con **detección** para inversiones/transposiciones; para reflexión/PLR ofrecer ambos caminos y medir.

---

## 7) API propuesta — guía para programar

### 7.1 Generación (`gen/`)

```python
# gen/universe.py
def build_midi_universe(S: set[int], o_min: int, o_max: int, edge_pc0: bool=False) -> list[int]: ...

# gen/generate.py
from typing import Iterable, Iterator

def gen_total(S: set[int], o_min: int, o_max: int, N: Iterable[int],
              edge_pc0: bool=False, early_filters: dict|None=None) -> Iterator[list[int]]: ...

def gen_struct(S: set[int], N: Iterable[int], max_span_struct: int=12) -> Iterator[tuple[int,...]]: ...
```

### 7.2 Tratamiento y firmas (`core/encoding.py` y `features/`)

*(Ver firmas completas en §4.1)*

### 7.3 Caché local conmutable (`store/`)

```python
class ChordStore:
    def get_or_add_abs(self, midi_list: list[int], meta: dict) -> int: ...
    def has_abs_mask(self, mask: int) -> bool: ...
    def get_by_struct_id(self, sid) -> list[int]: ...
```

Backends: `memory`, `parquet` (pyarrow), `sqlite` (índice único en `abs_mask_bigint`).

### 7.4 Filtros (`filters/`)

```python
def passes_span(midi_list: list[int], max_span: int) -> bool: ...
def passes_pc_requirements(midi_list: list[int], must_have: set[int], must_avoid: set[int]) -> bool: ...
def matches_interval_pattern(midi_list: list[int], pattern: tuple[int,...]) -> bool: ...
```

---

## 8) Políticas exactas y casos límite

1. **Sin unísonos**: combinaciones sin reposición; si llegan duplicados MIDI externos, se filtran.
2. **Rango de octavas**: cerrado \([o_{min},o_{max}]\), con `edge_pc0=True` para incluir pc=0 en \(o_{max}+1\).
3. **Reflexión**: puede salir de \(S\) cuando el modo esté activo; se registra en `origin`.
4. **PLR**: sólo tríadas mayor/menor en estructural.
5. **Span estructural**: `max_span_struct ∈ {12,24}`.
6. **Entrada del músico**: siempre en PCs y rangos; el motor combina en **MIDI** y proyecta a \(\mathbb{Z}_{12}\) cuando haga falta.

---

## 9) Complejidad y eficiencia

- **Conteo**: \(\sum_{k\in N} \binom{|M|}{k}\).
- **Mitigaciones**: streaming, corte temprano, memorización de `struct_id`, caché local, paralelización por \(k\) o por particiones de \(M\), sampling y límites de lote en GUI.

---

## 10) Casos de uso (para validación)

1. **Díadas estructurales** en cromática: `gen_struct(S=Z12, N={2}, max_span_struct=12)` → 12 clases.
2. **Tríadas diatónicas de Do** en dos octavas: `gen_total(S={0,2,4,5,7,9,11}, O=[4,5], N={3})` + filtro por (4‑3, 3‑4, 3‑3). Detección de inversiones por `struct_id`.
3. **Borde “primera del 5”**: `O=[3,4], edge_pc0=True`.

---

## 11) Plan de pruebas

- **Propiedades**:
  - Inyectividad `abs_mask_bigint`.
  - Invarianza por transposición de `pairwise_dist_hist_mod12` y `struct_id`.
  - Consistencia `pc_mask_of` con `chroma01`.
- **Dorados**: tríadas mayor/menor diatónicas; inversiones con `rotate_bass_up`; reflexión \(H_a\) de tríadas.
- **Escala**: medir tiempo/memoria y tasa de deduplicación en \(|M|∈{24,36,60}, k∈{2,3,4}\).

---

## 12) GUI para el músico — diseño ideal

## ES SOLO UNA SUGERENCIA

### 12.1 Layout

- **Panel Izquierdo — Generación**:

  1. Selector de **alfabeto** (Cromática, Diatónica, Pentatónica, Personalizada) + checkboxes 0–11.
  2. **Rango de octavas**: sliders `o_min`, `o_max`; toggle `edge_pc0`.
  3. **Cardinalidades** \(N\): checkboxes (2,3,4,5, …).
  4. **Filtros tempranos**: `max_span`, `must_have_pcs`, `must_avoid_pcs`, `interval_pattern`.
  5. **Caché**: `none | memory | parquet | sqlite`.
  6. Botones **GENERAR (Total)** / **ESTRUCTURAL**.
  7. Progreso: combinaciones evaluadas, deduplicación, tiempo, memoria.

- **Panel Derecho — Relaciones y Transformaciones**:

  - **Detección**: toggles para `Transposición`, `Rotación R`, `Reflexión H_a` (eje), `PLR` (tríadas), `Parsimonia` (grado voces/semis).
  - **Construcción** (opcional): aplicar R/T/H\_a/PLR a selección y **añadir al mapa**.
  - **Leyenda** de aristas si se muestra grafo.

- **Centro — Lienzo de Exploración**:

  - Scatter 2D (embedding elegido: UMAP/MDS/Laplacian/GeoMDS) QUIZA OTROS con **inserción incremental** (usar `.transform` o Nyström ya implementado).
  - Herramientas: selección por lazo, resalte por `struct_id`, color por `pc_mask`, tamaño por `span`.
  - Atajos: click=selección, shift+click=multi, alt+hover=vista previa del voicing.

- **Inferior — Inspector**:

  - Datos del acorde seleccionado: `midi_list`, `pc_tuple`, `chroma01`, `adjacent_intervals_semitones`, `pairwise_dist_hist_mod12`, `span`.
  - Acciones rápidas: `R` (rotate\_bass\_up), `T±1`, `H_a`.
  - Exportar CSV/JSON.

### 12.2 Rendimiento UX

- Límite de puntos por inserción (p. ej., 50k) y **cola de trabajos** con opciones de muestreo.
- Indicadores de calidad del embedding (trust, stress, continuity).

---

##

---

## 14) Pendientes experimentales (no bloqueantes)

- Punto de cruce `cache=memory` vs `cache=sqlite` en sesiones repetidas.
- Beneficio de `pairwise_dist_hist_semitones` de longitud fija para ML.
- Parsimonia: fijar grado por defecto (p. ej., \(\le 2\) voces, \(\le 2\) semitonos/voz) y coste.

---

##

---

###

