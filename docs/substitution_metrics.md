# Arquitectura de Métricas de Sustitución Armónica en ChordSpace

Este documento propone una arquitectura modular para detectar y explicar sustituciones armónicas entre acordes a partir de:

- Medidas sensoriales (psicoacústicas) sobre vectores de rugosidad de Sethares.
- Medidas estructurales sobre clases de pitch e intervalos.
- Costes de conducción de voces (voice‑leading) aproximados.
- Rasgos tonales/funcionales (centro tonal, tritono, intercambio modal).

La propuesta se alinea con el pipeline actual del repositorio y detalla qué funciones nuevas hay que construir, con un plan ordenado desde lo inmediatamente viable hasta líneas futuras más ambiciosas.

## 1. Contexto actual (base disponible en el repo)

- Cálculo de vectores de rugosidad y totales por acorde (Sethares 12‑D) durante el preproceso.
  - Fuente: `ChordEntry.hist` y valores asociados en `tools/compare_proposals.py:429` y `pre_process.py`.
- Matrices de distancia y reducciones 2D (MDS/UMAP/K-MDS) para visualización y evaluación.
  - Fuentes: `reduction.py`, `metrics.py` (trustworthiness, continuity, stress, etc.).
- Reporte HTML con `customdata` enriquecido por punto.
  - Fuente: `visualisations/proposals.py:349` (construcción de payload) y `tools/compare_proposals.py:2390+` (interacciones JS).

Estos cimientos permiten construir, sin rehacer el pipeline, un módulo de métricas y vecinos (k‑NN) sobre rasgos ya presentes y rasgos estructurales simples derivados de `notes_abs_json`.

## 2. Notación y objetos

Sea un acorde \(C\) con notas absolutas (MIDI) `notes_abs = (n_1, …, n_m)` ordenadas de grave a agudo.

- Conjunto de pitch classes: \(\mathrm{PC}(C) = \{ n_i \bmod 12 \}\).
- Vector binario de PC de dimensión 12: \(\mathbf{b}_C \in \{0,1\}^{12}\) con \(\mathbf{b}_C[k]=1\iff k\in\mathrm{PC}(C)\).
- Vector de clases de intervalo (IC): \(\mathbf{IC}_C \in \mathbb{N}^6\), donde \(\mathbf{IC}_C[j]\) cuenta pares con intervalo \(j\in\{1,\dots,6\}\) mod 12 sin dirección.
- Histograma de rugosidad (Sethares) de 12 componentes: \(\mathbf{h}_C\in\mathbb{R}_{\ge 0}^{12}\). Versión probabilística \(\mathbf{p}_C = \mathbf{h}_C/\|\mathbf{h}_C\|_1\).
- Distancia circular en semitonos: \(d_{12}(x,y) = \min_{k\in\mathbb{Z}} |x - y + 12k|\).

## 3. Métricas atómicas (definiciones formales)

Se definen disimilaridades \(D\in[0,\infty)\) donde menor es “más similar”.

### 3.1 Sensorial (psicoacústica)

- Jensen–Shannon (JSD) entre \(\mathbf{p}_C\) y \(\mathbf{p}_D\):
  \[
  \operatorname{JSD}(\mathbf{p},\mathbf{q}) = \tfrac12 \operatorname{KL}(\mathbf{p}\,\|\,\mathbf{m}) + \tfrac12 \operatorname{KL}(\mathbf{q}\,\|\,\mathbf{m}),\quad \mathbf{m}=\tfrac12(\mathbf{p}+\mathbf{q}).
  \]
  Usamos \(D_\mathrm{JSD} = \sqrt{\operatorname{JSD}}\) para mantener métrica.

- Coseno (sobre hist o vector ajustado):
  \(D_\mathrm{cos}(\mathbf{x},\mathbf{y}) = 1 - \frac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\,\|\mathbf{y}\|}\).

### 3.2 Estructural (set/intervalo)

- Jaccard sobre PC‑set: \(D_\mathrm{Jac}(A,B)=1-\frac{|A\cap B|}{|A\cup B|}.\)
- L1 sobre IC: \(D_{\mathrm{IC}}(\mathbf{IC}_C,\mathbf{IC}_D)=\|\mathbf{IC}_C-\mathbf{IC}_D\|_1.\)
- Equivalencia Tn/TnI opcional (no métrica, regla de filtrado/penalización).

### 3.3 Conducción de voces (voice‑leading) aproximada

Sea \(C\) de \(m\) notas y \(D\) de \(m\) notas. Definimos un coste OPT simplificado:
\[
D_{\mathrm{VL}}(C,D) = \min_{t\in\{0,\dots,11\}}\;\min_{\sigma\in S_m}\;\frac{1}{m}\sum_{i=1}^m d_{12}\big((n^{(D)}_{\sigma(i)}+t)\bmod 12,\; n^{(C)}_i\bmod 12\big).
\]
Para cardinalidades distintas, se usa matching parcial (Hungarian sobre matriz de costes) y/o penalización por diferencia de cardinalidad.

### 3.4 Tonal/Funcional

- Centro tonal (Tonal Centroid, 6‑D) \(\mathbf{z}_C\) a partir de \(\mathbf{b}_C\) siguiendo Harte–Sandler (o Tonnetz extendido). Distancia \(\|\mathbf{z}_C-\mathbf{z}_D\|_2\).
- Detectores lógicos: tritono compartido, intercambio modal, upper‑structures (no métricas, reglas binarias con explicación).

## 4. Similaridad compuesta

Definimos una combinación ponderada (menor es más similar):
\[
D_{\boldsymbol{w}}(C,D)\;=\; w_1 D_{\mathrm{JSD}}\;+\; w_2 D_{\mathrm{cos}}\;+\; w_3 D_{\mathrm{Jac}}\;+\; w_4 D_{\mathrm{IC}}\;+\; w_5 D_{\mathrm{VL}}\;+\; w_6\,\|\mathbf{z}_C-\mathbf{z}_D\|_2.
\]
Ponderaciones \(\boldsymbol{w}\) dependientes de perfil (sensorial, funcional, color) y normalización por z‑scores por métrica para hacer comparables los rangos.

## 5. Diseño por módulos (nuevo código a construir)

Se propone una carpeta `substitution/` con los siguientes componentes.

### 5.1 `substitution/features.py`

Funciones puras (vectorizadas cuando sea posible) y una caché ligera.

```python
from dataclasses import dataclass
from typing import Sequence, Dict, Any
import numpy as np

def pcset(notes_abs: Sequence[int]) -> set[int]: ...
def pcvec(notes_abs: Sequence[int]) -> np.ndarray:  # (12,) binario
    ...
def interval_class_vector(notes_abs: Sequence[int]) -> np.ndarray:  # (6,)
    ...
def roughness_prob(hist: np.ndarray) -> np.ndarray:  # normaliza a prob
    ...
def tonal_centroid(pc_vec: np.ndarray) -> np.ndarray:  # (6,)
    ...  # implementar fórmula Harte–Sandler/Tonnetz

def voice_leading_cost(
    notes_a: Sequence[int],
    notes_b: Sequence[int],
    allow_transpose: bool = True,
    allow_inversion: bool = False,
    use_hungarian: bool = True,
) -> float: ...

@dataclass
class FeatureStore:
    # id -> dict con pcvec, ic, p_hist, tonal_centroid, notes_abs, etc.
    cache: Dict[Any, Dict[str, Any]]
    def get(self, entry) -> Dict[str, Any]: ...
    def preload(self, entries) -> None: ...  # vectoriza cálculo
```

Notas de implementación:

- `pcvec` y `interval_class_vector` se derivan de `notes_abs_json` ya disponible.
- `roughness_prob` parte de `entry.hist` (ya calculado en el pipeline).
- `voice_leading_cost` puede implementarse con Hungarian sobre la matriz \(m\times n\) de costes `d12` (con relleno para cardinalidades distintas) y un barrido pequeño de transposiciones \(t\in[0,11]\).

### 5.2 `substitution/metrics.py`

Implementa distancias atómicas (con broadcasting/NumPy cuando sea viable):

```python
import numpy as np

def d_jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float: ...
def d_cos(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float: ...
def d_jaccard_pc(a: np.ndarray, b: np.ndarray) -> float:  # usa pcvec binario
    ...
def d_ic_l1(a: np.ndarray, b: np.ndarray) -> float: ...
def d_vl(notes_a, notes_b) -> float: ...  # wrap de features.voice_leading_cost
def d_tonal_centroid(z1: np.ndarray, z2: np.ndarray) -> float: ...
```

### 5.3 `substitution/aggregate.py`

Construye \(D_{\boldsymbol{w}}\) y normaliza por perfiles.

```python
from typing import Dict

DEFAULT_WEIGHTS = {
    "jsd": 0.35, "cos": 0.15, "jaccard": 0.2, "ic_l1": 0.15, "vl": 0.15, "tonal": 0.0,
}

def composite_distance(entry_a, entry_b, feats, weights: Dict[str, float] = None) -> float:
    ...
```

### 5.4 `substitution/index.py`

Índice de vecinos y consultas filtradas.

```python
from typing import Iterable, Dict, Any
import numpy as np

def knn(entries: Iterable, feats, weights, k: int = 8,
        same_cardinality: bool = True, max_reg_diff: int | None = None) -> Dict[Any, list]:
    """Devuelve, para cada id, lista de (neighbor_id, dist, breakdown)."""
    ...
```

Versión ANN futura: wrapper para Annoy/FAISS si \(N\) crece.

### 5.5 `substitution/detectors.py`

Reglas explicables (señales binarias con detalles de por qué coincide):

```python
def detect_common_tone(pc_a, pc_b, min_common: int = 2) -> dict: ...
def detect_tritone(pc_a, pc_b) -> dict: ...  # dominante / tritono compartido
def detect_modal_interchange(pc_a, key_ctx=None) -> dict: ...
def detect_upper_structure(pc_a, pc_b) -> dict: ...
```

### 5.6 Integración en el reporte (fase 2)

- Extender `visualisations/proposals.py` para añadir un panel “Vecinos por perfil” (Top‑K) por cada punto, usando el índice calculado offline y añadiendo los ids de vecinos a `customdata` del punto.
- En JS, dibujar líneas temporales hacia los vecinos cuando el punto se señala, similar a los enlaces de inversiones ya implementados (`tools/compare_proposals.py:2390+`).

## 6. Perfiles de uso (pesos y reglas)

1) Perfil “Básico” (MVP) — implementable ya:

- \(D_{\boldsymbol{w}}\) con `jsd`, `cos`, `jaccard`, `ic_l1` y `vl` opcional como desempate.
- Filtros: misma cardinalidad; tolerancia de registro (span de semitonos similar).
- Salida: Top‑K vecinos + desglose por métrica.

2) Perfil “Funcional” (próximo):

- Añadir `tonal_centroid` y detectores (tritono, intercambio modal), con etiquetas explicativas.

3) Perfil “Color/Texto” (siguiente):

- Menor peso en `vl`, mayor en `jsd` y `ic_l1`; permitir distintas cardinalidades.

4) Perfil “Aprendido” (futuro):

- Ajustar \(\boldsymbol{w}\) por aprendizaje supervisado con pares curados de “buenas/malas” sustituciones.

## 7. Plan de trabajo (ordenado por viabilidad)

1) MVP Métrico + k‑NN (rápido)

- Implementar `features.py` (pcvec, ic, roughness_prob, voice_leading_cost simple).
- Implementar `metrics.py` (jsd, cos, jaccard, ic_l1, vl wrapper).
- Implementar `aggregate.py` (composite + breakdown) y `index.py` (k‑NN exacto con vectorización y filtros básicos).
- Exportar resultados a JSON por escenario (ej.: `outputs/.../neighbors.json`).
- Integrar un panel simple en el reporte que lea y muestre Top‑K con desglose (sin líneas aún, opcional).

2) Detectores funcionales y líneas de vecinos (corto plazo)

- Añadir `detectors.py` (tritono, common‑tone, modal interchange, upper‑structure).
- Añadir overlay de líneas (reutilizar lógica de inversiones) y tooltips con reglas activas.

3) Tonal centroid y perfiles (medio plazo)

- Implementar `tonal_centroid` y sumar la métrica al compuesto; perfilar pesos por cardinalidad.

4) ANN y datasets grandes (medio plazo)

- Backend con Annoy/FAISS, serialización del índice y carga en reporte.

5) Aprendizaje de pesos (largo plazo)

- Recoger data curada; loss ranking (triplet loss) o regresión de distancias objetivo.

## 8. Complejidad y rendimiento

- Cálculo de features: \(\mathcal{O}(N\cdot m^2)\) por IC y \(\mathcal{O}(N\cdot 12)\) por PC/roughness; muy asequible para miles de acordes.
- VL aproximado: \(\mathcal{O}(11\cdot m^3)\) con Hungarian; para triadas y cuatríadas es inmediato.
- k‑NN exacto: \(\mathcal{O}(N^2)\) si se calcula pairwise una vez; reutilizable por escenario.
- ANN: \(\mathcal{O}(N\log N)\) construcción; \(\mathcal{O}(\log N)\) por consulta.

## 9. Interfaz de datos y formato de salida

Estructura JSON por escenario (por id global del punto, siguiendo `customdata[7]`):

```json
{
  "neighbors": {
    "<id>": [
      {
        "neighbor": <id2>,
        "distance": 0.213,
        "breakdown": {"jsd": 0.08, "cos": 0.04, "jaccard": 0.05, "ic_l1": 0.02},
        "rules": ["common_tone>=2", "triad_ic_close"]
      }
    ]
  },
  "meta": {"weights": {"jsd": 0.35, ...}}
}
```

## 10. Riesgos y decisiones

- Normalización de escalas: usar z‑scores por métrica y por población/escenario.
- Invariancias: decidir si colapsar Tn/TnI según perfil o usar como filtro.
- Contexto: cuando no hay tonalidad global confiable, los detectores funcionales se limitan a reglas locales (tritono común, etc.).

## 11. Ejemplo de integración en el reporte (wireframe)

- “Vecinos (Perfil básico)”: lista Top‑K en el panel derecho, con distancias y etiquetas de reglas; al pasar el mouse por un vecino, dibujar una línea desde el acorde actual hasta ese vecino y resaltar ambos.

## 12. Qué existe vs. qué falta

Disponible hoy:

- Hist de rugosidad por acorde y `notes_abs_json` (para PC/IC/VL).
- Infraestructura de reporte con `customdata` e interacciones.
- Evaluaciones/embeddings y exportadores HTML.

Falta construir (MVP):

- `substitution/features.py`: `pcvec`, `interval_class_vector`, `roughness_prob`, `voice_leading_cost`.
- `substitution/metrics.py`: `d_jsd`, `d_cos`, `d_jaccard_pc`, `d_ic_l1`, `d_vl`.
- `substitution/aggregate.py`: `composite_distance` (+ breakdown).
- `substitution/index.py`: `knn` con filtros.
- Exportador JSON + lector sencillo en el reporte.

Siguientes (corto/medio plazo):

- `detectors.py` (tritono, common‑tone, modal interchange, upper‑structure).
- `features.tonal_centroid` y métrica tonal.
- Overlays de líneas hacia vecinos en el scatter.

Futuro (largo plazo):

- Índices ANN y ajuste de pesos por aprendizaje supervisado.

## 13. Idea más rápida para ejecutar ahora mismo (MVP)

- Implementar el Perfil “Básico” con 4 métricas (`jsd`, `cos`, `jaccard`, `ic_l1`) y `knn` exacto, generando un JSON de vecinos por escenario. Integrar panel de lista (sin líneas) en el reporte.
- Esfuerzo estimado: 1–2 jornadas, 300–450 LOC.

---

Autor: Equipo ChordSpace — Propuesta técnica inicial para sustitución armónica basada en métricas.

