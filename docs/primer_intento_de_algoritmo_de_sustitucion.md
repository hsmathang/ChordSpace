# Primer intento de algoritmo de sustitución

Este documento describe la arquitectura implementada para el primer MVP del algoritmo de sustitución armónica en ChordSpace.

- Se apoya en `visualisations/proposals.py` para calcular métricas y vecinos.
- Se visualiza y controla desde `tools/compare_proposals.py`.
- Amplía el modelo descrito en `docs/substitution_metrics.md`.

## Objetivos

1. Medir similitud entre acordes usando métricas disponibles sin modificar el pipeline de preprocesamiento.
2. Exponer los vecinos más cercanos (sustitutos) en `report.html`.
3. Permitir un modo interactivo de resaltado similar al de inversiones.
4. Registrar la información en `meta` para poder reutilizarla en el front-end.

## Componentes

### 1. Extracción de rasgos (`visualisations/proposals.py`)

Para cada `ChordEntry`:

- Histograma de rugosidad normalizado (vector 12‑D):
  \[
  \mathbf{p}_i = \frac{\mathbf{h}_i}{\|\mathbf{h}_i\|_1 + \varepsilon}
  \]
- Vector binario de pitch classes (12‑D):
  \[
  \mathbf{b}_i[k] =
  \begin{cases}
  1, & k \in \{n \bmod 12 : n \in \text{notes\_abs}_i \} \\
  0, & \text{en otro caso}
  \end{cases}
  \]
- Cardinalidad \(c_i = n\_notes\).

### 2. Métricas básicas

Dos disimilaridades se calculan para cada par \((i,j)\):

1. Jensen–Shannon:
   \[
   D_{\text{JSD}}(i,j) = \sqrt{ \tfrac12 \mathrm{KL}(\mathbf{p}_i\|\mathbf{m}_{ij}) + \tfrac12 \mathrm{KL}(\mathbf{p}_j\|\mathbf{m}_{ij}) }
   \quad\text{con}\quad \mathbf{m}_{ij} = \tfrac12 (\mathbf{p}_i + \mathbf{p}_j)
   \]
2. Jaccard sobre PC-set:
   \[
   D_{\text{Jac}}(i,j) = 1 - \frac{|\mathbf{b}_i \land \mathbf{b}_j|_1}{|\mathbf{b}_i \lor \mathbf{b}_j|_1}
   \]

La distancia combinada es:
\[
D(i,j) = 0.6 \cdot D_{\text{JSD}}(i,j) + 0.4 \cdot D_{\text{Jac}}(i,j)
\]

### 3. Vecinos Top-K

Para cada acorde \(i\):

```python
same_card = [j for j in range(N) if j != i and c_j == c_i]
dists = sorted((D(i, j), j) for j in same_card)
neighbors[i] = dists[:K]  # K=8
```

Se guarda en `meta["substitutionNeighbors"]` con la estructura:

```json
{
  "42": [
    {
      "neighbor": 17,
      "distance": 0.214,
      "components": {"jsd": 0.115, "jaccard": 0.099}
    }
  ]
}
```

### 4. Interfaz (`tools/compare_proposals.py`)

1. **Control**: se añade un checkbox `Resaltar sustituciones` junto a los toggles de inversiones.
2. **Resaltado**:
   - `setupSubstitutionHighlight` lee `layout.meta.substitutionNeighbors`.
   - En `plotly_hover`, si el toggle está activo, se crea un `Set` con el acorde y sus vecinos.
   - `applyGlobalIdHighlight` atenúa el resto (`opacity *= 0.1`).
3. **Panel de detalle**:
   - `registerCardDetail` añade una sección “Sustitutos sugeridos” que muestra `label` + `dist`.
   - Usa `lookupLabelById` para reutilizar el texto que ya se muestra en el scatter.

### 5. Pseudocódigo resumido

```python
# visualisations/proposals.py
for entry in entries:
    p = normalize_hist(entry.hist)
    b = build_pc_vector(entry.acorde.notes_abs)
    hist_list.append(p); pc_list.append(b); card_list.append(entry.n_notes)

for i in range(N):
    for j in range(i+1, N):
        jsd[i,j] = js_distance(hist_list[i], hist_list[j])
        jac[i,j] = jaccard(pc_list[i], pc_list[j])

dist = 0.6*jsd + 0.4*jac

for i in range(N):
    neighbors = [(dist[i,j], j) for j if card[j]==card[i]]
    meta["substitutionNeighbors"][id_i] = serialize_top(neighbors, K=8)
```

```javascript
// tools/compare_proposals.py
function setupSubstitutionHighlight(gd) {
  const neighbors = gd.layout.meta.substitutionNeighbors;
  const toggle = card.querySelector('.substitution-toggle');
  let lastHover = null;

  function applyFor(id) {
    if (!toggle.checked) return applyGlobalIdHighlight(gd, null);
    const set = new Set([id, ...neighbors[id].map(n => n.neighbor)]);
    applyGlobalIdHighlight(gd, set, 0.1);
  }

  gd.on('plotly_hover', ev => { lastHover = id; if (toggle.checked) applyFor(id); });
  gd.on('plotly_unhover', () => { lastHover = null; applyGlobalIdHighlight(gd, null); });
  toggle.addEventListener('change', () => {
    if (!toggle.checked) applyGlobalIdHighlight(gd, null);
    else if (Number.isFinite(lastHover)) applyFor(lastHover);
  });
}
```

## Referencias internas

- `visualisations/proposals.py` — sección “Sustituciones” dentro de `build_scatter_payload`.
- `tools/compare_proposals.py`
  - Checkbox en la plantilla HTML.
  - `setupSubstitutionHighlight` (JavaScript).
  - Nuevas ayudas `getBaseMarkerOpacities` / `applyGlobalIdHighlight`.
  - Panel detallado actualizado en `registerCardDetail`.

## Próximos pasos

1. Añadir métricas adicionales (IC‑L1, voice-leading, tonal centroid) usando el mismo pipeline.
2. Mostrar las contribuciones individuales (`components`) en el panel.
3. Reutilizar el overlay de líneas (similar al de inversiones) para visualizar conexiones.
4. Serializar resultados en archivos auxiliares para análisis fuera del reporte.

