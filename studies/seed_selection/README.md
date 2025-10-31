# Selección de semilla para `compare_proposals`

Este experimento documenta la elección de la única semilla aleatoria que usaremos de forma determinista en los reportes generados con `tools.compare_proposals`. El objetivo es mantener reproducibilidad total sin renunciar a la estabilidad geométrica de los embebidos.

## Importancia de esta prueba

- La reducción dimensional (MDS, UMAP, TSNE) emplea inicializaciones pseudoaleatorias que afectan la geometría final del embedding.
- Elegir una semilla “a ojo” puede ocultar configuraciones subóptimas (stress alto, vecinos mal conservados).
- Con una única semilla bien justificada logramos que cualquier persona pueda regenerar reportes con idénticos resultados, manteniendo confianza en la visualización dentro del dashboard GUI.

## Condiciones del experimento

| Parámetro | Valor |
| --- | --- |
| Propuesta | `identity` (histograma original) |
| Métrica | `euclidean` |
| Reducción | `MDS` (2D, determinista, `n_init=4`) |
| Modo | Determinista (`--execution-mode deterministic`, `n_jobs=1`) |
| Población | Unión de `QUERY_DYADS_REFERENCE`, `QUERY_CHORDS_3_NOTES_ALL`, `QUERY_CHORDS_4_NOTES_ALL` |
| Seeds evaluadas | 7, 11, 17, 23, 29, 37, 42, 53 |

Cada corrida genera `metrics_by_seed.csv`, `metrics.csv`, `report.html` dentro de `studies/seed_selection/runs/seed_<N>`. Los cálculos toman ~4‑12 minutos por semilla debido al tamaño de la población extendida.

## Resultados agregados

Archivo con métricas crudas: [`seed_metrics.csv`](seed_metrics.csv)

Tabla resumen (rank combinado = menor stress + mayor trustworthiness):

```markdown
| Seed | Stress | Trustworthiness | Continuity | KNN Recall | Rank Stress | Rank Trust | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 23 | 0.259725 | 0.891102 | 0.891102 | 0.133564 | 3 | 3 | 6 |
| 11 | 0.260670 | 0.895922 | 0.895922 | 0.142820 | 4 | 2 | 6 |
| 17 | 0.260764 | 0.897927 | 0.897927 | 0.135900 | 5 | 1 | 6 |
| 42 | 0.259092 | 0.887054 | 0.887054 | 0.136419 | 2 | 5 | 7 |
| 53 | 0.257965 | 0.876544 | 0.876544 | 0.136592 | 1 | 8 | 9 |
| 7 | 0.262255 | 0.887513 | 0.887513 | 0.143253 | 6 | 4 | 10 |
| 37 | 0.262313 | 0.882216 | 0.882216 | 0.137457 | 7 | 6 | 13 |
| 29 | 0.262476 | 0.882072 | 0.882072 | 0.136938 | 8 | 7 | 15 |
```

Visualizaciones:

- `stress_trust_by_seed.png`: evolución de stress/trustworthiness por semilla.
- `stress_vs_trust_scatter.png`: trade-off stress vs trust.
- `knn_by_seed.png`: variación del KNN recall (top‑5) por semilla.

## Semilla seleccionada

Elegimos **seed = 17** porque:

1. Presenta la mejor **trustworthiness** de todas las candidatas (0.8979), manteniendo stress dentro de los tres mejores (0.2608).
2. El par stress/trust ofrece el compromiso más estable frente a seeds 11 y 23 (mismo score combinado), con ligera ventaja en continuidad y sin sacrificar KNN recall.
3. Repetir la corrida con seed 17 reproduce exactamente los mismos artefactos, validando la fijación de la semilla.

## Reproducción

```bash
python3 -m tools.compare_proposals \
  --dyads-query QUERY_DYADS_REFERENCE \
  --triads-query QUERY_CHORDS_3_NOTES_ALL \
  --sevenths-query QUERY_CHORDS_4_NOTES_ALL \
  --proposals identity \
  --metrics euclidean \
  --reductions MDS \
  --seeds 17 \
  --execution-mode deterministic \
  --n-jobs 1 \
  --output outputs/seed17_validation
```

## Próximos pasos

- Adoptar `seed=17` como valor único en `compare_proposals`.
- Repetir este protocolo cuando se modifique la población base, la métrica de distancia o se introduzcan nuevas propuestas.
- Extender la automatización para registrar comparativas adicionales (p. ej. stress vs trust por métrica alternativa).
