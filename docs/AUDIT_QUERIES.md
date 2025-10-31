# Auditoría de consultas SQL

## 1. Propósito y alcance
Esta auditoría describe el estado actual de las consultas SQL utilizadas por el GUI Experiment Launcher y las utilidades asociadas (`compare_proposals`, `experiment_inversions`, etc.). El objetivo es:

- Identificar todas las rutas de acceso a datos (constantes en `config.py`, consultas personalizadas, plantillas dinámicas generadas por la GUI).
- Detectar patrones de rendimiento problemáticos (por ejemplo `ORDER BY RANDOM()` y casts costosos sobre `notes[1]`).
- Documentar los pasos necesarios para ejecutar la auditoría automatizada (`tools/audit_queries.py`) y dejar una base para futuras optimizaciones (factorización por plantillas, filtros paramétricos).

La revisión forma parte del plan para convertir el repositorio en un dashboard navegable y reproducible para músicos, con una capa de datos coherente y determinista.

## 2. Herramienta de auditoría (`tools/audit_queries.py`)
Se añadió el script `tools/audit_queries.py`, que ofrece:

- Descubrimiento de consultas registradas en `config.py` y `tools/custom_queries.json` (vía `tools.query_registry`).
- Catalogación de plantillas dinámicas empleadas por la GUI (selección por IDs, `abs_mask_int`, etc.).
- Detección estática de patrones riesgosos (`ORDER BY RANDOM()`, `('x'||notes[1])::bit(32)::int`, `SELECT *`).  
- Ejecución de `EXPLAIN`/`EXPLAIN ANALYZE` y muestreo controlado (cuando existan dependencias y base de datos disponible).
- Exportación de resultados en CSV/JSON (`outputs/audit_queries/<timestamp>/`).

Uso recomendado:

```bash
# Listado estático (sin ejecutar SQL)
python -m tools.audit_queries --dry-run --output-dir outputs/audit_queries/dryrun

# Auditoría completa (requiere dependencias de DB, numpy/pandas y acceso a Postgres)
python -m tools.audit_queries --limit 150 --with-analyze --output-dir outputs/audit_queries/$(date +%Y%m%d_%H%M%S)
```

> **Dependencias**: para la ejecución completa se necesitan `numpy`, `pandas` y un proveedor `QueryExecutor` (`chordcodex` o el fallback `synth_tools`). En este entorno solo se ejecutó el modo `--dry-run` porque faltan `numpy/pandas`.

## 3. Inventario de consultas
Se registraron 30 consultas/plantillas:

| Categoría | Conteo | Descripción |
|-----------|--------|-------------|
| `config`  | 25     | Constantes `QUERY_*` usadas por GUI/CLI. |
| `custom`  | 2      | Definidas en `tools/custom_queries.json`. |
| `template` | 4     | Plantillas dinámicas generadas por código (`SELECT * WHERE id IN (…)`, `abs_mask_int = ANY(%s)`, etc.). |

Los artefactos del `dry-run` se guardaron en `outputs/audit_queries/dryrun/audit_results.{csv,json}`.

## 4. Hallazgos clave (modo `--dry-run`)
Aunque no se ejecutó la base de datos, el análisis estático ya revela deuda importante:

1. **Uso masivo de `ORDER BY RANDOM()`**  
   - 17 consultas lo emplean (todas las muestras `%_SAMPLE_%`, poblaciones mixtas y queries personalizadas).  
   - Impacto: provoca un sort global sobre millones de filas en PostgreSQL.  
   - Recomendación: sustituir por `TABLESAMPLE`, criterios deterministas (`WHERE random() < p` ordenando por `id`), o prefiltros basados en índices.

2. **`SELECT *` en rutas centradas en GUI/CLI**  
   - 8 consultas (incluidas plantillas dinámicas).  
   - Impacto: ancho de fila innecesario, reduce cache y encarece transporte.  
   - Recomendación: proyectar solo columnas necesarias (p. ej. `id, n, interval, notes, tag`) y permitir que la GUI pida columnas adicionales bajo demanda.

3. **Cast `('x'||notes[1])::bit(32)::int`**  
   - Presente en 7 consultas.  
  - Impacto: invalida índices sobre `notes`, provoca `Seq Scan` y costes exponenciales.  
  - Recomendación: añadir columna persistida `root_note INT GENERATED ALWAYS AS ... STORED` o derivar raíz en una CTE con conversión a `INT` una sola vez, acompañada de índice.

4. **Plantillas no ejecutables automáticamente**  
   - Las consultas `INLINE_*` requieren parámetros (`id list`, `ANY(%s)`). Se catalogaron para documentar su uso, pero se deben auditar con datos reales (familias de acordes, selecciones manuales) al integrar la capa de datos.

5. **Cobertura de GUI**  
   - Todas las consultas que aparecen en la lista desplegable del launcher provienen de `config.py` o `custom_queries.json`. No se detectaron consultas “fantasma” en el código.  

## 5. Limitaciones de esta corrida
- Falta de dependencias (`numpy`, `pandas`) impidió ejecutar la auditoría completa (no hay `EXPLAIN`/`ANALYZE` ni muestreo real).  
- No se validó conformidad de tipos, valores `NULL` o joins inesperados.  
- Las plantillas dinámicas no tienen parámetros de prueba; se requiere la futura “capa de adquisición de datos” para alimentarlas con muestras controladas.

## 6. Próximos pasos y oportunidades
1. **Habilitar entorno completo**: instalar dependencias (`pip install numpy pandas chordcodex`) y ejecutar `tools.audit_queries` contra la base `ChordCodex`.  
2. **Factorizar plantillas parametrizables**: crear un “builder” de consultas (p. ej. `build_chords_query(tags=…, cardinality=…)`) que permita filtrar acordes desde la GUI sin duplicar SQL.  
3. **Refactor de muestreos**: reemplazar `ORDER BY RANDOM()` por estrategias deterministas y cacheables.  
4. **Columna `root_note` persistida**: discutir con el equipo de base de datos la creación de la columna y el índice correspondiente.  
5. **Proyección mínima de columnas**: revisar cada consulta `SELECT *` según necesidad de la interfaz; idealmente ofrecer “perfiles” (reproducción, visualización, análisis).  
6. **Test automático**: extender el script para ejecutarse en CI (modo `--dry-run`) y fallar si aparecen nuevas consultas con patrones prohibidos.  
7. **Integrar capa de adquisición**: la futura API deberá centralizar parámetros (cardinalidad, tags, familias) y heredar las optimizaciones reseñadas aquí.

## 7. Código relacionado
- `tools/audit_queries.py`: script principal de auditoría.
- `tools/query_registry.py`: catálogo de constantes SQL + consultas personalizadas.
- `tools/gui_experiment_launcher.py`: construye las plantillas dinámicas (`INLINE_*`).
- Archivos de resultados: `outputs/audit_queries/dryrun/audit_results.csv` y `.json`.

Con este informe queda documentada la deuda técnica en la capa de consultas y se deja una guía reproducible para avanzar hacia un dashboard más eficiente y flexible para el usuario músico.
