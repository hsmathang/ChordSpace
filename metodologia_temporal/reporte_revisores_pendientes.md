## Estado actual y retos pendientes (enfoque Q1/Q2)

### Avances logrados
- Contratos validados para el reporte: meta de figuras (`validate_scatter_payload_meta`) y `run_metadata` (descriptores combinatorial/DB) con pruebas.
- Orquestación separada: `tools/proposals_pipeline.experiment` y `runner` integrados en `compare_proposals.py`.
- Pruebas de humo y contratos (11 tests) cubriendo distancias, color, runner, metadatos y render de report.html.
- Compatibilidad de consola garantizada (logs ASCII) y bloqueos por Unicode resueltos.

### Retos pendientes
1) **Reproducibilidad end-to-end**  
   - Falta un smoke CLI/GUI real (población mínima + MDS/cosine) que genere `report.html` y verifique su estructura.  
   - Documentar seeds, versión de DB/dataset y comando exacto para replicar.
2) **Metadatos completos del reporte**  
   - Extender `validate_run_metadata` si la GUI añade campos nuevos.  
   - Asegurar en tests que la sección "Configuración de la población" se renderiza con fuentes combinatorial y DB reales (hoy se valida con datos sintéticos).
3) **Higiene de código**  
   - Pasar lint/orden de imports final en `compare_proposals.py` y módulos tocados.  
   - Eliminar artefactos temporales (`tmp_cell*.txt`, `pre*.tmp`) sin afectar reproducibilidad.
4) **Documentación metodológica**  
   - Actualizar `docs/reporting_pipeline.md` con la orquestación actual (runner) y ejemplos de `run_metadata` combinatorial/DB.  
   - Añadir guía de interpretación de metadatos y cómo leer las tablas del reporte.
5) **Pruebas de rendimiento/escala**  
   - No hay benchmark ni pruebas en datasets grandes; sugerido un smoke “grande” opcional (solo para entorno local) para detectar regresiones de tiempo/CPU.

### Propuesta inmediata
- Implementar smoke CLI/GUI pequeño (triadas de muestra, MDS+cosine) y fijar en CI/local.  
- Lint/imports finales y limpieza de archivos temporales.  
- Extender validación de `run_metadata` al conjunto completo de campos usados por la GUI y reflejarlo en un test end-to-end ligero.
