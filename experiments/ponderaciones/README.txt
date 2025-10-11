Objetivo (pendiente de aprobación de ponderaciones)
- Esta carpeta agrupa experimentos para estudiar cómo diferentes esquemas de ponderación
  (p. ej., importancia perceptual, consonancia, combinaciones) afectan los vectores
  de rugosidad y las comparaciones entre acordes/inversiones/voicings.

Qué se espera que haya aquí (cuando las ponderaciones estén aprobadas):
- notebooks/: Notebooks exploratorios con resultados SIN outputs guardados (usar nbstripout si es posible).
- scripts/: Scripts Python reproducibles (sin depender de DB) que tomen un set pequeño de acordes
  y produzcan artefactos en outputs/ (npy/csv/png/html) para comparación entre esquemas.
- datasets/: (opcional) CSV mínimos/estáticos con acordes de prueba (si se usan, sin datos sensibles).
- reportes/: (opcional) resúmenes en texto/markdown con hallazgos y parámetros usados.

Alcance de los experimentos (ideas iniciales):
- Efecto de inversión/orden en presencia de ponderaciones (p. ej., bajo/final/decay) vs. sin ponderación.
- Comparación cualitativa entre diferentes pesos y su impacto en métricas (
  continuidad/trustworthiness, stress, correlación de rangos, etc.)
- Sensibilidad a variaciones de parámetros (n_armonicos, decaimiento, base_freq).

Buenas prácticas:
- No guardar outputs pesados; si es necesario, mantenerlos fuera del repo (outputs/ está ignorado).
- No versionar datos sensibles (.env, credenciales, etc.).
- Documentar parámetros, seed y versiones en un TXT/MD por experimento para facilitar revisión.

Estado actual:
- Las ponderaciones NO están definidas como “finales”. Este directorio es un andamiaje para
  incorporar los experimentos cuando las ponderaciones sean aprobadas.
