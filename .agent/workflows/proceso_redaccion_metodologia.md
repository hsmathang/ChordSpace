# Workflow: Redacción Rigurosa de Metodología en LaTeX

## Descripción
Flujo para redactar el capítulo de metodología de la tesis de maestría en matemáticas aplicadas. Produce LaTeX en calidad publicable con rigor formal, citas integradas y placeholders para figuras/tablas.

**Principio rector**: `ESTRUCTURA_MATEMATICA_DETALLADA.md` es el insumo bruto. El `.tex` es la versión destilada para un jurado doctoral. Cada pieza se clasifica como: incluir, mover a apéndice, o descartar con justificación.

**Audiencia**: Jurado con formación doctoral en matemáticas o áreas afines.

---

## Fase 0: Inventario y Triage

**Objetivo**: Clasificar CADA elemento del MD antes de escribir LaTeX.

### Paso 0.1: Inventario atómico
Recorrer `ESTRUCTURA_MATEMATICA_DETALLADA.md` y producir tabla: `| ID | Tipo | Contenido | Destino | Justificación |`

**Tipos**: Definición, Proposición, Ecuación, Resolución (Q-XXX), Tabla, Ejemplo numérico, Nota del autor, Pregunta clave, Debilidad identificada.

### Paso 0.2: Reglas de clasificación (primera que aplique gana)

1. **→ Cap. Metodología**: definición/ecuación usada después; decisión que un evaluador cuestionaría; contribución original; ejemplo numérico aclaratorio.
2. **→ Apéndice**: derivación larga; detalle de implementación computacional; resolución cuyo argumento ya está sintetizado en el capítulo; tabla de datos extensos.
3. **→ Descartar**: nota de trabajo pendiente; comentario de workflow; pregunta sin resolver sin Q-XXX; contenido duplicado.
4. **→ Otro capítulo**: marco teórico (e.g., banda crítica auditiva); discusión o trabajo futuro.

### Paso 0.3: Validación
- Toda definición usada en ecuaciones posteriores → Cap. Metodología
- Toda "Debilidad a vigilar" tiene respuesta asignada
- Las preguntas del evaluador (Fase 3) están cubiertas
- Ningún elemento sin clasificar

**Entregable**: `INVENTARIO_TRIAGE.md` — presentar al usuario para aprobación.

---

## Fase 1: Esqueleto LaTeX con placeholders

### Paso 1.1: Estructura anotada
Crear `.tex` con secciones y comentarios internos especificando: definiciones, justificaciones, ejemplos, figuras, tablas, citas necesarias, conexiones anterior/siguiente, refs a apéndice.

### Paso 1.2: Placeholders de figuras y tablas

**Figuras obligatorias** (mín. 5):
1. Diagrama de flujo metodológico (pipeline completo)
2. Curva de disonancia Plomp-Levelt/Sethares
3. Ejemplo visual de Φ_raw (barplot 12 bins)
4. Esquema R^12 → reducción → R^2
5. Diagrama de Shepard

**Tablas obligatorias** (mín. 2):
1. Catálogo de 10 normalizaciones (fórmula, interpretación, dominio)
2. Factores del diseño experimental (normalización × métrica × reductor)

Usar `\fbox{\parbox{...}}` con caption definitivo y `\label{}`.

### Paso 1.3: Archivo .bib
Crear/actualizar `referencias_metodologia.bib` con TODAS las citas de resoluciones clasificadas como Cap. Metodología o Apéndice.

**Entregable**: `.tex` esqueleto + `.bib` poblado.

---

## Fase 2: Redacción por tipo de bloque

**No redactar linealmente por sección.** Redactar por tipo de bloque.

### Paso 2.1: DEFINICIONES (todas primero)
Criterios: usa notación del glosario; tiene `\label{def:nombre}`; referencia definiciones previas; cita autor original si es estándar; indica "Se propone..." si es original.

**Verificar orden topológico de dependencias**:
N → f(n) → Z_12 → π → A_m → Δ → ic → d → R → R_total → Φ_raw → Normalizaciones → Métricas → D → MDS/UMAP → Stress, T(k), C(k)

### Paso 2.2: JUSTIFICACIONES
Condensar cada Q-XXX a 1-2 párrafos. Patrón: Afirmación → Evidencia1+cite → Evidencia2+cite → Implicación.

**Justificaciones obligatorias**:
1. Orden estricto sin unísonos (Q-005)
2. Pitch chords vs PC-sets (Q-007, Q-008)
3. 12 bins vs 6 (Q-010)
4. Sethares vs otros modelos (Q-013, Q-014)
5. H=6 armónicos (Q-018)
6. MDS métrico (Q-030)
7. Criterio de falsificación

### Paso 2.3: EJEMPLOS NUMÉRICOS (mín. 3)
1. Post-Def Acorde: Do mayor (60,64,67), Δ=(4,3), span=7
2. Post-Φ_raw: valores numéricos para esa tríada (verificar contra repo)
3. Post-normalizaciones: comparar identity vs simplex

### Paso 2.4: TABLAS Y FIGURAS — completar contenido de tablas, captions definitivos.

### Paso 2.5: PROSA CONECTIVA (al final)
Párrafos de: motivación de sección, transición con anterior, cierre+anticipación, intro y cierre del capítulo.

**Entregable**: `.tex` con contenido completo.

---

## Fase 3: Verificación de completitud y rigor

### Paso 3.1: Cobertura vs inventario
- Cada ítem Cap. Metodología aparece en `.tex`
- Cada ítem Apéndice tiene ref. cruzada desde el capítulo
- Cada ítem Descartar tiene justificación válida

### Paso 3.2: Dependencias de notación
Verificar que cada símbolo fue definido antes de su primer uso:
N, f(n), Z_12, π, A_m, Δ, ic, d, S, R, R_total, Φ_raw, D, ρ, Stress, T(k), C(k)

### Paso 3.3: Citas
- Cada `\cite{}` tiene entrada en `.bib`; cada afirmación empírica tiene `\cite{}`
- Mínimo 35-45 citas; sin dependencia excesiva de una sola fuente

### Paso 3.4: Preguntas del evaluador

| Pregunta del evaluador | Ubicación en LaTeX |
|---|---|
| ¿Por qué no PC-sets / invariancia transposicional? | §3.1.X |
| ¿Por qué 12 bins y no 6? | §3.1.X |
| ¿Por qué Sethares y no Vassilakis/Parncutt? | §3.2.X |
| ¿Por qué H=6 y δ=0.88? | §3.2.X |
| ¿Base perceptual del suavizado gaussiano? | §3.3.X |
| ¿Qué pasa con ceros en JSD? | §3.4.X |
| ¿Por qué MDS métrico? | §3.5.X |
| ¿Stress es suficiente como métrica? | §3.7.X |
| ¿Qué resultado descartaría la hipótesis? | §3.6.X |
| ¿Rugosidad captura "similitud sonora"? | §3.8.X |

Si alguna celda vacía → volver a Fase 2.

### Paso 3.5: Integridad LaTeX
- Toda `\label{}` tiene `\ref{}`; toda `\ref{}` tiene `\label{}`
- Definiciones numeradas consecutivamente; ecuaciones importantes con `\label{eq:}`

**Entregable**: Lista de gaps → iteración en Fase 2 si hay, sino → Fase 4.

---

## Fase 4: Pulido y evaluación final

### Paso 4.1: Prosa — Scientific-Writing
- Sin bullet points en texto final; sección abre con párrafo motivacional
- Transiciones explícitas; tono formal pero accesible; párrafos 4-8 oraciones
- Voz activa para el modelo; pasiva para convenciones; terminología consistente

### Paso 4.2: ScholarEval (dimensiones 3, 7, 8)
- **Dim 3 — Methodology**: alineamiento diseño-preguntas, rigor, reproducibilidad, limitaciones (≥4.5/5)
- **Dim 7 — Writing**: claridad, tono, accesibilidad (≥4.5/5)
- **Dim 8 — Citations**: completitud, calidad, balance (≥4.0/5)

Si alguna <4.0 → iteración puntual.

### Paso 4.3: Compilación
- Compila sin errores; sin warnings de `\ref`/`\cite` indefinidos; placeholders renderizan

**Entregable**: `.tex` + `.bib` finales en calidad publicable.

---

## Entregables por fase

| Fase | Entregable | Aprobación |
|---|---|---|
| 0 | `INVENTARIO_TRIAGE.md` | **Sí** |
| 1 | `.tex` esqueleto + `.bib` | No |
| 2 | `.tex` contenido completo | No |
| 3 | Lista de gaps + correcciones | Sí si ambiguo |
| 4 | `.tex` + `.bib` finales | **Sí** |

## Notas operativas
- **Contexto**: El MD (~39K tokens) no cabe en una pasada. Fase 0 por secciones, Fase 2 por bloques temáticos.
- **Sección por sesión**: Priorizar una sección completa verificada sobre todo el capítulo a medias.
- **Continuidad inter-sesión**: `INVENTARIO_TRIAGE.md` y glosario de notación son los docs de continuidad.
- **Apéndices**: No se redactan aquí. Solo lista de contenidos con refs cruzadas.
