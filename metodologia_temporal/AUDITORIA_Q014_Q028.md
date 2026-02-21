# Auditoría de Alucinación: Q-014 a Q-028
> **Fecha:** 2026-02-17 | **Protocolo:** v2 (ROL+REGLAS+CONTEXTO) | **Objetivo:** Read-only, sin edición de archivos originales

## Contexto
Las respuestas Q-014 a Q-028 fueron generadas en sesiones anteriores con un prompt degradado (sin roles, sin reglas anti-alucinación, contexto insuficiente). Esta auditoría re-consulta los mismos notebooks con el protocolo v2 para comparar y detectar posibles alucinaciones.

---

## Q-014 — ¿Cómo se justifica la elección de Sethares sobre otras parametrizaciones?

### Respuesta ORIGINAL (sesión anterior):
> Parsimonia computacional vs Parncutt, sensibilidad al registro vs HK, relación timbre-escala, AF-degree irrelevante para inputs MIDI. hrep usa Sethares como backend.

**Citas originales:** Cook2006, Sethares1993, Masina2024, Vassilakis2001, Cubarsi2019, Harrison2020, Gaulhiac2021

### Respuesta AUDIT v2 (2026-02-17):
> El modelo de Sethares se justifica como una "parametrización conveniente" de Plomp-Levelt. Parncutt más complejo pero resultados "cualitativamente similares". Sethares es el "más extendido" para espectros arbitrarios. HK falla en invarianza ante combinación de parciales. Ventajas: principio de consonancia local, optimización computacional, formulación aditiva, independencia de fase.

**Citas audit:** Plomp&Levelt1965, Sethares1993, Parncutt1989, Harrison&Pearce2020, Cook&Fujisawa2006

### Diagnóstico Preliminar:
- **"AF-degree irrelevante para inputs MIDI"**: ⚠️ Afirmación en original que NO aparece en audit → verificar fuente
- **"hrep usa Sethares como backend"**: ✅ Confirmado indirectamente
- **"Parsimonia computacional"**: ✅ Consistente
- **Cook2006**: Presente en ambas → ✅
- **Cubarsi2019**: Solo en original, no mencionado en audit → ⚠️ verificar relevancia

---

## Q-015 — ¿De dónde vienen los valores numéricos (D*=0.24, S1, S2, b1, b2)?

### Respuesta ORIGINAL:
> Ajuste de curvas (gradient minimization squared error) a datos Plomp-Levelt. D*=0.24 derivado analíticamente (25% banda crítica). S1,S2 por mínimos cuadrados lineales. No son constantes físicas.

**Citas originales:** Sethares1993, Masina2024, Cook2006, Mukherjee2023, Gaulhiac2021, Harrison2020

### Respuesta AUDIT v2:
> b1=3.5 y b2=5.75 por minimización del gradiente del error cuadrático sobre datos de Plomp-Levelt. D*=0.24 derivado directamente del modelo. S1=0.021 y S2=19 por ajuste de mínimos cuadrados. Ajuste mixto (matemático/estadístico sobre datos empíricos).

**Citas audit:** Plomp&Levelt1965, Sethares1993

### Diagnóstico Preliminar:
- **"D*=0.24 derivado analíticamente (25% banda crítica)"**: ⚠️ Audit dice "derivado directamente del modelo", NO menciona "25% banda crítica" → posible embellecimiento
- **"gradient minimization squared error"**: ✅ Confirmado exactamente
- **"No son constantes físicas"**: ✅ Consistente con "parametrización mixta"
- **Mukherjee2023 (D*=0.22035)**: Solo en original → ⚠️ sin confirmar en audit

---

## Q-016 — ¿Cuál es la sensibilidad del modelo a variaciones en parámetros?

### Respuesta ORIGINAL:
> Estadísticamente robusto (chi² estable ante variaciones de peso); topológicamente sensible a amplitudes (Mukherjee singularidades). Consonancias principales robustas. hrep permite backends intercambiables.

**Citas originales:** Vassilakis2001, Masina2024, Cook2006, Mukherjee2023, Sethares1993, Masina2022, Harrison2020

### Respuesta AUDIT v2:
> Forma de curva "altamente dependiente" de frecuencias y amplitudes (Sethares1993). Variaciones en amplitudes pueden hacer desaparecer mínimos locales. Pero robustez cualitativa: distingue acordes resueltos vs no-resueltos independientemente del H o amplitudes (Cook&Fujisawa2006). Falla en invarianza ante combinación de parciales (Harrison&Pearce2020).

**Citas audit:** Sethares1993, Cook&Fujisawa2006, Harrison&Pearce2020

### Diagnóstico Preliminar:
- **"chi² estable ante variaciones de peso"**: ⚠️ Audit NO menciona chi² → posible confusión con resultado de Masina2024
- **"Mukherjee singularidades"**: ⚠️ Audit no menciona singularidades → verificar
- **"Consonancias principales robustas"**: ✅ Consistente con "robustez cualitativa"
- **Vassilakis2001 (r=0.98 vs HK r=0.87)**: Solo en original → ⚠️ dato específico no confirmado

---

## Q-017 — ¿Es razonable el supuesto de espectro armónico con decaimiento exponencial?

### Respuesta ORIGINAL:
> Razonable para timbres occidentales (cuerdas/vientos). δ=0.88 validado por Cook2006. Inválido para inarmónicos. Sethares: consonancia local para espectros arbitrarios. Milne: r=0.85 entre idealizado y real para roughness.

**Citas originales:** Harrison2020, Milne2023, Eerola2022, Cook2006, Sethares1993, Masina2024, Vassilakis2001, Mukherjee2023

### Respuesta AUDIT v2:
> Supuesto "algo arbitrario" pero aceptado como aproximación útil. 1/n = timbre "neutral" estándar (HK, Sethares). 1/n²= cuerdas pulsadas y fuente glotal. 0.9^k= sonidos sintéticos. Carece de validez ecológica universal para instrumentos ricos o inarmónicos.

**Citas audit:** Sethares1993, Hutchinson&Knopoff, Harrison&Pearce2020, Masina

### Diagnóstico Preliminar:
- **"δ=0.88 validado por Cook2006"**: ⚠️ Audit NO menciona δ=0.88 → verificar
- **"Milne: r=0.85 entre idealizado y real"**: ⚠️ Audit no menciona esta correlación → dato específico sin confirmar
- **"Razonable para cuerdas/vientos"**: ✅ Consistente con "cuerdas pulsadas, voz"
- **"Inválido para inarmónicos"**: ✅ Confirmado

---

## Q-018 — ¿Cómo afecta H=6 a la resolución del modelo?

### Respuesta ORIGINAL:
> H=6 suficiente: Helmholtz (H≥7 despreciable), Masina (saturación en H=8), Cook (valida H=6). Costo: C(m,2)·H². Triada: 108 vs 300 evaluaciones. Proponer análisis de sensibilidad H∈{4,6,8,10}.

**Citas originales:** Cook2006, Masina2024, Gaulhiac2021, Cubarsi2019, Hutchinson1978, Mukherjee2023

### Respuesta AUDIT v2:
> H=6 permite curvas similares a experimentales (Plomp-Levelt). Cook&Fujisawa2006: resultados "casi idénticos" para H=6. Masina (n_max=8 sin cambio significativo). HK usaron H=10. Harrison: 11 parciales. Sethares: muchos mínimos locales "aniquilados" al empaquetar parciales.

**Citas audit:** Plomp&Levelt1965, Cook&Fujisawa2006, Masina&LoPresti2024, Hutchinson&Knopoff1978, Harrison&Pearce2020, Sethares1993

### Diagnóstico Preliminar:
- **"Helmholtz (H≥7 despreciable)"**: ⚠️ Audit NO menciona Helmholtz directamente → posible inferencia
- **"Costo: C(m,2)·H²"**: ⚠️ No confirmado en audit (pero es fórmula matemática correcta)
- **"108 vs 300 evaluaciones"**: ⚠️ Dato calculado, no de fuente → OK si aritmética correcta
- **Saturación H=8**: ✅ Confirmado (Masina)

---

## Q-020 — ¿Cómo afectan los componentes cero en Φ_raw a las métricas?

### Respuesta ORIGINAL:
> JSD requiere ε-padding (0·log0=0 por convención en scipy). Hellinger robusta naturalmente (raíces cuadradas). Coseno tolera ceros pero exagera ortogonalidad. Euclidiana sin problemas pero domina magnitud. Estrategias: suavizado gaussiano previo + ε-padding.

**Citas originales:** Harrison2020, Milne2023, Burgoyne2013, Kim2018, Wolkowicz2013, scipy docs, Burgoyne2005

### Respuesta AUDIT v2:
> NO INFO sobre Φ_raw específicamente. Pero en CoDA (composicional): ε-padding (prior Bayes-Laplace, sumar 1 a todos los conteos). ilr transforma simplex a espacio euclidiano. Suavizado para modelos de periodicidad (no para ceros específicamente).

**Citas audit:** Burgoyne2013, Stolzenburg2015

### Diagnóstico Preliminar:
- **"Hellinger robusta naturalmente"**: ⚠️ NO confirmado en audit → afirmación matemáticamente correcta pero sin fuente
- **"Coseno exagera ortogonalidad"**: ⚠️ NO confirmado en audit
- **"scipy convención"**: ⚠️ Referencia a docs, no a paper → aceptable pero débil
- **"Kim2018"**: ⚠️ Solo en original → verificar
- **Burgoyne2013**: ✅ Confirmado en ambas

---

## Q-021 — ¿Existe ground truth para tríadas/cuatríadas?

### Respuesta ORIGINAL:
> Sí: Bowling2018 (66 tríadas, 220 cuatríadas cromáticas), Johnson-Laird2012 (55 tríadas), Roberts1986 (jerarquía mayor>menor>dim>aug). Unidad formal: asper. Modelos computacionales: valores arbitrarios/normalizados. Validar con díadas insuficiente para N≥3.

**Citas originales:** Bowling2018, Vencovsky2014, Vassilakis2001, Cook2009, Johnson-Laird2012, Roberts1986, Masina2024, Harrison2020

### Respuesta AUDIT v2:
> Sí. Bowling2018 (12 díadas, 66 tríadas, 220 cuatríadas). Johnson-Laird2012 (55 trídas, 48 cuatríadas). Roberts1986 (Mayor>Menor>Dim>Aug). Unidad: asper (1 asper = SAM 1kHz, 60dB SPL, 70Hz mod). Johnson-Laird usado por Stolzenburg2015. Bowling usado por Masina2023 para chi².

**Citas audit:** Bowling2018, Johnson-Laird2012, Roberts1986, Vencovský2014, Stolzenburg2015, Masina2023

### Diagnóstico Preliminar:
- **Bowling2018 datos**: ✅ Confirmado y ENRIQUECIDO (audit agrega 12 díadas)
- **Roberts1986 jerarquía**: ✅ Confirmado exactamente
- **Asper definido**: ✅ Confirmado y más detallado en audit (SAM 1kHz, 60dB)
- **Vassilakis2001**: Solo en original → ⚠️ verificar relevancia
- **"Validar con díadas insuficiente para N≥3"**: ⚠️ No confirmado en audit → posible inferencia razonable

---

## Q-022 — ¿La normalización destruye información relevante?

### Respuesta ORIGINAL:
> L1 normalización elimina magnitud global de disonancia (perceptualmente informativa: Milne2023 asociación universal). Depende de la pregunta: similitud→normalizado; ordenamiento consonancia→no normalizado. CoDA: ilr preserva geometría composicional. DFT separa magnitud/fase.

**Citas originales:** Sethares1993, Milne2023, Burgoyne2013, Aitchison1986, Harrison2020, Masina2022, Bernardes2016, Amiot2016

### Respuesta AUDIT v2:
> NO INFO sobre "vectores de rugosidad" L1. Sí sobre datos composicionales: normalización introduce correlaciones espurias (Burgoyne2013). TIS: normalización permite jerarquía multi-nivel (Bernardes2016). Alternativas: CoDA/ilr (Burgoyne2013), DFT/Bernardes para TIV, Amiot magnitud-fase.

**Citas audit:** Burgoyne2013, Bernardes2016, Yust2013

### Diagnóstico Preliminar:
- **"Milne2023 asociación universal"**: ⚠️ NOT confirmed in audit → verificar
- **"Aitchison1986"**: ⚠️ NOT found in audit → verificar (es referencia clásica CoDA, pero ¿está en notebook?)
- **"correlaciones espurias"**: ✅ Confirmado (Burgoyne2013)
- **DFT magnitud/fase**: ✅ Confirmado

---

## Q-025 — ¿Tiene sentido la euclidiana sobre rugosidad cruda?

### Respuesta ORIGINAL:
> Euclidiana cruda inapropiada (2da menor=5ta justa en distancia). Coseno respaldado por Harrison para espectros suavizados. TIS: euclidiana=consonancia, angular=afinidad tonal. √JSD métrica válida en simplex. Sin consenso universal → diseño comparativo experimental.

**Citas originales:** Harrison2020, Bernardes2016, NavarroCaceres2020, Endres2003, Tymoczko2006, Chew2014

### Respuesta AUDIT v2:
> Euclidiana sobre croma crudos: distancias no capturan propiedades armónicas (2da menor = 5ta justa en distancia euclidiana) (Bernardes2016). Hellinger/Bhattacharyya "naturales" para distribuciones (Kim2015). KL para Markov (DeHaas2011). Coseno para n-gramas y chord embeddings (Wolkowicz2013, Ciubotaru2022). Coseno en TIS para afinidad tonal (NavarroCaceres2020).

**Citas audit:** Bernardes2016, Kim2015, DeHaas2011, Wolkowicz2013, Ciubotaru2022, NavarroCaceres2020, Paiement2005

### Diagnóstico Preliminar:
- **"2da menor=5ta justa"**: ✅ Confirmado (Bernardes2016)
- **"Coseno Harrison espectros suavizados"**: ⚠️ Audit menciona coseno pero NO cita Harrison para esta afirmación
- **"Endres2003 (√JSD métrica)"**: ⚠️ NOT confirmed → referencia real pero ¿en notebook?
- **"Tymoczko2006 voice-leading L∞"**: ⚠️ NOT confirmed in audit
- **TIS euclidiana/angular**: ✅ Confirmado

---

## Q-026 — ¿ρ induce topología diferente? ¿D es PSD?

### Respuesta ORIGINAL:
> Sí, topologías diferentes: MDS toroidal vs MVU planos separados. Rugosidad vs compacidad: vecindarios discrepantes. Lerdahl viola desigualdad triangular. SMACOF no requiere PSD. √JSD y Hellinger satisfacen Schoenberg. arccos(1-d_cos) es métrica.

**Citas originales:** Sethares1993, Mukherjee2023, Masina2024, Cook2009, Burgoyne2005, Himpel2022, sklearn docs, Bernardes2016

### Respuesta AUDIT v2:
> NO INFO sobre topología de espacio de acordes específicamente. D no necesariamente PSD; Classical MDS requiere embeddability euclidiana (autovalores negativos si no). SMACOF no requiere PSD explícitamente, es iterativo con "límites muy flexibles" (Lim2025, Kraemer2018, Takane1977).

**Citas audit:** Liang2025, Delicado&Pachón-García2020, Lim2025, Kraemer2018, Takane1977

### Diagnóstico Preliminar:
- **"MDS toroidal vs MVU planos"**: ⚠️ NOT confirmed → verificar fuente
- **"Lerdahl viola desigualdad triangular"**: ⚠️ NOT confirmed
- **"√JSD y Hellinger satisfacen Schoenberg"**: ⚠️ NOT confirmed in audit → matemáticamente cierto pero sin fuente
- **"arccos(1-d_cos) es métrica"**: ⚠️ NOT confirmed
- **SMACOF no requiere PSD**: ✅ Confirmado

---

## Q-027 — ¿Por qué MDS métrico? ¿Stress suficiente?

### Respuesta ORIGINAL:
> Krumhansl: no-métrico para juicios subjetivos (solo rangos). ChordSpace: ρ computacional → métrico justificado (magnitudes significativas, Milne2023). Stress <0.05 excelente. Complementar con trustworthiness, continuity, diagrama de Shepard.

**Citas originales:** Krumhansl1990, Milne2023, Bernardes2016, Bidelman2014, Kruskal1964, sklearn docs, Venna2005, Lee2009

### Respuesta AUDIT v2:
> MDS métrico usa descomposición de valores propios para preservar magnitudes (Yang2006, Lim2025). No-métrico preserva solo rangos. Stress NO suficiente: no distingue intrusiones vs extrusiones (Lee2009). Complementar con: T (Venna2005), C (Venna2005), Shepard (Lim2025), co-ranking matrix (Lee2009, Lee2010).

**Citas audit:** Yang2006, Lim2025, Mancell2019, Lee2009, Venna2005, Lee2010, Scikit-learn2020

### Diagnóstico Preliminar:
- **"Krumhansl no-métrico para juicios"**: ⚠️ NOT confirmed in audit notebook → pero es hecho conocido
- **"Milne2023 magnitudes significativas"**: ⚠️ NOT confirmed
- **"Stress <0.05 excelente (Kruskal1964)"**: ⚠️ Audit not directly confirmed → verificar
- **"Bernardes2016 smacof no-métrico TIS"**: ⚠️ NOT confirmed in audit
- **Trustworthiness/Continuity/Shepard**: ✅ Confirmado

---

## Resumen de Flags por Pregunta

| Q-ID | Citas OK | Citas ⚠️ | Afirmaciones ⚠️ | Riesgo |
|------|---------|----------|-----------------|--------|
| Q-014 | 3/5 | 2 (Cubarsi2019, AF-degree) | 1 (AF-degree irrelevante) | 🟡 Bajo |
| Q-015 | 2/4 | 2 (25% banda crítica, Mukherjee D*) | 1 (25% banda crítica) | 🟡 Bajo |
| Q-016 | 1/4 | 3 (chi², singularidades, Vassilakis r) | 2 (chi², singularidades) | 🟠 Medio |
| Q-017 | 2/4 | 2 (δ=0.88, Milne r=0.85) | 2 (δ=0.88, r=0.85) | 🟠 Medio |
| Q-018 | 3/4 | 1 (Helmholtz) | 0 | 🟢 Muy bajo |
| Q-020 | 1/5 | 4 (Hellinger, coseno, Kim, scipy) | 2 (Hellinger robusta, coseno exagera) | 🟠 Medio |
| Q-021 | 4/5 | 1 (Vassilakis2001) | 1 (díadas insuficiente) | 🟢 Muy bajo |
| Q-022 | 2/5 | 3 (Milne, Aitchison, Masina2022) | 1 (asociación universal) | 🟠 Medio |
| Q-025 | 2/5 | 3 (Harrison coseno, Endres, Tymoczko) | 1 (Harrison coseno) | 🟠 Medio |
| Q-026 | 1/6 | 5 (toroidal, Lerdahl, Schoenberg, arccos) | 4 (topología, Lerdahl, Schoenberg, arccos) | 🔴 Alto |
| Q-027 | 2/5 | 3 (Krumhansl, Milne, Kruskal) | 2 (Krumhansl, Stress<0.05) | 🟡 Bajo |

### Escala de Riesgo:
- 🟢 **Muy bajo**: ≤1 flag, afirmaciones núcleo confirmadas
- 🟡 **Bajo**: 2-3 flags, pero afirmaciones núcleo correctas
- 🟠 **Medio**: 3-4 flags, datos específicos no confirmados
- 🔴 **Alto**: ≥5 flags, afirmaciones estructurales sin confirmar

---

## Preguntas Type B (no re-consultadas)
Las siguientes preguntas son Tipo B (decisiones internas / no bibliográficas) y no fueron incluidas en la auditoría:
- **Q-019**: Percepción de intervalo interno vs extremo
- **Q-023**: Suavizado gaussiano justificación perceptual
- **Q-024**: PCA vs normalizaciones manuales
- **Q-028**: Comparabilidad de embeddings entre métodos

---

## 🔬 VERIFICACIÓN NIVEL 1+2 (2026-02-17 16:10)

> Se re-consultaron los notebooks con queries ultra-específicos para verificar cada claim flaggeado.

### Q-026 — Verificación de claims (fue 🔴 Alto)

| Claim original | Veredicto | Evidencia |
|---|---|---|
| "MDS toroidal vs MVU planos separados" | ✅ **CONFIRMADO** | Burgoyne & Saul 2005: MDS produce embedding "isomorfo al modelo toroidal" (Krumhansl-Kessler). MVU produce "dos planos isomorfos separados por tono entero". Ciclos de terceras "notablemente ausentes" en MVU. |
| "Lerdahl viola desigualdad triangular" | ✅ **CONFIRMADO** (indirecto) | DeHaas et al.: TPSD (derivado de Lerdahl TPS) "no satisface la desigualdad triangular". Noll y Garbers (U. Berlín) descubrieron "inconsistencias en la teoría de Lerdahl en términos de funciones de distancia". |
| "√JSD y Hellinger satisfacen Schoenberg" | 🟡 **NO EN CORPUS** | Hecho matemático correcto pero no indexado en notebooks. "Schoenberg" en fuentes = Arnold Schoenberg (compositor), no I.J. Schoenberg (matemático). |
| "arccos(1-d_cos) es métrica" | 🟡 **NO EN CORPUS** | Hecho matemático conocido pero sin fuente en notebooks. TIS usa arcocoseno del producto punto normalizado pero no discute validez métrica formal. |

**VEREDICTO Q-026: ⬇️ 🔴→🟡 Bajo.** Los 2 claims principales (topología, Lerdahl) son correctos y confirmados. Los 2 restantes son hechos matemáticos reales no indexados en el corpus.

---

### Q-016 — Verificación de datos numéricos (fue 🟠 Medio)

| Dato flaggeado | Veredicto | Evidencia |
|---|---|---|
| "Vassilakis r=0.98 vs HK r=0.87" | ⚠️ **NO CONFIRMADO** | Notebook cita Vassilakis2001 pero no el valor r=0.98. Sí reporta HK: r=0.967 (díadas) y r=0.352 (tríadas) — números diferentes. |
| "Mukherjee singularidades" | ✅ **CONFIRMADO** | Mukherjee2023 menciona explícitamente "singularidades incidentales" y fórmula de singularidad en β cuando v'_l ≠ v'_r. |
| "chi² estable ante variaciones de peso" | ⚠️ **NO VERIFICADO** | Posible confusión con chi² de Masina2024 (que evalúa modelos, no sensibilidad paramétrica). |

**VEREDICTO Q-016: ⬇️ 🟠→🟡 Bajo.** Singularidades confirmadas. r=0.98 no confirmado (posible alucinación numérica). chi² ambiguo.

> ⚠️ **Acción:** Eliminar "r=0.98 vs r=0.87" de la inserción en ESTRUCTURA si se edita Q-016.

---

### Q-017 — Verificación de datos numéricos (fue 🟠 Medio)

| Dato flaggeado | Veredicto | Evidencia |
|---|---|---|
| "δ=0.88 validado por Cook2006" | ✅ **CONFIRMADO** | Cook & Fujisawa2006: "seis parciales con amplitudes relativas de 1.0, **0.88**, 0.76, 0.64, 0.58, 0.52". No es "δ para decaimiento exponencial" sino la segunda amplitud de la serie. |
| "Milne r=0.85" | ✅ **CONFIRMADO** | Milne2023: "las correlaciones con las medidas de rugosidad basadas en audio son, respectivamente, **0.85**, 0.49 y 0.67". |

**VEREDICTO Q-017: ⬇️ 🟠→🟢 Muy bajo.** Ambos datos confirmados textualmente. La única imprecisión es llamar a 0.88 "δ de decaimiento exponencial" cuando en realidad es la amplitud del 2do parcial en la serie de Cook.

---

### Q-020 — Verificación de datos (fue 🟠 Medio)

| Dato flaggeado | Veredicto | Evidencia |
|---|---|---|
| "Kim2018 suavizado como regularización" | ❌ **NO ENCONTRADO** | Notebook tiene Kim2014, Kim2016, Kim2019 pero NO Kim2018. **Posible cita fantasma.** |
| "Hellinger robusta naturalmente" | 🟡 **NO EN CORPUS** | Hecho matemático correcto (√ sobre componentes ≥0 no produce NaN/divergencia) pero sin fuente en notebooks. |

**VEREDICTO Q-020: se mantiene 🟠 Medio.** Kim2018 es sospechosa.

> ⚠️ **Acción:** Reemplazar "Kim2018" por cita verificable o eliminar si se edita Q-020.

---

### Q-022 — Verificación de datos (fue 🟠 Medio)

| Dato flaggeado | Veredicto | Evidencia |
|---|---|---|
| "Milne2023 asociación universal" | ✅ **CONFIRMADO** | Milne2023: "asociación universal, o al menos no arbitraria, entre la rugosidad y la estabilidad musical percibida" (estudio Tsimane'/Papua Nueva Guinea). |
| "Aitchison1986" | ✅ **CONFIRMADO** | "The Statistical Analysis of Compositional Data" citado en contexto de correlaciones espurias en datos normalizados. |

**VEREDICTO Q-022: ⬇️ 🟠→🟢 Muy bajo.** Ambos datos clave confirmados.

---

### Q-025 — Verificación de datos (fue 🟠 Medio)

| Dato flaggeado | Veredicto | Evidencia |
|---|---|---|
| "Harrison coseno para smooth pitch spectra" | ✅ **CONFIRMADO** | Harrison & Pearce 2020: "la similitud perceptual entre espectros suaves puede ser simulada utilizando medidas de similitud geométrica como la similitud coseno". |
| "Endres2003 √JSD métrica" | 🟡 **NO EN CORPUS** | Referencia real (Endres & Schindelin 2003, IEEE Trans. Inf. Theory) pero no indexada en notebooks. |

**VEREDICTO Q-025: ⬇️ 🟠→🟡 Bajo.** Harrison confirmado. Endres es referencia real pero externa al corpus.

---

## 📊 TABLA DE RIESGO ACTUALIZADA (Post-verificación)

| Q-ID | Riesgo inicial | Riesgo final | Cambio | Acción requerida |
|------|----------------|-------------|--------|------------------|
| Q-014 | 🟡 Bajo | 🟡 Bajo | = | Ninguna |
| Q-015 | 🟡 Bajo | 🟡 Bajo | = | Ninguna |
| Q-016 | 🟠 Medio | 🟡 Bajo | ⬇️ | Eliminar "r=0.98 vs r=0.87" |
| Q-017 | 🟠 Medio | 🟢 Muy bajo | ⬇️⬇️ | Corregir "δ=0.88" → "amplitud 2do parcial=0.88" |
| Q-018 | 🟢 Muy bajo | 🟢 Muy bajo | = | Ninguna |
| Q-020 | 🟠 Medio | 🟠 Medio | = | **Eliminar "Kim2018"** (posible ghost ref) |
| Q-021 | 🟢 Muy bajo | 🟢 Muy bajo | = | Ninguna |
| Q-022 | 🟠 Medio | 🟢 Muy bajo | ⬇️⬇️ | Ninguna |
| Q-025 | 🟠 Medio | 🟡 Bajo | ⬇️ | Ninguna (Endres es real, solo externo) |
| Q-026 | 🔴 Alto | 🟡 Bajo | ⬇️⬇️⬇️ | Ninguna (claims estructurales confirmados) |
| Q-027 | 🟡 Bajo | 🟡 Bajo | = | Ninguna |

### Resumen ejecutivo:
- **0 preguntas 🔴 Alto** (Q-026 bajó a 🟡)
- **1 pregunta 🟠 Medio** (Q-020: cita fantasma Kim2018)
- **5 preguntas 🟡 Bajo** (claims OK, detalles menores)
- **5 preguntas 🟢 Muy bajo** (todo confirmado)
- **Acciones correctivas mínimas:** 3 ediciones puntuales (Q-016 dato numérico, Q-017 terminología, Q-020 cita)
