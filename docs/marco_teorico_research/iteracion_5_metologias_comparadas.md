## 2.2.2 Extensión a Espectros Armónicos Reales: Sumatoria Inter-parcial

Para evaluar la disonancia de un objeto musical compuesto (por ejemplo, una tríada temperada), el modelo de Sethares no procesa las notas topológicamente como cajas negras indivisibles. En su lugar, descompone la verticalidad en un único macromapaje espectral discreto $\mathcal{F}$ que engloba a todos los parciales de todas las notas constituyentes.

### Generación del Espectro Compuesto y Decaimiento Temporal
Si el acorde topológico se compone de $K$ notas y limitamos la simulación física a $H$ armónicos por nota (típicamente $H=6$ o $H=11$), el espectro total consolida $N = K \times H$ parciales en un solo vector. Cada $n$-ésimo parcial hereda un tono base $f_n$ (múltiplo entero riguroso de la fundamental que lo origina) y una amplitud $v_n$.

Para inducir realismo y emular la pérdida de energía en series de Fourier para timbres físicos (como los cordófonos o la voz), se impone un factor de decaimiento temporal y espectral monótono, $\delta$. En la literatura contemporánea, una penalidad por amortiguación suave de $\delta = 0.88$ rinde una serie de amplitudes sintéticas decaídas progresivamente, por ejemplo: $\{1.0, \, 0.88, \, 0.77, \, 0.68, \dots \}$.

### Iteración Matricial
La medida universal de rugosidad $D_{\mathcal{F}}$ del campo sonoro se totaliza matemáticamente calculando la norma de interferencia sobre todas las componentes de la matriz triangular superior estricta formada por los pares del macromapaje de parciales espectrales. Permutando todos los posibles encadenamientos de colisiones $i, j$, sumamos el funcional base $d$:

$$ D_{\mathcal{F}} = \sum_{i=1}^{N-1} \sum_{j=i+1}^N d(f_i, f_j, v_i, v_j) $$

El factor de decaimiento constante permite al modelo iluminar las interferencias críticas ocultas en los armónicos agudos (e.g. el conflicto inarmónico en una tríada aumentada generada por choques entre el quinto armónico de la raíz y el cuarto armónico de la tercera mayor) en lugar de sepultarlos o de exagerar su impacto.

## 2.2.3 Alternativas Topológicas y el Estado del Arte

Aunque el modelo iterativo asimilado de Sethares rige imperantemente en la literatura moderna de psicoacústica por su balance geométrico y eficiencia analítica, otras parametrizaciones compiten modelando desde orígenes fisiológicos distintos.

| Modelo / Autor | Fundamento Matemático y Ponderación | Ventajas Analíticas (Pros) | Limitaciones y Omisiones (Contras) |
| :--- | :--- | :--- | :--- |
| **Hutchinson & Knopoff** (1978) | Interferencia pura. Suma los batidos primarios con una aproximación de banda crítica tabulada. Ponderación de amplitud con caída $1/n$. | Primer modelo en iterar sobre armónicos complejos en lugar de evaluar díadas simples aisladas. Expande el análisis real. | Dependencia en curvas empíricas tabuladas no parametrizadas. El factor enmascaramiento es deficiente frente a la actualización de "mínimo" empleada por Sethares. |
| **William A. Sethares** (1993/2005) | Parametrización exponencial de la curva de Plomp-Levelt con escalamiento frecuencial. Usa $v_{12} = \min(v_1, v_2)$. *Standard Base*. | Escalable geométrica y algorítmicamente a cualquier timbre o sistema de afinación. Integración continua rigurosa. | Omisión ciega de variables macro-estructurales (armonicidad o tensión por simetría) y fallo para priorizar la tríada menor sobre la disminuida. |
| **Pantelis Vassilakis** (2001) | Modificación centrándose estrictamente en fluctuaciones agudas de envolvente de onda y modulación de presiones sonoras. | Produce valores más exactos en ciertos intervalos complejos como la séptima mayor, simulando el pulso físico más fielmente. | Elevada carga computacional. Es un derivado ciego a la "armonicidad", incapaz de discernir consonancias perfectas desde un plano cognitivo. |
| **Norman D. Cook** (2006/2017) | Aproximación de tres tonos integrando el fenómeno Sethares con penalizaciones cognitivas ("Tensión" por equidistancia) y "Modalidad". | Logra reproducir exactamente la ordenación perceptual empírica de tríadas (Mayor > Menor > Suspendida > Disminuida > Aumentada). | La "Tensión" por equidistancia es conceptualmente *ad hoc* y cultural; rompe la biológica pureza del *bottom-up* auditivo original para "forzar" los resultados musicales esperados. |
| **Masina & Lo Presti** (2024) | Función heurística compuesta que suma **Compacidad** (Periodicidad algorítmica suavizada con Gaussianas) + **Rugosidad**. | Corrige topológicamente la tríada clásica de Sethares fusionando periodicidad fundamental (armonicidad de un bajo espectral). Restringe excesos físicos en batidos de segundo orden. | Destruye la elegancia unificada del parámetro único de Sethares. Exige ajustar constantes empíricas (como el peso $F$ al 50\%, o penalizaciones $c_5, c_8$ sobre quintas/octavas). |

En el diseño formal del modelo `ChordSpace`, nos abstraeremos estratégicamente del debate sumando heurísticas dudosas. Implementaremos intrínsecamente la topología clásica de Sethares como vector característico fundamental para preservar la pureza bottom-up puramente psicoacústica, blindando el modelo paramétrico de interpretaciones idiomáticas (como la tensión equidistante cultural de Cook), aunque sacrificamos expresividad en conjuntos extremales simétricos como las tríadas aumentadas.
