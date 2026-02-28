## 2.3 Formalización Combinatoria y Computacional del Espacio

Habiendo cimentado la métrica de interferencia puramente acústica mediante el funcional paramétrico de Sethares, debemos ahora dotar de precisión geométrica al "objeto musical" sobre el que iterarán estos cálculos. Al matematizar la teoría musical, las decisiones fundacionales de espacios de representación definen los límites insalvables de nuestra inferencia analítica posterior.

### 2.3.1 El Espacio Discreto de Acordes $\mathcal{A}$

Sea $P \subset \mathbb{R}$ un espacio discreto que representa el dominio de alturas tonales absolutas (pitches), parametrizado logarítmicamente (ej. mediante números de nota MIDI unidimensionales, donde la altura es proporcional a $\log_2(f)$). Definimos formalmente un "Acorde" $\mathcal{C}$ de cardinalidad $n$ como una tupla estrictamente ordenada:

$$ \mathcal{C} = (p_1, p_2, \dots, p_n) \in P^n $$

sujeta a la restricción de monotonicidad absoluta $p_1 < p_2 < \dots < p_n$. 

Bajo esta formulación topológica, la estructura combinatoria de $\mathcal{C}$ preserva las coordenadas directas y el empaquetado exacto (el *voicing*), declarando unívocamente a $p_1$ como el estrato inferior vinculante ("bajo" acústico) del sistema continuo.

### 2.3.2 El Rechazo a los Pitch-Class Sets (PC-sets)

El paradigma computacional hegemónico del siglo XX en teoría atonal (postulado rigurosamente por Allen Forte y John Rahn) modela los acordes mediante la compactación geométrica del espacio continuo $P$ sobre un grupo cociente modular $Z_{12} = P / \sim_{octava}$. Este sistema decreta formalmente dos clases de equivalencia universales, las cuales asumimos rechazar íntegramente en la construcción topológica de `ChordSpace`:

1.  **Equivalencia de Octava:** Asume redundancia cíclica donde $p \sim p'$ si $p \equiv p' \pmod{12}$.
2.  **Equivalencia de Transposición ($T_n$):** Postula invariancia analítica bajo traslaciones globales uniformes en el dominio logarítmico (una melodía movida paralela es idéntica a sí misma).

Para que un modelo predictivo base sus fronteras en perceptos acústicos humanos estables (disonancia bottom-up) y no en heurísticas simbólicas ciegas, es imperativo abstraerse de los PC-Sets hacia representaciones de *"Pitch Chords"* (Harrison & Pearce, Eerola, Bowling). Matemáticamente, esto implica que si definimos un mapeo de disonancia o identidad acústica $\mathcal{R}: P^n \to \mathbb{R}$, ratificamos que $\mathcal{R}$ no es covariante al operador aritmético de módulo 12, ni tampoco puede ser globalmente conmutativo o estacionario bajo traslaciones afines del subconjunto métrico, i.e., $\mathcal{R}(\mathcal{C}) \neq \mathcal{R}(\mathcal{C} \pm k)$.

#### Justificación Bottom-Up: Ruptura de Simetría por Ancho de Banda Crítico

La necesidad inalienable de retener la tupla estricta sobre el dominio $P^n$ radica en la asimetría log-estructural del sistema auditivo periférico.

Como analizamos en secciones precedentes, la función que pondera la penalidad interferente (el Ancho de Banda Crítico $CBW$ incrustado en la curva paramétrica) depende inexorablemente del registro ($pitch\ height$). Como el factor logarítmico de un intervalo permanece estático pero las magnitudes hercianas de su base colapsan en registros bajos, un mismo bloque interválico trasladado ortogonalmente hacia el bajo sufrirá penalizaciones de disonancia exponenciales por incidir severamente dentro de anchos de banda críticos superpuestos (el síndrome del *low-interval limit*). 

Ignorar esta no-linealidad colapsando el grupo a clases de tono reduce ciegamente una topología disonante turbia (como un racimo de 3ras en un sub-bajo) a formar la misma entidad discreta con un clúster diáfano y terso transpuesto ochenta hercios arriba, un isomorfismo falso e inaceptable al momento de formular una métrica geométrica unificada de la percepción real sonora de acordes complejos.
