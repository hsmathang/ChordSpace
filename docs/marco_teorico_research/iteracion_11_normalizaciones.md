## 2.4.2 Transformaciones Composicionales y Tolerancia Cognitiva (Simplex)

El isomorfismo propuesto sobre divergencias JSD o Hellinger asume subrepticiamente la existencia de una distribución de probabilidad sobre el dominio de dimensión 12. La justificación combinatoria estricta obliga a ejecutar previamente transformaciones escalares (o normalizaciones) que restrinjan el crecimiento desmesurado de energía de tensores de alta densidad armónica (maldición de dimensionalidad por cardinalidad).

### El Problema de la Explosión Combinatoria

Al calcular la rugosidad macroscópica iterada sobre un acúmulo de notas formantes con el algoritmo matricial par-a-par de Sethares, la magnitud bruta latente en el vector DIC en $\mathbb{R}^{12}$ no escala de forma proporcionalmente lineal al número de notas de un acorde, sino polinomialmente (i.e. de orden $\mathcal{O}(K^2)$, donde $K$ es la cardinalidad en voces). Una pentáda (5 notas) irradia desproporcionadamente mayor voltaje absoluto de interferencia fisiológica total que su tétrada recesiva fundacional (4 notas). 

Si cruzamos sin normalizar las nubes de pentadas y de díadas sobre un mismo hiperespacio topológico, el racimo super-denso orbitará distorsionadamente distante en el vacío $\mathbb{R}^{12}$, segregando clústeres artificiales cimentados exclusiva y trivialmente por número de voces (cardinalidad pura), eclipsando su color, cualidad tímbrica y huella acústica cromática.

### Proyección Geométrica al Simplex Compositivo $\Delta^{11}$

Para corregir esta asimetría, la inyección geométrica original es compelida mediante operaciones algebraicas compositivas. El vector crudo o *Raw* $\vec{v}$ de rugosidad debe mapearse invariablemente hacia un volumen compacto universal representativo: el Simplex Unitario Euclidiano Estándar de 11 dimensiones libre de parámetros espúreos:

$$ \Delta^{11} = \left\{ (p_1, \dots, p_{12}) \in \mathbb{R}^{12} \, \middle| \, \sum_{k=1}^{12} p_k = 1, \quad p_k \geq 0 \, \forall k \right\} $$

La proyección ortogonal normalizadora mediante norma de Manhattan $L^1$ absorbe explícitamente el tamaño bruto escalar $N$:

$$ p_k = \frac{v_k}{\sum_{j=1}^{12} v_j} $$

Sobre este Simplex paramétrico, el foco algorítmico se desplaza imperativamente desde «¿cuánta disonancia total absoluta aloja el intervalo acústico?» (fenómeno regido por cardinalidad biológica pura) hacia el axioma diferencial más sutil: «¿se distribuye y conforma la masa de energía fisiológica aditiva preferencialmente sobre terceras, tritonos o séptimas menores?» permitiendo al algoritmo de modelado clasificar texturas densas como variaciones idénticas de texturas escuetas de baja voz (*voice-leading* relacional).

### Suavizado Gaussiano y Tolerancias de Clasificación Categórica

La formalización previa trata a los receptáculos del vector DIC de forma infinitamente esbelta y ortogonal aislando rígidamente al intervalo 3 del intervalo 4. Psicológicamente, la cognición sonora obedece al paradigma de "Percepción Categórica": el cerebro fusiona inexactitudes y deficiencias de micro-tono hacia la clase semitonal diatónica más próxima pre-concebida en la memoria cultural. 

En emulación profunda de este enmascaramiento neural cognitivo, la proyección al Simplex puro puede reforzarse transformando vectorialmente el espacio crudo bajo un operador de convolución borrosa (Kernel Suavizador o `simplex_smooth`). Un suavizado paramétrico Gaussiano transfiere una porción calculada $\alpha$ de la masa latente de rugosidad de un cesto a sus dos cestos colindantes adyacentes geométricamente:

$$ \tilde{p}_k = (1 - 2\sigma) p_k + \sigma \left(p_{k-1} + p_{k+1}\right) \pmod{12} $$

Este kernel isotrópico y estocástico inyecta tolerancia probabilística en el grafo del espacio musical, asimilando biológicamente las distancias topológicas entre acordes que prescriben rugosidades fronterizas que, bajo error perceptivo o modulación desafinada micro-interválica, oscilan difusamente su asunción identitaria.
