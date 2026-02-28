## 2.1.2 Fisiología de la Cóclea y el Ancho de Banda Crítico (ERB)

Desde una perspectiva fisiológica, el sistema auditivo humano no funciona como un simple micrófono, sino como un analizador de espectro biológico impulsado por las propiedades mecánicas de la membrana basilar ubicada dentro de la cóclea. Esta estructura cónica exhibe una respuesta tonotópica altamente selectiva: su base (estrecha y rígida) resuena ante las altas frecuencias, mientras que su ápice (ancho y flexible) es sensible a las bajas frecuencias.

A partir de este comportamiento geomecánico, se formaliza el concepto de **Ancho de Banda Crítico** (*Critical Bandwidth*, CB o *Equivalent Rectangular Bandwidth*, ERB), definido como el ancho de banda efectivo en el cual se dispersa la energía de un tono puro a lo largo del tejido de la membrana basilar. Matemáticamente, la cóclea puede modelarse como un banco continuo de filtros pasabanda asimétricos superpuestos. Si las frecuencias de dos estímulos sonoros concurrentes inciden dentro del mismo ancho de banda crítico, sus patrones de excitación espacial en la membrana se superponen significativamente, impidiendo que el mecanismo neural los resuelva como componentes discretos. 

Basándose en estos límites de resolución, Plomp y Levelt demostraron empíricamente que las interferencias auditivas (disonancia sensorial) no están parametrizadas por una diferencia estática en hercios, sino que dependen directamente de la dimensión de este filtro biológico dependiente de frecuencia. 

### La Dicotomía Matemática: Procesamiento Logarítmico vs. Interferencia Lineal Local

La modelización fundamental de la música a través de parámetros numéricos se enfrenta a una de las dualidades estructurales más acusadas en psicoacústica profunda: la altura tonal se codifica y conceptualiza como un epifenómeno macroscópico **logarítmico**, mientras que la disonancia sensorial es consecuencia subyacente de interferencias cinemáticas estríctamente **lineales**.

#### Percepción Logarítmica del Tono
Conforme a la ley general psicofísica de Weber-Fechner, el cerebro humano codifica las magnitudes de estímulos amplios de manera logarítmica para incrementar el rango dinámico biológico. En acústica musical, el tono fundamental iterado ("pitch") corresponde a esta clase de función: percibimos psicológicamente como "equivalentes" aquellas frecuencias que preservan una misma proporción geométrica $f_2/f_1$, haciéndonos independientes de sus distancias aritméticas discretas. 

Matemáticamente, definimos la medida tradicional de la distancia melódica (el intervalo en `cents`) mediante el logaritmo en base 2 de su fracción de hercios:

$$ I_{cents} = 1200 \cdot \log_2\left(\frac{f_2}{f_1}\right) $$

Bajo esta transformación al continuo unidimensional $I$, un incremento de $200 \to 300$ Hz genera el mismo grado de tensión escalar topológica musical que uno de $400 \to 600$ Hz (ambos $\approx 702$ cents, proyectados como la *quinta justa*).

#### Interferencia Auditiva Lineal (Batimientos)
Sin embargo, la interferencia destructiva de dos de dichas excitaciones cercanas (el origen de la "disonancia sensorial") se rige netamente por la diferencia aritmética $|\Delta f|$, originada por las identidades elementales del principio de superposición en el álgebra periódica:

$$ \cos(2\pi f_1 t) + \cos(2\pi f_2 t) = 2 \cos(2\pi f_m t) \cos(\pi \Delta f t) $$

donde la frecuencia portadora es la media aritmética de ambas oscilaciones $f_m = \frac{f_1 + f_2}{2}$ y el par modulador de amplitud (la frecuencia fundamental de *batimiento*) es la diferencia estricta $\Delta f = |f_1 - f_2|$.
Esta tasa de batimiento moduladora opera estrictamente en términos lineales (hercios absolutos). 

#### El Conflicto en el Espacio Tonal
El ensanchamiento o angostamiento relativo de la banda crítica es el núcleo de este conflicto asimétrico. El Ancho de Banda Crítico de la cóclea no es constante: en frecuencias bajas crece con la asimetría de forma poco empinada, abarcando espectros proporcionales más amplios del logaritmo (de media $\approx 100$ Hz por debajo de 500 Hz), pero crece de forma proporcional relativa para frecuencias altas (aproximadamente un $20\%$ de la frecuencia central $f_{c}$).

Dado que un intervalo musical impone un ratio logarítmico invariante $f_2/f_1$, la diferencia de sustracción simple $|\Delta f| = |f_2 - f_1|$ se aproxima a cero conformemente transponemos su dominio $I_{cents}$ a conjuntos compactos de decaimiento en hercios graves. Por ejemplo, una *tercera mayor* en registros agudos ($1000$ Hz vs $1250$ Hz) nos arroja una distancia lineal de $\Delta f = 250$ Hz; suficiente para exceder el umbral de su ancho de banda crítico local de la cóclea en esas frecuencias resolviendo una armonía consonante y transparente. 
Inversamente, el mismo objeto topológico en registro de sub-graves ($100$ Hz y $125$ Hz) engendra una penalidad de interferencia estrecha de $\Delta f = 25$ Hz. Esta ventana de colisión incide desastrosamente cerca del valor máximo de perturbación $\sim 25\%$ del ancho de banda crítico respectivo en esa porción biológica particular, produciendo en el receptor una señal caótica, densamente difuminada y descrita comúnmente como perturbación de **Rugosidad** (*Roughness*).
