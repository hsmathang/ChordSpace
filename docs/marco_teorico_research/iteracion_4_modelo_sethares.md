## 2.2 El Modelo Computacional de Rugosidad

Establecida la causa fisiológica de la disonancia sensorial en el ancho de banda crítico de la cóclea, y diferenciada firmemente del constructo cultural de la disonancia musical, se requiere una formalización matemática explícita aplicable a un modelo computacional.

### 2.2.1 Parametrización Analítica de Sethares (1993)

William A. Sethares (1993) propuso una modelización algorítmica y continua que aproxima las curvas empíricas de disonancia sensorial discreta previamente caracterizadas por Plomp y Levelt. Su formulación captura algorítmicamente el comportamiento de la curva de interferencia coclear (arranque nulo, ascenso abrupto a máxima penalización y relajamiento asintótico paulatino) para la interacción de todo par de tonos puramente sinusoidales.

La función central de disonancia paramétrica, $d(x)$, para un par de armónicos separados por una variable de distancia en hercios $x = |f_2 - f_1|$, se conceptualizó inicialmente como la sustracción de dos decaimientos exponenciales:

$$ d(x) = e^{-ax} - e^{-bx} $$

#### Optimización y Minimización del Error
Para que este cascarón algorítmico abstrajera la data clínica real, Sethares procedió con un ajuste numérico, empelando un método de convergencia por minimización del gradiente del error cuadrático. Minimizando la diferencia entre la salida de esta función paramétrica $d(x)$ y los puntos de datos experimentales promedios recolectados por Plomp y Levelt, halló que las tasas de decaimiento óptimas que gobiernan la geometría de la disonancia son:

$$ a = 3.5, \quad b = 5.75 $$

Si derivamos analíticamente dicha función $d'(x)$ e igualamos a cero, el salto estacionario de divergencia o "punto de disonancia máxima teórica", denotado como $x^*$, se ubica en $0.2434$, haciendo un calce cuasi-perfecto con la asunción del $25\%$ ($\approx \frac{1}{4}$) del Ancho de Banda Crítico.

#### Ajustes de Escalamiento al Espectro Frecuencial y Amplitud

Sin embargo, la fórmula $d(x)$ anterior presupone inmutabilidad con respecto al registro acústico (pitch height), asumiendo falsamente amplitudes unitarias y un ancho de banda estacionario. Como se exploró en secciones precedentes, el filtro pasabanda coclear se ensancha a alta frecuencia. 

Para dotar al modelo de invariancia translacional bajo cualquier par de frecuencias dadas $f_1, f_2$ (con $f_1 < f_2$) portando magnitudes espectrales de excitación $v_1, v_2$, Sethares formuló finalmente un sistema de parámetros afines a la ecuación. La función calibrada a registro y fuerza queda modelada como:

$$ d(f_1, f_2, v_1, v_2) = v_{12} \left[ e^{-a \cdot s \cdot (f_2 - f_1)} - e^{-b \cdot s \cdot (f_2 - f_1)} \right] $$

En esta formulación rectora intervienen dos factores críticos de escalamiento:

1.  **El Factor de Contracción/Estiramiento Frecuencial ($s$):**
    $$ s = \frac{x^*}{s_1 \cdot f_1 + s_2} $$
    Donde se desliza e interpola el ancho del domino para asegurar empíricamente que la cúspide disonante incida matemáticamente siempre en proporción a la banda crítica de la frecuencia fundamental base $f_1$. A través del mismo protocolo de mínimos cuadrados contra la curva de Bark/ERB, se dedujeron las constantes anatómicas como $s_1 = 0.021$ y $s_2 = 19$.
2.  **Ponderación de Enmascaramiento y Amplitudes ($v_{12}$):**
    Originalmente definida ingenierilmente cruzando la energía como un producto punto de amplitudes de parcial $v_{12} = v_1 \cdot v_2$. Posteriormente (Sethares 2005), esta ponderación fue reconsiderada topológicamente, adoptando en su lugar la cota inferior $v_{12} = \min(v_1, v_2)$. Al emplear el operador `min`, la ecuación se restringe fielmente a modelar abstracciones del fenómeno biológico psicoacústico del *enmascaramiento simultáneo*, garantizando que un parcial tenue incidiendo contra un parcial muy denso no proyecte anomalías matemáticas de rugosidad inexistente.
