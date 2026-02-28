# 2.1 Fundamentos Acústicos y Perceptivos del Sonido Musical (Boca del embudo)

## 2.1.1 Física Acústica y Descomposición Espectral (Series de Fourier)

Para modelar rigurosamente un sonido complejo desde una perspectiva psicoacústica y matemática, consideramos la señal acústica como una función $F(t)$ que representa la variación de presión en el tiempo. Si asumimos que $F(t)$ es una función periódica (o cuasi-periódica en su estado estacionario) con periodo $T$ y frecuencia fundamental $f = 1/T$, esta pertenece al espacio de funciones de cuadrado integrable y admite una descomposición ortogonal mediante la Serie de Fourier.

Matemáticamente, la señal se expande como una combinación lineal de funciones base sinusoidales:

$$ F(t) = \frac{a_0}{2} + \sum_{n=1}^\infty \left[ a_n \cos(2n\pi f t) + b_n \sin(2n\pi f t) \right] $$

donde los coeficientes de Fourier vienen dados por las proyecciones estándar sobre la base para $n \geq 1$:

$$ a_n = \frac{2}{T} \int_0^T F(t) \cos(2n\pi f t) dt, \quad b_n = \frac{2}{T} \int_0^T F(t) \sin(2n\pi f t) dt $$

En el análisis de la percepción musical, esta transformación del dominio del tiempo al dominio de la frecuencia es crucial, ya que el sistema auditivo periférico (específicamente la membrana basilar en la cóclea) actúa biológicamente como un analizador de espectros.

Cada término $n$ de la serie, con frecuencia $f_n = n \cdot f$, se denomina **parcial** o **armónico**. Cuando el espectro de un sonido está constituido exclusivamente por frecuencias que son múltiplos enteros exactos de la frecuencia fundamental ($f, 2f, 3f, \ldots$), se dice que el sonido posee una *estructura armónica*. Este es el caso idealizado para aproximar instrumentos de cuerda y de viento de la armonía tradicional occidental.

El concepto perceptivo de **timbre** (el "color" del sonido que permite distinguir dos instrumentos tocando la misma nota fundamental) se correlaciona directamente con la distribución de la energía en este espectro de Fourier. Aunque cada componente sinusoidal posee una fase (codificada implícitamente en $a_n$ y $b_n$), el sistema auditivo humano es principalmente sensible a la diferencia de magnitud de esas excitaciones. 

Por tanto, un sonido complejo se caracteriza matemáticamente mediante su espectro de magnitud discreto, expresado como un conjunto de pares frecuencia-amplitud:

$$ \mathcal{F} = \{ (f_i, v_i) \}_{i=1}^N $$

donde la amplitud del $i$-ésimo armónico corresponde a $v_i = \sqrt{a_i^2 + b_i^2}$. 

En este marco, el timbre armónico queda formalmente definido por una función envolvente espectral continua $\zeta: \mathbb{R} \to \mathbb{R}^+$ que acota y determina las amplitudes $v_i = \zeta(f_i)$ de la serie armónica. Si la frecuencia fundamental $f$ se transpone por un escalar $\alpha > 0$, las nuevas frecuencias serán $\alpha f_i$, pero sus amplitudes continuarán determinadas por la evaluación de la misma envolvente espectral $\zeta(\alpha f_i)$, proporcionando así un modelo de identidad (y disonancia par-a-par) evaluable sistemáticamente bajo transformaciones afines en el espacio de frecuencias.
