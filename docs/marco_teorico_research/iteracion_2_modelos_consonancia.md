# Iteración 2: Modelos de Consonancia/Rugosidad

## Cuantificación de la Disonancia Sensorial: El Modelo de Sethares
Si bien el modelo de Plomp y Levelt (1965) estableció la relación cualitativa y empírica entre la banda crítica y la rugosidad, su formulación requería una parametrización rigurosa para ser aplicable a espectros complejos arbitrarios. William A. Sethares (1993, 2005) resolvió este problema formulando un modelo algebraico continuo que permite calcular la disonancia total de una señal a partir de su descomposición en series de Fourier.

### Descomposición Espectral y Definición del Espacio
Desde la perspectiva del análisis armónico, cualquier señal acústica periódica (o cuasi-periódica) puede ser descompuesta mediante la Transformada Discreta de Fourier (DFT) en una suma de ondas sinusoidales simples, denominadas parciales.

Matemáticamente, definimos el espectro discreto de un sonido $F$ como un conjunto finito de $N$ componentes espectrales:
$$F = \{(f_i, v_i)\}_{i=1}^N$$
donde $f_i \in \mathbb{R}^+$ representa la frecuencia del $i$-ésimo parcial (típicamente ordenados tal que $f_1 < f_2 < \dots < f_N$) y $v_i \in \mathbb{R}^+$ representa su amplitud (relacionada con la sonoridad o energía de esa componente).

La premisa central del modelo de Sethares es el principio de aditividad: la disonancia sensorial (rugosidad) total de un espectro complejo $F$ se define como la suma de las disonancias generadas por las interferencias (batimientos) de todas las posibles parejas de parciales $(f_i, f_j)$.

### Parametrización de la Curva Diádica de Plomp-Levelt
Para modelar la interacción entre dos sinusoides puros de frecuencias $f_1$ y $f_2$ (con $f_1 < f_2$), Sethares buscó una función continua $d(x)$ que dependiera de la diferencia de frecuencias $x = f_2 - f_1$ y que cumpliera con las restricciones topológicas observadas empíricamente por Plomp y Levelt:
1. $d(0) = 0$ (el unísono perfecto no tiene batimientos).
2. $\lim_{x\to\infty} d(x) = 0$ (frecuencias muy alejadas no interfieren en la membrana basilar).
3. Posee un único máximo global en $x^* > 0$ (el punto de máxima rugosidad).

Sethares parametrizó esta curva utilizando la diferencia de dos funciones exponenciales decrecientes:
$$d(x) = e^{-b_1 x} - e^{-b_2 x}$$
Mediante un ajuste de mínimos cuadrados sobre los datos empíricos originales, Sethares determinó que los parámetros óptimos que controlan las tasas de subida y caída de la función son $b_1 = 3.5$ y $b_2 = 5.75$. Calculando la derivada e igualando a cero ($d'(x)=0$), el máximo teórico de esta función se encuentra alrededor de $x^* \approx 0.24$.

### Dependencia Frecuencial (Escalamiento de la Banda Crítica)
La ecuación anterior asume una banda de interferencia constante. Sin embargo, el ancho de la banda crítica del oído humano se expande a medida que aumenta la frecuencia base. Para incorporar la asimetría del sistema auditivo periférico, Sethares introdujo un factor de escalamiento $s(f_1)$ que dilata o comprime el eje $x$ en función de la frecuencia del parcial inferior $f_1$.

Este factor de estiramiento se define de manera que el máximo de la curva exponencial ($x^* = 0.24$) coincida siempre con el 25% del ancho de banda crítico para cualquier frecuencia $f_1$:
$$s(f_1) = \frac{x^*}{s_1 f_1 + s_2}$$
Los valores $s_1 = 0.021$ y $s_2 = 19$ fueron obtenidos empíricamente para garantizar que el término del denominador aproximara de forma lineal el comportamiento de la banda crítica humana.

### Ponderación de Amplitudes y Enmascaramiento
La interferencia entre dos ondas también depende críticamente de sus energías. Si uno de los parciales tiene una amplitud muy cercana a cero, no puede generar una modulación de amplitud audible sobre el otro.

En su formulación original (1993), Sethares propuso modelar esta ponderación de amplitud como el producto $v_1 v_2$. Sin embargo, en revisiones posteriores del modelo (Sethares 2005), se demostró que un modelo más preciso psicoacústicamente para capturar el cuello de botella del enmascaramiento auditivo es tomar el mínimo de ambas amplitudes:
$$v_{12} = \min(v_1, v_2)$$
Esto refleja el hecho físico de que la profundidad del batimiento está estrictamente limitada por el parcial más débil que logra "sobrevivir" a la interferencia del más fuerte.

### La Función de Disonancia Total
Combinando la curva base, el factor de escalamiento espacial y la ponderación de amplitudes, se obtiene la función de disonancia diádica $d(f_1, f_2, v_1, v_2)$ para cualquier par de parciales:
$$d(f_i, f_j, v_i, v_j) = \min(v_i, v_j) \left[ e^{-b_1 \cdot s(f_i) \cdot (f_j - f_i)} - e^{-b_2 \cdot s(f_i) \cdot (f_j - f_i)} \right]$$

Finalmente, dada la descomposición en series de Fourier de un espectro completo $F=\{(f_i, v_i)\}_{i=1}^N$, la disonancia intrínseca o rugosidad total del sonido $D_F$ se formaliza como la doble sumatoria sobre la matriz triangular superior de todas las interacciones espectrales posibles:
$$D_F = \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} d(f_i, f_j, v_i, v_j)$$
(Nota: Sethares a menudo denota esto como $\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N$, lo cual es matemáticamente equivalente dado que la disonancia de un parcial consigo mismo es $d_{ii}=0$ y la matriz de interacciones es simétrica).

### Implicaciones Geométricas del Modelo
Desde una perspectiva matemática, esta sumatoria transforma el cálculo de la consonancia en un problema topológico continuo. Si fijamos un espectro $F$ y lo transponemos por un intervalo continuo $\alpha$ (creando el espectro transpuesto $\alpha F = \{(\alpha f_i, v_i)\}$), la función de disonancia cruzada $D_F(\alpha)$ generará una curva parametrizada por $\alpha$. Los mínimos locales de esta función revelan los puntos de estabilidad armónica geométrica donde las "fricciones" de los parciales se anulan.

---

### Referencias BibTeX

```bibtex
@article{sethares1993local,
  author    = {Sethares, William A.},
  title     = {Local consonance and the relationship between timbre and scale},
  journal   = {The Journal of the Acoustical Society of America},
  volume    = {94},
  number    = {3},
  pages     = {1218--1228},
  year      = {1993},
  publisher = {Acoustical Society of America}
}

@book{sethares2005tuning,
  author    = {Sethares, William A.},
  title     = {Tuning, Timbre, Spectrum, Scale},
  edition   = {2nd},
  publisher = {Springer-Verlag},
  address   = {London},
  year      = {2005},
  isbn      = {978-1-85233-797-1}
}

@article{weisser2013minimum,
  author    = {Weisser, Stephanie and Lartillot, Olivier},
  title     = {Investigating the relationship between timbre and roughness: A psychoacoustical model},
  journal   = {Journal of New Music Research},
  volume    = {42},
  number    = {4},
  pages     = {321--337},
  year      = {2013},
  publisher = {Taylor & Francis}
}
```

