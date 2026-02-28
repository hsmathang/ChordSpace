# Iteración 1: Acústica Básica y Psicoacústica

## Fundamentos Psicoacústicos: Interferencia, Batimientos y Bandas Críticas
Para modelar matemáticamente la percepción musical, es imperativo comprender primero cómo el sistema auditivo periférico procesa las señales acústicas. La audición humana no es un simple micrófono de banda ancha, sino que opera mediante una transformación de frecuencia a espacio, analizando el espectro de la señal acústica de manera análoga a una Transformada de Fourier.

### Transducción Biológica y Análisis Espectral (La Membrana Basilar)
Desde una perspectiva física, el sonido es una onda de presión que se propaga a través de un medio elástico. Cuando esta onda ingresa al oído interno, excita la cóclea, un conducto lleno de fluido que contiene la membrana basilar.

La membrana basilar actúa como un analizador biológico de espectro. Sus propiedades mecánicas (estrecha y rígida en la base, ancha y flexible en el ápice) varían a lo largo de su longitud, lo que genera una resonancia dependiente de la posición. Las altas frecuencias excitan la base de la membrana, mientras que las bajas frecuencias excitan el ápice. Este mapeo continuo se conoce como **organización tonotópica** (una transformación matemática de frecuencia a espacio) y permite modelar la cóclea como un banco de filtros paso banda superpuestos.

La señal mecánica es luego transducida en impulsos neuronales por las células ciliadas, un proceso que involucra una rectificación de media onda (no linealidad) que preserva las fluctuaciones de la envolvente de la señal.

### Interferencia de Ondas y la Teoría de Helmholtz
Cuando dos ondas sinusoidales puras se emiten simultáneamente, sus fases relativas y frecuencias determinan un patrón de interferencia. Si las frecuencias son idénticas, ocurre interferencia constructiva o destructiva. Si las frecuencias son diferentes pero cercanas, se produce el fenómeno de los batimientos.

Matemáticamente, la superposición de dos sinusoides de igual amplitud y frecuencias $f_1$ y $f_2$ se describe mediante la identidad trigonométrica:
$$\cos(2\pi f_1 t) + \cos(2\pi f_2 t) = 2\cos(2\pi f_m t)\cos(\pi \Delta f t)$$
donde $f_m = (f_1 + f_2) / 2$ es la frecuencia portadora promedio, y $\Delta f = |f_1 - f_2|$ es la diferencia de frecuencias.

El resultado es percibido biológicamente como un solo tono de frecuencia $f_m$ cuya amplitud está modulada por una envolvente de frecuencia $\Delta f$. En 1863, **Hermann von Helmholtz** fue el primero en proponer que esta interferencia física es la base de la percepción de la disonancia sensorial. Según Helmholtz, si $\Delta f$ es pequeña (e.g., 1 a 5 Hz), el oído percibe una ondulación lenta y agradable en la sonoridad. Sin embargo, si la fluctuación de amplitud es rápida (típicamente entre 20 y 30 Hz), el sistema auditivo es incapaz de seguir la modulación suavemente, lo que resulta en una sensación áspera y desagradable denominada **rugosidad** (roughness). Para Helmholtz, la disonancia de un intervalo complejo no es más que la suma de las rugosidades generadas por las interferencias (batimientos primarios) entre todos los pares de parciales de los sonidos concurrentes.

### El Concepto de Banda Crítica
La teoría de Helmholtz postulaba que la máxima rugosidad ocurría a una diferencia constante de aproximadamente 33 Hz. Sin embargo, la fisiología de la membrana basilar exige una generalización topológica de este concepto. Dado que la membrana opera como un banco de filtros espaciales, la interferencia no depende de una diferencia de hercios constante, sino del ancho de banda del filtro biológico excitado.

A esto se le denomina **Banda Crítica** (*Critical Band*). Formulada inicialmente por investigadores como Zwicker en 1957, la banda crítica representa el ancho de banda efectivo para la dispersión de energía de un tono puro a lo largo de la membrana basilar. Si las frecuencias de dos tonos, $f_1$ y $f_2$, caen dentro de la misma banda crítica, sus patrones de excitación espacial en la membrana basilar se superponen significativamente, causando interferencia mecánica directa.

El ancho de la banda crítica $b(\bar{f})$ no es constante: es aproximadamente constante (unos 100 Hz) para frecuencias bajas (< 500 Hz), y crece de forma casi proporcional (aprox. 20% de la frecuencia central) para las altas frecuencias. Una aproximación analítica común para el ancho de banda crítico en Hz respecto a la frecuencia central $\bar{f}$ es:
$$b(\bar{f}) \approx 0.003\bar{f}^{1.47} + 90$$
(basado en los datos de Zwicker, Flottorp y Stevens). Alternativamente, se modela mediante el Ancho de Banda Rectangular Equivalente (ERB) de Moore y Glasberg.

### El Modelo de Plomp y Levelt: Disonancia Sensorial Continua
En 1965, **R. Plomp y W. J. M. Levelt** refinaron la teoría de Helmholtz al vincular empíricamente el grado de rugosidad directamente con el concepto de banda crítica, creando un puente formal entre la física de la membrana basilar y la disonancia perceptual.

A través de experimentos rigurosos con pares de tonos puros, Plomp y Levelt demostraron que:
1. La disonancia máxima no ocurre a una diferencia fija de hercios, sino invariablemente cuando la diferencia de frecuencias ($\Delta f$) es aproximadamente el 25% del ancho de banda crítico en esa región espectral.
2. La rugosidad desaparece (el intervalo se vuelve consonante) cuando la diferencia de frecuencias excede el ancho de la banda crítica, momento en el cual el sistema auditivo puede resolver (separar) las frecuencias analíticamente como dos tonos distintos sin interferencia superpuesta en la membrana basilar.

Geométrica y matemáticamente, los hallazgos de Plomp y Levelt permitieron parametrizar la "curva de disonancia" como una función continua y diferenciable que depende de la diferencia de frecuencias escalada por el ancho de la banda crítica. Este modelo es la piedra angular que permite a los matemáticos y acústicos computar la topología de la rugosidad total de un acorde sumando las interacciones par-a-par de todos los componentes espectrales sobre el espacio de frecuencias.

---

### Referencias BibTeX

```bibtex
@book{helmholtz1863lehre,
  author    = {von Helmholtz, Hermann L. F.},
  title     = {Die Lehre von den Tonempfindungen als physiologische Grundlage f{\"u}r die Theorie der Musik},
  year      = {1863},
  publisher = {Verlag F. Vieweg \& Sohn},
  address   = {Braunschweig}
}

@article{zwicker1957critical,
  author    = {Zwicker, Eberhard and Flottorp, Gordon and Stevens, Stanley S.},
  title     = {Critical band width in loudness summation},
  journal   = {The Journal of the Acoustical Society of America},
  volume    = {29},
  number    = {5},
  pages     = {548--557},
  year      = {1957},
  publisher = {Acoustical Society of America}
}

@article{moore1983suggested,
  author    = {Moore, Brian C. J. and Glasberg, Brian R.},
  title     = {Suggested formulae for calculating auditory-filter bandwidths and excitation patterns},
  journal   = {The Journal of the Acoustical Society of America},
  volume    = {74},
  number    = {3},
  pages     = {750--753},
  year      = {1983},
  publisher = {Acoustical Society of America}
}

@article{plomp1965tonal,
  author    = {Plomp, Reinier and Levelt, Willem J. M.},
  title     = {Tonal consonance and critical bandwidth},
  journal   = {The Journal of the Acoustical Society of America},
  volume    = {38},
  number    = {4},
  pages     = {548--560},
  year      = {1965},
  publisher = {Acoustical Society of America}
}
```

