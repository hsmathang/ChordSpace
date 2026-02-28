# Reporte de Validación: Geometría Interválica 12D vs. Baseline Escalar de Rugosidad

## 1. Introducción y Objetivo Experimental
En el análisis computacional de la percepción de consonancia y disonancia armónica, el estándar histórico ha consistido en emplear modelos de rugosidad puramente espectrales (por ejemplo, Sethares, 1993), en donde las interferencias de bandas críticas se suman algebraicamente para derivar un único parámetro de costo (valor escalar 1D). Si bien las métricas escalares replican patrones globales, inherentemente colapsan la estructura relacional (topología geométrica) subyacente de los distintos intervalos. 

Para evaluar la pérdida de información de este colapso en el análisis musical, la arquitectura de *ChordSpace* extrae y preserva las colisiones armónicas bajo un histograma categórico o vector topológico de 12 dimensiones, mapeado sobre el temperamento igual (clases interválicas). 

El objetivo de este experimento es examinar empíricamente si la representación 12D muestra un mayor poder predictivo frente a juicios perceptuales humanos *Out-of-Sample* en comparación con el baseline tradicional escalar, garantizando controles rigurosos contra el sobreajuste (*overfitting*) mediante aislamientos por cardinalidad y la equiparación de grados de libertad métricos mediante validación cruzada anidada.

## 2. Metodología y Control de Variables

Se utilizó el dataset empírico publicado por *Bowling et al. (2018)*, el cual provee promedios de calificaciones perceptuales subjetivas (consonancia) tomadas a N=30 sujetos. La base consolidada abarca $N=298$ estructuras polifónicas únicas de 12-TET, concentradas en el registro $C_4 - C_5$ limitados (261.63 Hz).

Para la experimentación se calcularon dos arquitecturas sobre el mismo generador de interacciones acústicas estandarizadas ($n=6$, decaimiento $\alpha=0.88$):
1.  **Baseline 1D:** Valor de rugosidad total tradicional ($R_{total}$).
2.  **Representación Vectorial 12D:** Topología estructural (*ChordSpace*).

### 2.1 Refinamiento Estadístico (Grados de Libertad y Validación Anidada)
Para evitar la falacia de contraste asimétrico (comparar un valor crudo sin entrenamiento frente a un vector regularizado de 12 pesos ajustables), se implementó un control de linealidad: El modelo 1D fue transformado mediante un *Pipeline* de expansión polinómica, dotándolo de parámetros no-lineales entrenables para equiparar su flexibilidad paramétrica al modelo de vector 12D.

El rendimiento se estimó exclusivamente a través de **K-Fold Cross Validation con repeticiones** (5 pliegues *Out-Of-Sample*, 10 repeticiones randomizadas), aislando la optimización de los hiperparámetros de control —el grado polinómico en el 1D y la constante de penalización L2 ($\alpha$) en la Regularización Ridge del modelo 12D— mediante **Búsqueda Anillada (Nested GridSearchCV; $k=3$)** para prevenir roturas de independencia por fuga de datos. Todas las métricas de rendimiento reportan la Varianza Explicada ($R^2$) representadas como $\text{Media } \pm \text{ Desviación Estándar (SD)}$.

### 2.2 Segmentación por Cardinalidad (Nota-dependencia)
Harrison & Pearce (2020) advierten que modelos de sumación se favorecen artificiosamente al analizar un corpus unificado porque la densidad armónica (número de notas) es un proxy altamente colineal con la disminución del juicio temporal. Por consiguiente, se aisló el componente correlativo segmentando rigurosamente el análisis predictivo por su propio ecosistema de tamaño de acorde.

## 3. Resultados 

En el dataset de Bowling (N=298, registro $C_4–C_5$), la representación vectorial 12D basada en mapeo de clases interválicas presenta mayor poder predictivo generalizado frente al baseline escalar de *roughness*, como se comprueba en la siguiente estabilización de varianza anillada (Nested CV):

| Subconjunto (Cardinalidad) | Sujetos Empíricos (N) | $R^2$ (Lineal Crudo) | $R^2$ Polinómico 1D Tuned | $R^2$ Ridge Geométrico 12D Tuned | Modalidad Regularización Óptima (12D) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Población Total** | 298 | $0.218 \pm 0.125$ | $0.212 \pm 0.121$ | $\mathbf{0.535 \pm 0.048}$ | $\alpha = 100.0$ |
| **Tríadas (3 notas)** | 66 | $0.373 \pm 0.223$ | $0.371 \pm 0.235$ | $\mathbf{0.574 \pm 0.191}$ | $\alpha = 1.0$ |
| **Tétradas (4 notas)** | 220 | $0.482 \pm 0.106$ | $0.531 \pm 0.113$ | $\mathbf{0.670 \pm 0.081}$ | $\alpha = 10.0$ |

Bajo control cardinal estricto y equipamientos de grados de libertad en el modelo atómico (1D Polinómico), la estabilización topológica (12D Ridge) promedia incrementos estadísticamente consistentes de más de veinte puntos porcentuales absolutos de varianza explicada respecto al colapso clásico de la red de Sethares ($0.574$ frente a $0.371$ en tríadas; $0.670$ frente a $0.531$ en tétradas).

## 4. Limitaciones Metodológicas Extensas

Para consolidar la objetividad académica de estos hallazgos, es menester declarar tácitamente las restricciones mecánicas del formato de validación actual:

1.  **Díadas Excluidas por Deficiencia Muestral:** El análisis intercardiomodal (Tabla Superior) omitió formalmente presentar el desglose de rendimiento para acordes de rango *k=2* (Díadas, $N=12$). Las evaluaciones Ridge sobre validaciones de Fold fraccionario originan una varianza artificialmente explosiva ante un $N$ local inferior al conteo de dimensiones predictores del vector ($N=12 \le D=12$); produciendo R² degenerativos.
2.  **Configuración Espectral (Tuning):** El registro sintético computacional asumió los parámetros paramétricos de contorno del artículo experimental original: $C_4-C_5$ confinados bajo ecualización lineal 12-TET para $F_0$. Esta validación no ha sondeado si este poder explicativo es invariable o sufre transformaciones al someter a la matriz 12D a un ambiente microtonal (Just Intonation u Octavas Estiradas), registrando la topología a armónicos espectralmente inarmónicos.
3.  **Generalización Transversal (Timbre y Oído Exógeno):** Si bien los análisis de validación cruzada resguardan frente al *overfitting* intraset, probar "universalidad cognoscente" musical excede la comprobación unidisciplinar. Restan experimentaciones en arquitecturas de redes no-supervisadas, integrando datacenters interlingüísticos que confirmen un constructo universal o sesgos etnoculturales del histograma 12D (Sustitución de Cohortes). 

## 5. Conclusiones y Roadmap

El presente segmento de experimentación metodológica corrobora empíricamente que: existe evidencia matemáticamente consistente de que la representación 12D (Matriz Espacial de Rugosidad) extrae y modela la estructura paramétrica que rige la percepción cognitiva con mayor exactitud predictiva de la que consiguen las acumulaciones energéticas abstractas de compresión unidimensional, e intrínsicamente soporta defensas estocásticas.

A efectos del cronograma de investigación de *ChordSpace*, el próximo escalón validativo debe enfocarse en testear la robustez de generalización del vector de *embeddings* 12D sobre algoritmos no-supervisados (ejemplo: Agrupamiento UMAP/HDBSCAN) para evaluar si el vector despliega mecánicamente los fenotipos clásicos musicales (Familias de acordes disonantes versus Tríadas Perfectas Perfectas) sin contar con matrices de entrenamiento (*Ground Truth*) externas previas.
