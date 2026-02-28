## 2.4 Reducción Dimensional y Disposición Topológica Global

### 2.4.1 Distancias Vectoriales vs Distancias sobre Distribuciones de Probabilidad

Dotados del formalismo vectorial dicromático $\vec{v} \in \mathbb{R}^{12}$, el análisis de vecindad entre dos acordes $\mathcal{C}_1$ y $\mathcal{C}_2$ demanda definir una función de distancia estricta u operador de similitud $D(\vec{v}_1, \vec{v}_2)$ que codifique la disimilitud perceptivo-cognitiva que experimenta el sistema auditivo al transitar entre ambos eventos armónicos.

La elección subyacente de la métrica $D(\cdot, \cdot)$ condiciona inexorablemente la topología del *manifold* musical y decreta las órbitas melódicas funcionales predichas por el modelo.

#### La Falla de la Distancia Euclidiana en Vectores Perceptuales
Trivialmente, el instinto analítico en sistemas embebidos continuos decanta por la devaluación mediante la métrica hiper-Euclidiana estándar (Norma $L^2$):
$$ D_E(\vec{v}_1, \vec{v}_2) = \sqrt{\sum_{k=1}^{12} (v_{1,k}  -v_{2,k})^2} $$
Sin embargo, suponer la ortogonalidad y homogeneidad lineal implícita en la Norma de Frobenius o $L^2$ es frecuentemente defectuoso sobre representaciones psicoacústicas densas. En el vector $\mathbb{R}^{12}$, una alteración sutil de amplitud por una nota agregada a una tríada es sancionada aditivamente a un ratio constante con respecto al origen de la métrica por $D_E$. Esto induce una penalización combinatoria desorbitada (la maldición de la cardinalidad), separando irrealmente acordes emparentados pero con distinto recuento de estratos armónicos (ej., una tétrada frente a su tríada generatriz) a un grado inasimilable en la esfera logarítmica, que empíricamente debiesen conservar extrema proximidad por contener sustancialmente el mismo vector latente modal-interferente de rugosidad.

#### Divergencias en el Simplex (Jensen-Shannon y Hellinger)
La táctica resolutiva, validada en el marco matemático de *Data Science* e Inteligencia Artificial subsimbólica, impone restringir y abstraer los vectores de energía residual asimétricos para que adopten la geometría de un **Simplex Estadístico Unitario** $\Delta^{11}$. Dividiendo coercitivamente el vector entre su masa $L^1$, transformamos la inyección cruda $\vec{v}$ en una **Distribución Discreta de Probabilidades** $P = \vec{v}_k / ||\vec{v}||_1$, sumando unitariamente 1.

Con la información encapsulada en forma de un espectro categórico $P, Q \in \Delta^{11}$, descartamos métricas espaciales tradicionales y adoptamos medidas informacionales entrópicas y asimétricas:

1.  **Divergencia de Jensen-Shannon ($JSD$):**
    Una versión suavizada, acotada (entre 0 y 1, si se usa $\log_2$) y fundamentalmente simétrica de la divergencia direccional asimétrica originaria de Kullback-Leibler ($D_{KL}$). Su definición parte del cálculo probabilístico entrópico hacia el promedio central $M = \frac{1}{2}(P+Q)$:
    $$ JSD(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M) $$
    Para que the $JSD$ satisfaga los axiomas rigurosos de una distancia métrica (identidad de indiscernibles, simetría rigurosa y desigualdad de triángulo), se asevera que la **Raíz Cuadrada Exacta de la Divergencia de Jensen-Shannon** proporciona el isomorfismo métrico buscado sobre el Simplex unitario: $D_{JSD}(P, Q) = \sqrt{JSD(P \parallel Q)}$.

2.  **Distancia Hellinger ($H$):**
    Una métrica probabilistica estrictamente euclídea derivada de proyecciones sobre sub-variedades esféricas. Responde extremadamente eficaz para contrarrestar los *outliers* ruidosos en series difusas de características marginales de la rugosidad psicoacústica. Opera evaluando la masa entre las raíces cuadradas aisladas de las componentes categóricas:
    $$ D_H(P, Q) = \frac{1}{\sqrt{2}} \sqrt{ \sum_{k=1}^{12} (\sqrt{p_k} - \sqrt{q_k})^2 } $$

El empleo de distancias métricas orientadas a distribuciones ($JSD$ y Hellinger) sobre la huella interválica de un acorde preserva y enaltece el isomorfismo local de características, capturando la "forma de la constelación sónica" por sobre el volumen aditivo absoluto del mismo. La regularidad abstracta que emana logra codificar efectivamente cuándo dos acordes (de disímiles cardinalidades) originan el mismo perfil atractor fisiológico en la cóclea.
