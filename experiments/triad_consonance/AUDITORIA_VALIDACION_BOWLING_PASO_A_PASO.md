# Auditoría paso a paso de la validación Bowling en ChordSpace

Este documento explica, de principio a fin, cómo se hizo la validación contra Bowling dentro del repositorio, pero escrito para ingenieros que sí conocen machine learning y validación estadística, aunque no conozcan ni la estructura del repo ni la parte psicoacústica. La idea es que se pueda auditar el experimento como si fuera cualquier pipeline de ML supervisado: primero se identifica el dataset, luego se reconstruyen las features, después se definen los modelos, luego se revisa el protocolo de validación y por último se inspeccionan las salidas numéricas y gráficas.

## 1. Qué problema resuelve esta validación

La pregunta experimental es simple si se traduce al lenguaje de ML. Tenemos una variable objetivo continua, `rating`, que representa la consonancia percibida por humanos en una escala aproximada de 1 a 4, y queremos predecirla a partir de dos representaciones distintas del mismo acorde. La primera representación es un único escalar, `scalar_roughness`, que colapsa toda la rugosidad del acorde en un solo número. La segunda es un vector de doce dimensiones, `v0` a `v11`, que reparte esa rugosidad por clases intervalares. La validación busca responder si el vector 12D generaliza mejor que el escalar 1D cuando ambos intentan predecir los juicios humanos fuera de muestra.

En términos puramente de ML, el experimento compara dos diseños de feature engineering sobre el mismo target. No está intentando descubrir si Ridge es “el mejor algoritmo posible” en abstracto. Está intentando medir si una representación más rica del acorde contiene información predictiva adicional respecto a la suma escalar tradicional.

## 2. De dónde sale el dataset supervisado

El punto de partida no es un dataset nativo del repositorio, sino una hoja de cálculo suplementaria del artículo de Bowling. El script que extrae esos datos es `experiments/triad_consonance/extract_bowling.py`. Ese script abre el archivo Excel `metodologia_temporal/pnas.1713206115.sd01.xlsx`, recorre las hojas `dyads`, `triads` y `tetrads`, identifica las columnas donde están los tonos del acorde y la media de rating humano, y exporta el resultado a `experiments/triad_consonance/bowling_data.csv`.

Ese CSV tiene una estructura mínima y limpia. La columna `k` indica la cardinalidad del acorde, es decir, si tiene 2, 3 o 4 notas. La columna `tones` guarda las alturas relativas del acorde respecto a la raíz, por ejemplo `0_4_7` para una tríada mayor en posición fundamental. La columna `rating` contiene el promedio empírico reportado por Bowling. Si se quiere auditar el dataset de entrada antes de mirar cualquier modelo, basta con abrir `bowling_data.csv` y verificar que el experimento trabaja sobre 298 acordes en total.

## 3. Cómo se transforman los acordes en features

El siguiente paso del pipeline es convertir cada acorde en variables predictoras. Esto lo hace `experiments/triad_consonance/run_bowling_model.py`, que toma `bowling_data.csv`, crea un objeto `Acorde` por fila y lo procesa con `ModeloSetharesVec`, definido en `pre_process.py`.

La traducción a lenguaje de ingeniería es la siguiente. Para cada acorde, el sistema enumera todos los pares de notas, calcula una contribución de rugosidad para cada par a partir del modelo espectral de Sethares, y asigna esa contribución a uno de doce bins según la clase intervalar correspondiente. El resultado es un histograma de longitud doce, no normalizado, donde cada coordenada representa cuánta rugosidad cayó en una familia intervalar concreta. Ese histograma es el vector 12D. La suma total de todas las contribuciones por pares es el escalar `scalar_roughness`.

Esto importa porque fija la relación entre ambos espacios de features. El baseline 1D y el modelo 12D no usan información distinta de origen. Usan el mismo cálculo psicoacústico, pero el baseline la colapsa a un único número y el modelo 12D la conserva repartida por estructura intervalar. Desde el punto de vista de auditoría, eso es una buena propiedad experimental, porque la comparación no cambia simultáneamente el target, el dataset y el origen del descriptor.

## 4. Qué archivo contiene la matriz final para modelado

La salida de `run_bowling_model.py` es `experiments/triad_consonance/bowling_results.csv`. Ese archivo es la tabla supervisada final sobre la que entrenan todos los modelos de validación. Las primeras columnas son `k`, `tones` y `rating`. Luego aparece `scalar_roughness` y después las doce columnas `v0` a `v11`.

Si se mira este archivo con ojos de ML, se ve enseguida que el vector es muy ralo. En una díada solo se activa un bin, porque solo existe un par de notas. En una tríada suelen activarse dos o tres bins. En una tétrada se activan entre tres y seis. Esa estructura dispersa es importante para entender por qué más adelante aparece Ridge como regularización lineal conservadora.

## 5. Qué modelos se comparan exactamente

En el repositorio hay dos niveles de análisis y conviene separarlos para no mezclar resultados. El primer nivel es el análisis preliminar, implementado en `experiments/triad_consonance/analyze_bowling_correlation.py`. Ese script define tres modelos. El primero es un baseline 1D lineal con `LinearRegression`, usando solo `scalar_roughness`. El segundo es un baseline 1D polinómico con `PolynomialFeatures(degree=3)` seguido de `LinearRegression`, pensado como control de grados de libertad. El tercero es el modelo 12D con `Ridge(alpha=1.0)`, usando las doce columnas `v0...v11`.

El segundo nivel es el análisis metodológicamente fuerte, implementado en `experiments/triad_consonance/analyze_bowling_nested_cv.py`. En ese script el baseline 1D lineal sigue siendo `LinearRegression`, el baseline 1D polinómico se convierte en un `Pipeline` con `PolynomialFeatures` y `LinearRegression`, y el modelo 12D usa `Ridge()`. La diferencia clave es que aquí no se fija el grado polinómico ni el `alpha` a priori, sino que ambos se sintonizan dentro de un esquema de validación anidada.

## 6. Cómo funciona la validación preliminar

La validación preliminar de `analyze_bowling_correlation.py` es muy fácil de auditar porque está escrita de forma directa. El script carga `bowling_results.csv`, separa `ratings`, `scalar_X` y `vector_X`, define `CV = KFold(n_splits=5, shuffle=True, random_state=42)` y luego usa `cross_val_predict` para generar predicciones fuera de muestra para cada observación.

La lógica estadística es la habitual en regresión supervisada. El dataset se divide en cinco folds. En cada iteración se entrena el modelo sobre cuatro folds y se predice el fold restante. Cuando termina el bucle, cada acorde tiene una predicción obtenida por un modelo que no vio ese acorde durante el entrenamiento. A partir de ese vector completo de predicciones OOS se calculan dos métricas globales: la correlación de Pearson entre `predicción` y `rating`, y el `R²` usando `r2_score(ratings, preds_oos)`.

El motivo por el cual este script es preliminar y no final es que el modelo 12D usa `Ridge(alpha=1.0)` fijo. Eso no invalida la comparación como control inicial, pero sí la deja metodológicamente más débil frente a una pregunta de auditoría sobre selección de hiperparámetros. Por eso el análisis final no debe apoyarse solo en este script.

## 7. Qué se grafica en la validación preliminar

La parte gráfica de `analyze_bowling_correlation.py` genera una figura de dos paneles. El panel izquierdo corresponde al modelo vectorial 12D con Ridge y muestra un scatter plot donde el eje x es el rating predicho fuera de muestra y el eje y es el rating humano real. El panel derecho hace exactamente lo mismo para el modelo escalar 1D lineal. En ambos casos se dibuja además la diagonal ideal `y = x`, que sirve como referencia visual de predicción perfecta.

Desde la perspectiva de auditoría, esa figura no es un artefacto decorativo sino una inspección visual del error de generalización. Permite ver si el modelo cubre razonablemente todo el rango de ratings, si hay sesgo sistemático hacia la media, si la nube de puntos se dispersa en torno a la diagonal y si la mejora del modelo 12D es visible más allá del valor agregado de `R²`. El script guarda esa figura como `experiments/triad_consonance/bowling_correlation_paper_hq.png` y `experiments/triad_consonance/bowling_correlation_paper_hq.pdf`.

Además, el mismo script genera `bowling_correlation_report.html`, que resume en una tabla las métricas OOS de los tres modelos y reporta las diferencias de `R²` entre el 12D y los dos baselines 1D. Ese HTML es útil como salida de inspección rápida, pero no añade un protocolo de validación más fuerte que el del propio script.

## 8. Qué números produce la validación preliminar

Al ejecutar `analyze_bowling_correlation.py` en el repositorio actual, el baseline 1D lineal produce `Pearson r = 0.468` y `R² = 0.218`. El baseline 1D polinómico de grado 3 produce `Pearson r = 0.456` y `R² = 0.204`. El modelo 12D con `Ridge(alpha=1.0)` produce `Pearson r = 0.767` y `R² = 0.588`.

Para un auditor de ML, la lectura correcta de esos números es que el vector 12D mejora claramente sobre el escalar 1D incluso en un protocolo relativamente simple. La lectura que no conviene hacer todavía es que `alpha=1.0` sea el mejor valor o que esos números sean la versión definitiva para redacción metodológica. Para eso está el segundo nivel de validación.

## 9. Cómo funciona la validación final con nested CV

La validación final está en `analyze_bowling_nested_cv.py` y es el script que realmente debe considerarse el respaldo técnico de tesis. La lógica aquí sí es la de una auditoría moderna de modelos supervisados con tuning. El bucle externo usa `RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)`. Esto significa que no se hace un único `5-fold`, sino cincuenta particiones de test en total, lo que permite estimar no solo un rendimiento promedio sino también su dispersión.

Dentro de cada split externo, el baseline 1D polinómico y el modelo 12D no se entrenan directamente. Se encapsulan en un `GridSearchCV` interno de tres folds. Para el baseline 1D se busca el mejor grado en la rejilla `2, 3, 4`. Para el modelo 12D se busca el mejor `alpha` en la rejilla `0.1, 1.0, 10.0, 100.0`. El score usado para optimizar ambos es `r2`. Luego, el mejor modelo interno se evalúa sobre el test fold externo correspondiente. Este diseño separa claramente tuning y evaluación, y evita fuga de datos durante la selección de hiperparámetros.

Desde el punto de vista de auditoría, este es el paso más importante de todo el pipeline. Si un revisor pregunta “cómo escogieron `alpha` y cómo evitaron overfitting en la búsqueda”, la respuesta está completa en este script. No se dejó `alpha` por defecto, no se eligió usando todo el dataset y no se informó el score del mejor modelo sobre el mismo fold con el que fue sintonizado.

## 10. Por qué también se segmenta por cardinalidad

El script anidado no solo evalúa el conjunto completo, sino también los subconjuntos con `k = 2`, `k = 3` y `k = 4`. La razón es muy natural para cualquier ingeniero de datos: la cardinalidad del acorde cambia la estructura combinatoria del feature space. Una díada tiene un solo par de notas; una tríada tiene tres; una tétrada tiene seis. Si se mezclan todos los acordes en un único pool, parte de la señal predictiva puede venir simplemente de diferencias de tamaño del objeto y no de la topología específica del descriptor.

La segmentación por `k` actúa entonces como un control de confusión estructural. Si el modelo 12D sigue ganando dentro del universo de tríadas y dentro del universo de tétradas, la ventaja no puede atribuirse solo a que el vector “cuenta” más energía total cuando hay más notas. Para ingeniería de ML, esto equivale a controlar una variable de estratificación que de otro modo estaría mezclando regímenes distintos del problema.

## 11. Qué números produce la validación final auditada

Al ejecutar `analyze_bowling_nested_cv.py` en el repositorio actual, el conjunto completo de 298 acordes produce `R² = 0.218 ± 0.125` para el baseline 1D lineal, `R² = 0.212 ± 0.121` para el baseline 1D polinómico y `R² = 0.535 ± 0.048` para el modelo 12D Ridge, con `alpha` modal `100.0`.

En tríadas, con `N = 66`, el baseline 1D lineal produce `0.373 ± 0.223`, el 1D polinómico `0.371 ± 0.235` y el 12D Ridge `0.574 ± 0.191`, con `alpha` modal `1.0`. En tétradas, con `N = 220`, el baseline 1D lineal produce `0.482 ± 0.106`, el 1D polinómico `0.531 ± 0.113` y el 12D Ridge `0.670 ± 0.081`, con `alpha` modal `10.0`.

Las díadas deben interpretarse como un caso patológico desde el punto de vista estadístico. Como solo hay doce ejemplos y el modelo 12D tiene doce predictores, el score tiene varianza enorme y pierde utilidad como evidencia de generalización. El propio repositorio lo reconoce como limitación metodológica y por eso las conclusiones serias se apoyan en el total, las tríadas y las tétradas.

## 12. Qué se grafica en la validación final

El script anidado no genera una figura análoga a la preliminar. Su salida principal es una tabla en consola con `R² mean ± std` por subconjunto y el `alpha` modal de Ridge. En otras palabras, la validación preliminar es la que aporta la visualización intuitiva de predicción OOS punto a punto, mientras que la validación final aporta el protocolo estadístico fuerte y los números agregados auditables.

Eso significa que, si se quiere presentar el experimento de forma honesta, conviene decir que el repositorio usa dos artefactos complementarios. La figura `bowling_correlation_paper_hq.png` ilustra visualmente la comparación OOS entre 1D y 12D. El script de nested CV, en cambio, proporciona la evaluación final con tuning interno y estimación de varianza por repetición. La figura no reemplaza al nested CV, y el nested CV no reemplaza la figura.

## 13. Por qué aparece Ridge en este problema

Para un equipo que sabe de ML pero no de psicoacústica, Ridge puede entenderse como una regularización lineal prudente para un feature space estructurado y relativamente pequeño. El modelo 12D no está tratando con embeddings densos de alta muestra, sino con un histograma de doce bins cuyo patrón de activación depende fuertemente de la cardinalidad y de la geometría interna del acorde. En tríadas, por ejemplo, cada observación activa solo dos o tres dimensiones en promedio. En ese contexto, Ridge reduce varianza y estabiliza coeficientes sin forzar selección dura de variables.

La interpretación correcta no es que el dataset muestre una multicolinealidad extrema de manual. Las correlaciones entre columnas no son tan altas como para justificar por sí solas una narrativa de colapso numérico. La justificación más exacta es que el problema combina muestra efectiva reducida por fold, predictoras estructuralmente dependientes y una hipótesis científica que quiere preservar todas las clases intervalares en el modelo, no eliminar algunas con Lasso.

## 14. Qué inconsistencias del repositorio debe conocer un auditor

Hay algunas diferencias entre código, gráficos y documentos que conviene registrar desde el inicio. El HTML generado por `analyze_bowling_correlation.py` habla de “5-fold CV estratificado”, pero el código usa `KFold`, no `StratifiedKFold`. En regresión continua eso no cambia la lógica básica del experimento, pero sí es una imprecisión de redacción que no debe repetirse en una auditoría formal.

También hay coexistencia de versiones metodológicas. Algunos documentos viejos del repo describen el baseline 1D lineal como si también fuera Ridge, o describen el modelo 12D con `alpha=1.0` fijo como si fuera la versión final. El código actual no hace eso en el análisis anidado. Por tanto, la referencia de auditoría debe ser siempre el comportamiento de `analyze_bowling_nested_cv.py`, no cualquier copia textual previa.

## 15. Cómo reproducir todo el pipeline de auditoría

La reproducción completa puede pensarse en cuatro ejecuciones secuenciales. Primero se extrae el dataset de Bowling desde el Excel original con `python extract_bowling.py`. Segundo, se reconstruyen las features con `python run_bowling_model.py`. Tercero, se ejecuta la validación preliminar con `python analyze_bowling_correlation.py`, que produce métricas OOS y la figura comparativa. Cuarto, se ejecuta la validación final con `python analyze_bowling_nested_cv.py`, que produce las métricas agregadas con repeated nested CV.

Si se trabaja desde `experiments/triad_consonance`, esos cuatro comandos bastan para rehacer el flujo principal. Para una auditoría estricta, conviene además archivar la versión exacta de `bowling_data.csv`, `bowling_results.csv`, `bowling_correlation_paper_hq.png`, `bowling_correlation_report.html` y la salida de consola del nested CV, porque entre todos forman la traza reproducible completa del experimento.

## 16. Cómo leer el experimento como ingeniero de ML

La forma más limpia de resumirlo es esta. El pipeline parte de un dataset supervisado externo con ratings humanos. Convierte cada acorde en dos representaciones competidoras, una escalar y una vectorial. Evalúa ambas con un protocolo preliminar de predicción fuera de muestra que además produce una visualización directa de error. Después ejecuta un protocolo final con tuning anidado y estimación de varianza por repeticiones. En ambos niveles la representación 12D supera al colapso 1D, y en el nivel final esa superioridad sigue apareciendo cuando se controla por cardinalidad y por selección de hiperparámetros.

Si el experimento se quisiera describir en una sola frase para auditoría, la frase correcta sería que Bowling en este repositorio es una comparación supervisada entre dos spaces de features derivados del mismo modelo psicoacústico, evaluados fuera de muestra, donde la versión 12D obtiene mejor capacidad de generalización que la versión escalar y esa ventaja se verifica de forma más robusta en un esquema de repeated nested cross-validation.
