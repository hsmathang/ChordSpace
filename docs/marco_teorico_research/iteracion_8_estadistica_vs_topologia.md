## 2.3.5 El Estado del Arte Computacional: Modelos Algorítmicos vs Aproximaciones Estadísticas

El mapeo explícito de representaciones musicales al dominio de los números reales es el eje central del análisis musical simbólico moderno. El esfuerzo contemporáneo para acometer la métrica de similitud se divide paradigmáticamente en dos frentes ontológicamente irreconciliables: la estadística sub-simbólica y la topología analítica.

### La Tendencia "Caja Negra" (Deep Learning y Word-Embeddings)
Durante la última década, se ha polarizado masivamente la literatura de "Music Information Retrieval" (MIR) hacia la apropiación directa de arquitecturas derivadas del Procesamiento de Lenguaje Natural (NLP). Estrategias vanguardistas aplican algoritmos como *Word2Vec* (reconvertido a *Chord2Vec*), y modelos fundacionales Transformer o BERT, para abstraer la información interválica desde inmensas corpus de partituras (MIDI o transcripciones).

Bajo estos esquemas computacionales estadísticos, un acorde topológico pierde su naturaleza combinatoria y acústica para colapsar en un token opaco e indivisible en un diccionario gigante. El sistema neuronal proyecta este token sobre un *manifold* latente de alta dimensionalidad basándose única y estrictamente en su probabilidad estocástica de **co-ocurrencia estática**. Si las tríadas de Do Mayor y Fa Mayor ocurren adyacentes con una inmensa frecuencia escalar en el cuerpo de entrenamiento (e.g. la data de los Corales de Bach), la norma euclídea del vector empotrado resultante forzará matemáticamente a que su distancia analítica sea nimia.

### La Crisis de Explicabilidad y la Limitación Estilística
Aventurar modelos abstractos para la métrica disonante mediante algoritmos de aprendizaje profundo incurre velozmente en una fatal debilidad epistemológica en psicoacústica:

> [!WARNING]
> La similitud topológica forzada por un modelo estadístico no exhibe **isomorfismo bottom-up perceptual**, sino que enmascara y colapsa la identidad en una métrica de asimetría puramente de **frecuencia de uso idiomático cultural**.

Estos modelos adolecen de una severa trampa estocástica: un *manifold* neuronal de NLP considerará "ortogonales" o inmensamente distantes a dos entidades raras (ej. un clúster disonante de segundas menores vs la misma estructura transpuesta a registros inexplorados), sencillamente porque no figuran combinados en el *Dataset* de entrenamiento sesgado por el gusto europeo clásico, incluso si físicamente el comportamiento físico de su colisión parcial (y por tanto su rugosidad biológico sensorial subyacente) sea cuasi-idéntico. La métrica degenera de medir el "universo subyacente musical" para convertirse en un clasificador ciego sobre un "género y época histórica" restrictivos empotrados en sus tensores paramétricos estáticos. 

### El Retorno Riguroso a Modelos Topológicos Deterministas y Explicables (White-Box)
Para una audiencia y propósito del área de matemáticas aplicadas, evadir la dependencia en la inferencia de parámetros inexplicables demanda transicionar a un marco constructivo deductivo (*white-box approach*). 

La formalización topológica de `ChordSpace` defendida a lo largo de este diseño adopta sin concesiones una postura fisicalista: la identidad y similitud del objeto-acorde es invariante por escala a la propensión probabilista de que sea o no tocado en el Siglo XVIII. Un modelo paramétrico sobre la rugosidad fisiológica, a través de la inyección matemática directa al vector cromático de Sethares, impone regularidad geométrica y universaliza las distancias reales. Esto habilita que la máquina deduzca si el intervalo no-tonal contemporáneo genera tensión fisiológica asimétrica con certeza abovedada en reglas axiomáticas inmutables, brindando confianza estricta, causalidad evidente en el resultado y computabilidad explicable formal un-a-un entre el comportamiento físico (interferencia en la cóclea) y la predicción del software.
