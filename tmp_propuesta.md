| PROYECTO |     | DE  | TESIS | DE  | MAESTRIA |     | EN  | MATEMATICA |     |     |
| -------- | --- | --- | ----- | --- | -------- | --- | --- | ---------- | --- | --- |
APLICADA
| Modelo | computacional |     |             | para | la exploraci´on |         |          | de  | acordes | en la |
| ------ | ------------- | --- | ----------- | ---- | --------------- | ------- | -------- | --- | ------- | ----- |
|        |               |     | composicion |      |                 | musical |          |     |         |       |
|        | Proponente:   |     | Hern´an     |      | Santiago        |         | Angarita |     | Garc´ıa |       |
|        |               |     | Director:   |      | Carlos          | Andr´es | Torres   |     |         |       |
Master en Ingenier´ıa de Sistemas - Universidad Nacional de Colombia
|                  | Licenciado  |             | en           | Mu´sica   | - Universidad |              |          | de Antioquia   |     |     |
| ---------------- | ----------- | ----------- | ------------ | --------- | ------------- | ------------ | -------- | -------------- | --- | --- |
|                  | Codirector: |             |              | Francisco | Albeiro       |              | G´omez   | Jaramillo      |     |     |
|                  |             |             | Departamento |           | de            | Matem´aticas |          |                |     |     |
|                  |             |             |              | Facultad  | de            | Ciencias     |          |                |     |     |
|                  |             | Universidad |              |           | Nacional      | de           | Colombia |                |     |     |
| 1. Planteamiento |             |             |              | del       | problema      |              | y        | Justificaci´on |     |     |
Un grupo de notas tocadas simulta´neamente crea una sensacio´n sonora que puede
diferenciarse en funcio´n de la ´epoca y la cultura en la que se interpreta. La teor´ıa
armo´nica estudia las relaciones entre estos grupos de notas, conocidos como acordes
[1]. Estos estudios suelen basarse en el an´alisis de obras maestras, los tipos de acordes,
las sensaciones sonoras que se producen al combinarse con otros en una progresio´n,
entre otros aspectos. Sin embargo, en el proceso compositivo, tambi´en es importante
explorar sonoridades nunca antes exploradas en el contexto arm´onico de una ´epoca
| particular | [2]. |     |     |     |     |     |     |     |     |     |
| ---------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
La eleccio´n de un acorde sustituto por parte del mu´sico suele basarse en un an´alisis de
las notas que componen el acorde original y de la sensacio´n sonora que se desea trans-
mitir al tocar el acorde en progresio´n con otros. Este ana´lisis armo´nico esta´ limitado
por la teor´ıa arm´onica de cada ´epoca y la experiencia del mu´sico. Una exploraci´on
exhaustiva de todos los grupos de notas posibles y una codificaci´on computacional de
las caracter´ısticas sonoras deseadas permitir´ıa al mu´sico ampliar el espacio de acor-
des sustituibles y encontrar sonoridades nunca antes exploradas en el contexto de
una teor´ıa arm´onica asociada a una ´epoca determinada. Adema´s, la visualizaci´on en
un espacio de representacio´n de las medidas de distancia de sensaci´on sonora entre
1

acordes proporcionar´ıa al mu´sico una forma interpretable de explorar y enriquecer su
composicio´n.
1.1. Planteamiento y preguntas de investigacio´n
La generaci´on autom´atica o asistida de acordes interpretables y originales puede te-
ner un gran impacto en la tecnolog´ıa mu´sica. Sin embargo, hasta donde conocemos
las aproximaciones actuales se enfocan en la caracterizacio´n de estructuras musicales
conocidas. En este contexto surge la siguiente pregunta ¿Co´mo encontrar una manera
de facilitar la exploracio´n de nuevas sonoridades en la composicio´n musical, superan-
do las limitaciones de la teor´ıa arm´onica tradicional?.
Esteproyectoproponeconstruirunespacioderepresentaci´onparaacordesquepermi-
ta ubicar a aquellos con sonoridades similares en una misma a´rea, utilizando t´ecnicas
de codificacio´n de caracter´ısticas, algoritmos de generaci´on combinatorial y represen-
tacio´n computacional.
Para abordar esta problem´atica se propone abordar la siguiente pregunta de investi-
gacio´n:
¿ Es posible construir un espacio de representacio´n para acordes que permita ubicar
cerca aquellos posibles sustitutos por sonoridad similar?
Explorando en profundidad la siguiente hipo´tesis de trabajo:
Dado un contexto armo´nico definido existen acordes no explorados que satisfacen
caracter´ısticas de similitud sonora que los habilitan como sustitutos de otros acordes
conocidos.
2. Marco Te´orico y Estado del Arte
En este trabajo, presentamos una revisio´n de la literatura sobre la exploracio´n y re-
presentacio´n de acordes musicales. Primero, abordaremos algunos elementos teo´ricos
sobre la exploracio´n de acordes, incluyendo su importancia para la codificacio´n y
representacio´n computacional de las caracter´ısticas sonoras de un acorde. Tambi´en
discutiremos las diferencias entre los enfoques ma´s importantes de la codificacio´n de
acordesysuimpactoenlarepresentacio´ngeom´etricadealgunasrelacionesentreellos.
Finalmente, analizaremos los trabajos y herramientas utilizados en la representacio´n
y sustituci´on de acordes, desde nuestra perspectiva de exploraci´on y representacio´n.
La sustituci´on de acordes, la rearmonizacio´n y la exploraci´on de un espacio de acordes
en general se han discutido desde dos perspectivas: la teor´ıa musical y la tecnolog´ıa
musical. La primera se enmarca en la historia de la Teor´ıa Musical Armo´nica, y
2

co´mo, con el tiempo, las restricciones en la forma en que se construye un acorde van
cambiando. El segundo, aprovecha la tecnolog´ıa y los recursos computacionales para
potenciar las tareas del mu´sico.
La mayor´ıa de los intentos de lograr sustituciones autom´aticas de acordes se han
enmarcado dentro de la teor´ıa musical occidental, espec´ıficamente la teor´ıa armo´nica
occidental que se origino´ entre los an˜os 1650 y 1900; donde se definen esta´ndares y
se establecen limitaciones para la formacio´n de acordes, adema´s de popularizar la
armon´ıa tonal. All´ı se tienen en cuenta caracter´ısticas como las distancias entre las
notas, el orden en que se tocan y la tonalidad subyacente, para definir el tipo de
sensacio´n sonora que tiene un conjunto de notas, y decidir si sera´ consonante o no
armonioso.
En este contexto, la sustituci´on de acordes se refiere al reemplazamieno de un acorde
por otro que tenga una sensacio´n sonora similar, pero que no sea igual. Esto se puede
lograr a trav´es de la modificacio´n de algunas de sus notas, o a trav´es de la adici´on o
eliminacio´n de algunas de ellas, manteniendo el orden y la tonalidad. La rearmoniza-
cio´n”; por su parte, implica la modificaci´on de la estructura arm´onica de una pieza
musical, mediante la sustituci´on de acordes o la modificaci´on de su secuencia.
La tecnolog´ıa musical ha permitido la automatizacio´n de estos procesos, mediante
el uso de algoritmos y herramientas computacionales que facilitan la exploracio´n y
representacio´n del espacio de acordes. Esto ha permitido una mayor flexibilidad y
precisio´n en la sustituci´on y rearmonizaci´on de acordes, adema´s de facilitar la toma
de decisiones en la creaci´on musical.
En resumen, la sustitucio´n y rearmonizaci´on de acordes se ha discutido desde la pers-
pectiva de la teor´ıa musical y la tecnolog´ıa musical, permitiendo la automatizacio´n de
estos procesos mediante el uso de herramientas computacionales. Esto ha permitido
una mayor flexibilidad y precisio´n en la creacio´n musical.
´
CODIFICACION
Una codificaci´on detallada de las caracter´ısticas del acorde permite una representa-
cio´n confiable del contexto armo´nico y las relaciones de sonido entre las notas que lo
componen, como lo demuestran los sistemas automa´ticos de identificaci´on de acordes
[3]. La forma en que se codifican las caracter´ısticas de un acorde ha ido cambiando a
medida que la teor´ıa musical ha permitido ciertos grados de libertad. La importancia
del contexto tonal, la cardinalidad del acorde y la disonancia entre los componentes
del acorde son algunas de las caracter´ısticas que se codifican.
Ha habido diferentes tipos de codificaci´on desde la tecnolog´ıa de la mu´sica, por men-
cionar algunos b´asicos, la codificacio´n en nu´meros romanos para referirse a los distin-
tos acordes de la esacala de la tonalidad, la representacio´n MIDI (Musical Instrument
Digital Interface) que codifica elementos como el tono, el tiempo y el volumen de una
nota.
3

Una de las representaciones ma´s populares es la General Chord Type (GCT) [4] que
reorganiza las notas de un acorde de forma que la base de la codificaci´on sea lo m´as
consonante posible, en base a una clasificacio´n binaria de consonancia y disonan-
cia, una de sus mayores caracter´ısticas es que logra reconocer una buena variedad
muestras sonoras para los acordes, sin importar si es explicito o no el tono sobre el
que se esta´ ejecutando la progresio´n. Sin embargo recientemente se han encontra-
do algunos errores de ambigu¨edad ( como identificar mal la ra´ız del acorde). Han
presentado mejoras del algoritmo que prueban tener un mejor rendimiento , determi-
nando reordenaciones de un acorde, e identificando correctamente acordes extran˜os
con disonancia alta [4]. Algunos otros algoritmos de codificacio´n previos GCT esta´n
en [5],[6],[7].
Es relevante la codificacio´n para una generacio´n autom´atica y detallada de acordes[3],
entre otras razones, porque se desea capturar la mayor cantidad de relaciones arm´oni-
cas percibidas por el o´ıdo entrenado del mu´sico [8].
Desde la perspectiva de la tecnolog´ıa musical, se han implementado diferentes codi-
ficaciones para representar acordes con diferentes objetivos, como la clasificaci´on de
acordes, la bu´squeda del siguiente acorde en una progresio´n, la sustituci´on de acordes,
el reconocimiento automa´tico de estructuras arm´onicas, la recomendaci´on de acordes,
y la representaci´on geom´etrica de estructuras arm´onicas.
La importancia de una representacio´n geom´etrica ha permitido dotar de una es-
tructura topolo´gica a las progresiones de acordes, como es el caso del Tonnetz, una
representacio´n geom´etrica que utiliza la nocio´n de distancia en el espacio para reflejar
relaciones interesantes entre acordes, como la jerarqu´ıa de las tensiones presentes en
los intervalos de un acorde. Esto ha facilitado la exploraci´on y representacio´n de las
relaciones entre acordes en la creaci´on musical [9].
La noci´on de distancia se hace presente, y se profundiza a trav´es del c´alculo de to-
dos los posibles intervalos desde un centro tonal usando la transformada de Fourier
Discreta (DFT). Distintas maneras de medir consonancia entre intervalos, provocan
distintas representaciones espaciales de la similitud entre acordes, en particular el es-
pacio de intervalo tonal se logran representar relaciones de consonancia que coinciden
de buena manera con la percepci´on del mu´sico [10], consideramos u´til este acerca-
mientoparaimplementarloalmomentodemedirdisonanciaytensio´njer´arquicaentre
grupos de notas.
´ ´
HERRAMIENTAS DE RECOMENDACION Y SUSTITUCION
Recientemente, Se han explorado distintos modelos basados en machine learning y
redes neuronales profundas en la caracterizacio´n de la estructura musical. La gran
mayor´ıa de estos trabajos esta´n enmarcados en aprender las caracter´ısticas sonoras
de un genero [11], o en particular de las progresiones de acordes ma´s populares o
la direcci´on de voz con mayor probabilidad de aparici´on[7], [12]–[21] , por lo que el
4

´enfasis no ha sido la exploracio´n total de los conjuntos de notas, sino ma´s bien el
| aprendizaje | supervisado. |     |     |
| ----------- | ------------ | --- | --- |
Por ejemplo, Giannos y Cambouropoulos proponen ChordAIS-Gen[3],, una herra-
mienta que asiste al mu´sico para que pueda generar progresiones de acordes que
cumplan con determinados criterios de tensio´n. Los autores presentan un algoritmo
gen´etico junto con un “artificial immune system” que se encarga de aprender reglas ,
como por ejemplo consonancia, tensio´n, y voice leading. Evalu´an su desempen˜o me-
diante un aplicativo que pide calificar al mu´sico su percepcio´n de las progresiones de
| acordes sugeridas. |     |     |     |
| ------------------ | --- | --- | --- |
Huang et al proponen ChordRiple [22], una herramienta computacional que presen-
ta recomendaciones de acordes desde un modelo de red neuronal entrenado con un
corpus corpus de progresiones MIDI que captura la probabilidad de que aparezca un
acorde de acuerdo a su contexto usando un modelo basado en WORD2VEC mide la
| cercan´ıa entre | acordes usando | la distancia | coseno. |
| --------------- | -------------- | ------------ | ------- |
Cabe destacar la falta ausencia de un enfoque exploratorio en la literatura existen-
te que permita generar acordes interpretables y originales. El enfoque ma´s comu´n
ha sido la representacio´n de caracter´ısticas sonoras para utilizar acordes conocidos y
progresiones de acordes en la direcci´on de la voz. Nuestro enfoque.
Enestetrabajoseproponeunnuevoenfoqueexploratorioquebuscaencontrarnuevos
acordesnuncaantesexploradosquepreservenalgunasdelascaracter´ısticassonorasde
los acordes m´as conocidos. Por lo tanto, los espacios de representacio´n mencionados
en la literatura deben ser considerados como subespacios del espacio de generaci´on
total.
3. Objetivos
Con el fin de abordar la hipo´tesis planteada se proponen los siguientes objetivos
| 3.1. Objetivo | general |     |     |
| ------------- | ------- | --- | --- |
Desarrollar un modelo computacional que para la construccio´n de un espacio de
| representacio´n | y sustitucio´n | arm´onica. |     |
| --------------- | -------------- | ---------- | --- |
| 3.2. Objetivos  | espec´ıficos   |            |     |
Modelar matem´aticamente un conjunto de reglas y caracter´ısticas para la ge-
| neracio´n | de acordes | del espacio. |     |
| --------- | ---------- | ------------ | --- |
Implementar t´ecnicas de reduccio´n de dimensionalidad para conseguir represen-
taciones interpretables que conserve la nocio´n de distancia entre acordes.
5

Establecer medidas de similitud sonora que permitan identificar acordes susti-
|     | tutos | en el contexto | de  | la mu´sica | Barroca. |     |     |     |
| --- | ----- | -------------- | --- | ---------- | -------- | --- | --- | --- |
Evaluar cuantitativamente la calidad del espacio de representaci´on propuesta.
| 4.  | Metodolog´ıa |     |     | y Cronograma |     |     | de  | Actividades: |
| --- | ------------ | --- | --- | ------------ | --- | --- | --- | ------------ |
En base a los objetivos planteados, hemos establecido cuatro etapas. Cada una de
estas etapas nos permitira´ avanzar en el objetivo principal de nuestro trabajo, que es
expandirlateor´ıaarmo´nicatradicionalatrav´esdelusodelacodificaci´oncomputacio-
nal para explorar nuevas sonoridades en la composici´on musical. Esperamos que esta
metodolog´ıa nos permita desarrollar una herramienta eficiente y u´til para los mu´sicos
y compositores. 1. An´alisis de caracter´ısticas: se definira´n las variables y se codifi-
cara´n las caracter´ısticas (en dos niveles). 2. Disen˜o de un algoritmo de generacio´n de
| grupos | de notas      | (candidatos     |     | a acordes). |     |       |     |     |
| ------ | ------------- | --------------- | --- | ----------- | --- | ----- | --- | --- |
| 3.     | Agrupamiento. | 4. Evaluacio´n. |     |             |     |       |     |     |
| a.     | Revisio´n     | bibliogra´fica  | del | estado      | del | arte. |     |     |
b. An´alisis de caracter´ısticas: se definira´n las variables y se codificar´an las carac-
ter´ısticas:
|     | Esta | etapa se desarrollara´ |     | en  | dos niveles: |     |     |     |
| --- | ---- | ---------------------- | --- | --- | ------------ | --- | --- | --- |
Nivel 1: Identificacio´n de variables. Con base en la literatura musical, se iden-
tificara´n las variables ma´s importantes para la generacio´n de acordes, como la
cardinalidad del acorde, la distancia entre notas, el nivel de disonancia interva-
|     | lica, la | nota ra´ız, el | rango | del | acorde, | las extensiones, |     | entre otras. |
| --- | -------- | -------------- | ----- | --- | ------- | ---------------- | --- | ------------ |
Nivel 2: Codificacio´n. Algunas de las variables han sido codificadas en la litera-
tura que hemos revisado, adaptaremos y propondremos refinaciones, y alterna-
tivas para la codificacio´n de variables segu´n vayamos avanzando. Adem´as, en
la revisio´n de la literatura no percibimos un modelamiento para codificar las
distancias entre las notas del acorde, por lo que propondremos uno propio.
c. Disen˜o de algoritmo para la generaci´on de grupos de notas (candidatos a acor-
des) Una vez que hayamos definido las variables pertinentes, trabajaremos en
|     | dos niveles: |     |     |     |     |     |     |     |
| --- | ------------ | --- | --- | --- | --- | --- | --- | --- |
Nivel 1: Generaci´on de la poblaci´on. Disen˜aremos para´metros de generaci´on
con t´ecnicas combinatoriales, como el taman˜o del conjunto de notas con el
que generaremos acordes, la nota m´as grave, la m´as aguda, entre otros. Poste-
riormente, implementaremos algoritmos de generacio´n combinatoria (un todos
contra todos) que formen, a partir del conjunto semilla, los individuos candida-
|     | tos a | acordes. |     |     |     |     |     |     |
| --- | ----- | -------- | --- | --- | --- | --- | --- | --- |
6

Nivel2:Evaluacio´ndelapoblacio´n.Unavezquehayamosgeneradolapoblacio´n
de candidatos a acordes, evaluaremos cada uno de ellos utilizando las variables
codificadasenlaEtapa1.Deestamanera,podremosidentificaraquellosacordes
que se ajusten mejor a las caracter´ısticas que hemos definido como relevantes.
d. Etapa 3: Agrupamientos y Clusters En esta etapa aplicaremos t´ecnicas de reduc-
cio´n de dimensionalidad y filtros que nos permitan confirmar que los acordes
| etiquetados | est´en cerca | en el espacio | de  |     |     |     |
| ----------- | ------------ | ------------- | --- | --- | --- | --- |
representacio´n que hemos construido. De esta manera, validaremos el correcto
| funcionamiento | de nuestra | hip´otesis. |     |     |     |     |
| -------------- | ---------- | ----------- | --- | --- | --- | --- |
e.
Etapa4:Evaluacio´nEnestaetapa,propondremosuninstrumentodeevaluacio´n
para mu´sicos que les permita calificar las sugerencias y recomendaciones que
nuestro espacio de representacio´n les ofrece para un contexto arm´onico definido
en las reglas de arm´onicas del Barroco. De esta manera, podremos medir la
eficiencia y utilidad de nuestra herramienta en un contexto real de composicio´n
musical.
| f. Elaboraci´on | del documento. |     |     |     |     |     |
| --------------- | -------------- | --- | --- | --- | --- | --- |
Activ.⧹sem.
|             | 1 2      | 3 4 5 | 6 7 8 | 9 10 11 | 12 13 14 | 15 16 |
| ----------- | -------- | ----- | ----- | ------- | -------- | ----- |
| a           | x x      | x x   |       |         |          |       |
| b           |          | x     | x x   |         |          |       |
| c           |          | x     | x x x | x x     |          |       |
| d           |          |       |       | x       | x x      |       |
| e           |          |       |       |         | x x      | x x   |
| f           |          |       |       |         | x        | x x   |
| 5. Recursos | F´ısicos |       |       |         |          |       |
Acceso mediante Internet a bases de datos con publicaciones cientificas perio-
| dicas como | ScienceDirect | y EBSCOhost. |     |     |     |     |
| ---------- | ------------- | ------------ | --- | --- | --- | --- |
Computador con software especializado en: manejo de datos y programacio´n
(como C++).
7

| 6. Costos |     | y   | fuentes |     | de  | financiacio´n |     |     |     |
| --------- | --- | --- | ------- | --- | --- | ------------- | --- | --- | --- |
Loscostosasociadosparaeldesarrollodeesteproyectoestandetalladosenlasiguiente
tabla:
|     | Concepto |     |     |     |     |     |     | Costo |     |
| --- | -------- | --- | --- | --- | --- | --- | --- | ----- | --- |
$
|     | Fotocopias |     | de  | Art´ıculos |     | y libros |     | 1’500,000 |     |
| --- | ---------- | --- | --- | ---------- | --- | -------- | --- | --------- | --- |
$
|     | Computador |     |     | con | software | especializado |     | 2’000,000 |     |
| --- | ---------- | --- | --- | --- | -------- | ------------- | --- | --------- | --- |
$
|     | Trabajo |     | de Digitaci´on |     |     | y encudernacio´n |     | 500,000 |     |
| --- | ------- | --- | -------------- | --- | --- | ---------------- | --- | ------- | --- |
$
|     | Papeleria |     | y otros |     | gastos | de impresi´on |     | 500,000 |     |
| --- | --------- | --- | ------- | --- | ------ | ------------- | --- | ------- | --- |
$
|     | Uso | de  | tecnolog´ıas |     | en la | nube |     | 1’000,000 |     |
| --- | --- | --- | ------------ | --- | ----- | ---- | --- | --------- | --- |
$
|     | Total |     |     |     |     |     |     | 5’500,000 |     |
| --- | ----- | --- | --- | --- | --- | --- | --- | --------- | --- |
Los costos demandados por este trabajo, ser´an asumidos con recursos propios de el
proponente.
Referencias
| [1] W. Piston, | Harmony. |     | 1962. |     |     |     |     |     |     |
| -------------- | -------- | --- | ----- | --- | --- | --- | --- | --- | --- |
[2] M. Tiz´on D´ıaz and D. M. Vela Gonza´lez, “CIFRADO Y FUNCIONALIDAD EN
´
| LA ARMONIA    |     | TONAL:   |          | UNA | PROPUESTA |          | PARA    | EL AULA Encryption | and |
| ------------- | --- | -------- | -------- | --- | --------- | -------- | ------- | ------------------ | --- |
| Functionality |     | in Tonal | Harmony: |     | A         | Proposal | for the | Classroom.”        |     |
[3] W. W. Warthog, Handbook of Pythagorean Triplets, Claremont College Press,
1993.
[4] M. Navarro-Ca´ceres, J. F. M. Sa´nchez-Jara, V. R. Quietinho Leithardt, and R.
Garc´ıa-Ovejero, “Assistive model to generate chord progressions using genetic
programming with artificial immune properties,” Applied Sciences (Switzerland),
| vol. 10, | no. 17, | 2020, | doi: | 10.3390/app10176039. |     |     |     |     |     |
| -------- | ------- | ----- | ---- | -------------------- | --- | --- | --- | --- | --- |
[5] K. Giannos and E. Cambouropoulos, “Symbolic Encoding of Simultaneities: Re-
designing the General Chord Type Representation,” in ACM International Con-
| ference | Proceeding |     | Series, | 2021. | doi: | 10.1145/3469013.3469022. |     |     |     |
| ------- | ---------- | --- | ------- | ----- | ---- | ------------------------ | --- | --- | --- |
[6] D. Tymoczko, “The geometry of musical chords. Supporting Online Material,”
| Science, | vol. | 313, | no. 5783, | 2006. |     |     |     |     |     |
| -------- | ---- | ---- | --------- | ----- | --- | --- | --- | --- | --- |
[7] L. Bigo and A. Spicher, “Self-assembly of musical representations in MGS,” In-
ternational Journal of Unconventional Computing, vol. 10, no. 3, 2014.
8

[8] I. Nemoto and M. Kawakatsu, “A two-dimensional representation of musical
chordsusingthesimplicityoffrequencyandperiodratiosascoordinates,”Journal
of Mathematics and Music, 2021, doi: 10.1080/17459737.2021.1924304.
[9] F. Marmel, A. Parbery-Clark, E. Skoe, T. Nicol, and N. Kraus, “Harmonic rela-
tionships influence auditory brainstem encoding of chords,” Neuroreport, vol. 22,
no. 10, 2011, doi: 10.1097/WNR.0b013e328348ab19.
[10] D. Tymoczko, “The generalized Tonnetz,” Journal of Music Theory, vol. 56, no.
1. 2012. doi: 10.1215/00222909-1546958.
[11] G. Bernardes, D. Cocharro, M. Caetano, C. Guedes, and M. E. P. Da-
vies, “A multi-level tonal interval space for modelling pitch relatedness
and musical consonance,” J New Music Res, vol. 45, no. 4, 2016, doi:
10.1080/09298215.2016.1182192.
[12] L. Lozano, A. Medaglia, and N. Velasco, “Generation of Pop-Rock Chord Se-
quences Using Genetic Algorithms and Variable Neighborhood Search,” 2009.
[13] M. Barthet, A. Anglade, G. Fazekas, S. Kolozali, and R. Macrae, “Music recom-
mendation for music learning: Hotttabs, a multimedia guitar tutor,” in CEUR
Workshop Proceedings, 2011, vol. 793.
[14] H. T. Cheng, Y. H. Yang, Y. C. Lin, I. bin Liao, and H. H. Chen, “Automatic
chord recognition for music classification and retrieval,” in 2008 IEEE Internatio-
nal Conference on Multimedia and Expo, ICME 2008 - Proceedings, 2008. doi:
10.1109/ICME.2008.4607732.
[15] C. H. Chuan, K. Agres, and D. Herremans, “From context to concept: exploring
semantic relationships in music with word2vec,” Neural Comput Appl, vol. 32,
no. 4, 2020, doi: 10.1007/s00521-018-3923-1.
[16] J. A. Burgoyne, J. Wild, and I. Fujinaga, “An expert ground-truth set for audio
chord recognition and music analysis,” in Proceedings of the 12th International
Society for Music Information Retrieval Conference, ISMIR 2011, 2011.
[17] C. M. Wilk and S. Sagayama, “Automatic music completion based on joint opti-
mization of harmony progression and voicing,” Journal of Information Processing,
vol. 27, 2019, doi: 10.2197/IPSJJIP.27.693.
[18] A. Xambo´, J. Pauwels, G. Roma, M. Barthet, and G. Fazekas, “Jam
with Jamendo: Querying a large music collection by chords from a learner’s
perspective,” in ACM International Conference Proceeding Series, 2018. doi:
10.1145/3243274.3243291.
9

[19] H. W. Dong, W. Y. Hsiao, L. C. Yang, and Y. H. Yang, “Musegan: Multi-
track sequential generative adversarial networks for symbolic music generation
and accompaniment,” in 32nd AAAI Conference on Artificial Intelligence, AAAI
| 2018, 2018. | doi: 10.1609/aaai.v32i1.11312. |     |
| ----------- | ------------------------------ | --- |
[20] J. Pauwels and M. B. Sandler, “A web-based system for suggesting new practice
material to music learners based on chord content,” in CEUR Workshop Procee-
| dings, 2019, | vol. 2327. |     |
| ------------ | ---------- | --- |
[21] I. Jimenez, T. Kuusi, and C. Doll, “Common Chord Progressions and Feelings of
Remembering,” Music Sci (Lond), vol. 3, 2020, doi: 10.1177/2059204320916849.
[22] B. McFee and J. P. Bello, “Structured training for large-vocabulary chord recog-
nition,” in Proceedings of the 18th International Society for Music Information
| Retrieval | Conference, ISMIR | 2017, 2017. |
| --------- | ----------------- | ----------- |
[23] C. Z. A. Huang, D. Duvenaud, and K. Z. Gajos, “ChordRipple: Recommen-
ding chords to help novice composers go beyond the ordinary,” in International
Conference on Intelligent User Interfaces, Proceedings IUI, Mar. 2016, vol. 07-10-
| March-2016, | pp. 241–250. | doi: 10.1145/2856767.2856792. |
| ----------- | ------------ | ----------------------------- |
Firma Proponente:
Firma Director:
Firma Codirector:
10