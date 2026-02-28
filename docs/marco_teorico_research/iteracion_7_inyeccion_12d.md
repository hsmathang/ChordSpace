## 2.3.3 La Inyección a $\mathbb{R}^{12}$: El Vector de Características de Disonancia

Habiendo calculado la magnitud tensorial total para la disonancia $D_{\mathcal{F}}$ de un espacio armónico, surge el reto de proyectar y organizar esta métrica escalar en un espacio analítico vectorial tratable geométricamente, preservando la información estructural del `voicing` del acorde.

Para comparar perceptualmente dos acordes $\mathcal{C}_1, \mathcal{C}_2 \in P^n$, se define un mapeo de "características vectoriales" (un embedding fundamentado físicamente) transfiriendo el acorde desde su representación de alturas continuas a un vector estadístico en $\mathbb{R}^{12}$, al que denominaremos vector `dic` (Directed Interval Class) o histograma de disonancias.

### Construcción Matemática del Vector "Raw"

El vector representativo $\vec{v} \in \mathbb{R}^{12}$ de un acorde se define formalmente agregando las penalizaciones de la función de Sethares $d(\cdot)$ sobre tensores unificados por su "clase de intervalo dirigido" residual. 

Para un acorde $\mathcal{C}$ de notas con fundamentales $f_i$, extraemos las distancias interválicas discretizadas entre todo par de notas $(f_i, f_j)$ en semitonos. Designemos el intervalo modular estricto como $k = \phi(f_j, f_i) \pmod{12}$, con $k \in \{1, \dots, 11\}$. 
El ensamble vectorial asigna a cada componente coordenado $\vec{v}_k$ la suma total escalar de la rugosidad psicoacústica emanada de todos los pares exactos de dicho acorde que manifiesten esa misma separación interválica logarítmica fundamental:

$$ \vec{v}_k = \sum_{\{i,j \,|\, \phi(i,j) \equiv k \pmod{12}\}} D_{\text{Sethares}}(i, j) $$

El resultado es un histograma distributivo en $\mathbb{R}^{12}$ donde cada dimensión reporta cuánta "masa" de rugosidad aportan las segundas menores, las terceras mayores, quintas justas, etc., al complejo sónico total.

### 2.3.4 El Rechazo Geométrico a la Equivalencia de Inversión

La teoría analítica de conjuntos atonal (e.g., Allen Forte) reduce clásicamente los intervalos a un Interval Class Vector (ICV) de 6 dimensiones, postulando simetría bajo el operador de inversión topológica (donde una tercera mayor de 4st y una sexta menor de 8st cartografían a la misma clase constitutiva `[Interval Class 4]`). Tymoczko formalizó esto como el espacio cociente $\mathbb{OPTI}$ (Octave, Permutation, Transposition, Inversion).

Para que un modelo logre isomorfismo real con la cognición auditiva (una preservación estricta de distancias acústicas), la equivalencia de inversión debe ser obligadamente invalidada. Acústicamente, el teorema subyacente de Fourier y las series inarmónicas dictaminan que un intervalo y su reverso modular originan batimientos espectrales abismalmente divergentes.

Una tercera mayor (ratio ideal $5:4$) engendra superposiciones primarias altamente consonantes (su 4º y 5º armónico coinciden sin rugosidad). En agudo contraste topológico, su complemento inverso afín, la sexta menor (ratio $8:5$), manifiesta colisiones inarmónicas perjudiciales profundas mucho antes en su serie de Taylor auditiva (su 5º y 8º armónico), arrastrando colateralmente fricción en el registro bajo periférico.

Permitir que el formalismo pliegue $\mathbb{R}^{12}$ a $\mathbb{R}^6$ mediante la equivalencia de inversión disuelve información acústica direccional invaluable (lo que Callender y Tymoczko aislarían teóricamente como restringirse solo al espacio $\mathbb{OPT}$). Por consiguiente, nuestro mapeo paramétrico impone asimetría direccional: las disonancias arrojadas por las terceras mayores alimentarán exclusivamente el índice coordenado $\vec{v}_4$, y las emanadas por sextas menores aislarán de forma asimétrica e independiente en el índice coordenado $\vec{v}_8$. Esta segregación dimensional es matemática y biologicamente vital para asegurar que la topología ulterior del espacio no aplaste fenómenos acústicamente distantes en vértices coincidentes por ciega convención analítica tradicional.
