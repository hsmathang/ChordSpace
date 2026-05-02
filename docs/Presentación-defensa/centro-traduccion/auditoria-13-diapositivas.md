# Auditoria de las 13 diapositivas actuales

Fecha: 2026-05-02  
Deck revisado: `docs/Presentacion-defensa/UNAL Design System/slides/index.html`  
Captura de conjunto: `auditoria-13-contact-sheet.png`

## Criterio usado

Esta auditoria no evalua solo si una diapositiva se ve limpia. Evalua si sostiene una explicacion oral rica con pocos textos y muchos objetos visuales coordinados.

Reglas aplicadas:

- Una frase oral clara no significa baja densidad visual.
- Antes de inventar diagramas, buscar el lenguaje simbolico existente.
- Las diapositivas originales no se reemplazan; se comparan con variantes.
- Una buena diapositiva debe tener objetos, capas, flechas, cajitas, colores y estados animables suficientes.
- La reduccion dimensional no debe ser protagonista en la introduccion; primero van percepcion, acorde, direccion intervalar y rugosidad.

## Lectura general

Las diapositivas 3 a 9 contienen el vocabulario visual mas fuerte del deck: piano, flechas curvas, arreglos, rueda PC-set, panel naranja, vector, esquema de pipeline, rugosidad y advertencias visuales. Las diapositivas piloto 10 a 13 tienen ideas utiles, pero todavia parecen bocetos mas limpios que el lenguaje real del proyecto. Deben enriquecer o conectar las originales, no reemplazarlas.

## Matriz de decision

| Slide | Funcion actual | Diagnostico | Decision | Accion siguiente |
|---|---|---|---|---|
| 1 | Portada institucional | Sobria y correcta. Tiene poca densidad, pero en portada eso puede funcionar. | Conservar | Mantener titulo exacto. Opcional: agregar un motivo visual minimo derivado de rugosidad/vector sin cargarla. |
| 2 | Titulo secundario en ingles | Muy debil frente al resto: poco material, texto en ingles, no cumple densidad ni narrativa. | Rehacer | Convertir en pregunta/tesis visual de alta densidad, usando mini rueda PC-set, piano, curva de rugosidad y flechas. |
| 3 | Intro musical | Buena base de estilo: piano, tabla, rueda de quintas por pasos. | Conservar y enriquecer | Usarla como primera entrada musical. Tomar `17-music-concepts-mapi.png` como fuente de lenguaje y aumentar capas iniciales. |
| 4 | C, Do-Mi-Sol, intervalos y vector | Muy valiosa. Tiene flechas curvas, tuple, arreglos y secuencia explicativa. | Conservar | Revisar si debe ir antes o despues de PC-set. Mantener como mecanismo principal para pasar de acorde a vector. |
| 5 | PC-Set Theory y rueda de clases de altura | Importante. Es el simbolo propio para clases de altura. La version piloto 13 no la mejora. | Conservar y adaptar | No inventar otra rueda. Limpiar texto, mejorar jerarquia y usar la rueda como vocabulario canonico. |
| 6 | Esquematico del modelo | Muy buena como mapa de proceso; alta densidad por capas. | Conservar, posiblemente reubicar | Puede funcionar como slide de cierre de intro o apertura del modelo, no necesariamente tan temprano. |
| 7 | Rugosidad auditiva | Conceptualmente fuerte y ya corregida con curva real por codigo. | Conservar y densificar | Sumar anotaciones visuales desde `21-rugosidad.png`, botones de audio y marcas de intervalos 1/11. |
| 8 | C Major Vector | Repite parte de la slide 4, pero en formato mas limpio. | Comparar / fusionar | Puede fusionarse con slide 4 o quedar como version paso-a-paso mas lenta. Revisar redundancia. |
| 9 | Acordes como tuplas -> iStruct/iVect | Muy util: explica compresion del vector clasico y conserva orden. | Conservar | Ubicar despues de PC-set y despues del ejemplo C mayor. Puede ser puente hacia propuesta 12D. |
| 10 | Pregunta piloto | Idea narrativa correcta, visualmente pobre para el estilo del deck. | Rehacer con lenguaje propio | Convertir en hub visual con simbolos ya existentes: piano, rueda PC-set, curva, vector y espacio navegable. |
| 11 | Capas musicales piloto | Idea util pero muy baja densidad. | Fusionar o reimaginar | Integrarla con la intro musical existente, no dejarla como slide autonoma simple. |
| 12 | Melodia vs complementarios | Buena idea, especialmente por audio. Todavia necesita mas notacion y riqueza visual. | Conservar como piloto a redisenar | Usar frase completa de La Cucaracha, duraciones mejores, pentagrama/contorno, flechas de intervalos y comparacion auditiva. |
| 13 | Doce direcciones piloto | La idea es correcta, pero inventa un simbolo inferior al PC-set existente. | Retirar o rehacer desde slide 5 | Rehacer usando la rueda canonica de clases de altura y no una rueda nueva simplificada. |

## Orden narrativo sugerido para el primer bloque

Este orden conserva lo bueno y ubica las variantes donde aportan:

1. Portada institucional.
2. Pregunta/tesis visual re-hecha a partir de slide 10, pero con densidad de las originales.
3. Intro musical existente.
4. PC-set y clases de altura usando la rueda canonica.
5. C mayor como acorde: Do-Mi-Sol, distancias y vector.
6. Demostracion auditiva: La Cucaracha vs complementarios.
7. Problema de compresion: acordes como tuplas, iStruct/iVect.
8. Rugosidad auditiva: curva real + intervalos 1/11.
9. Esquematico del modelo como mapa de lo que viene.

## Prioridades de trabajo

### Prioridad 1: no perder lenguaje propio

La rueda PC-set, el piano con puntos y lineas, las flechas curvas, las cajitas de arreglos, el panel naranja y los colores verde/naranja/azul/rojo son parte del idioma visual del deck. Cualquier slide nueva debe partir de esos elementos.

### Prioridad 2: corregir el bloque piloto

- Slide 10: rehacer como pregunta de investigacion densa.
- Slide 11: fusionar con intro musical o convertirla en transicion con mas elementos.
- Slide 12: fortalecer como slide auditiva real.
- Slide 13: retirar como grafica autonoma y rehacer desde la rueda canonica.

### Prioridad 3: resolver redundancias

Slides 4 y 8 explican C mayor hacia vector desde enfoques cercanos. No hay que borrar ninguna todavia, pero si decidir si:

- una queda como version animada principal;
- la otra queda como explicacion lenta;
- o ambas se fusionan en una secuencia mas rica.

### Prioridad 4: arreglos tecnicos visibles

- Revisar el asset faltante `assets/sombrero-bug.png`; aparece referenciado en varias slides.
- Generar capturas de estados finales de animacion, no solo estado inicial.
- Mantener numeracion dinamica y verificar que no se rompa al reordenar.

## Siguiente bloque de ejecucion propuesto

1. Crear un boceto imprimible de la nueva slide 2: pregunta/tesis visual con alta densidad.
2. Programarla como variante, sin borrar la actual.
3. Rehacer slide 13 usando la rueda PC-set existente.
4. Reorganizar 10-13 para que funcionen como laboratorio de variantes, no como parte final confusa.
5. Volver a mostrar el deck con originales + variantes comparables.

## Fuentes revisadas

- `slides/index.html`
- `slides/AnimatedPianoSlide.jsx`
- `slides/AnimatedVectorSlide.jsx`
- `slides/AnimatedPCSetSlide.jsx`
- `slides/AnimatedSchematicSlide.jsx`
- `slides/IntervalRoughnessSlide.jsx`
- `slides/CMajorVectorSlide.jsx`
- `slides/ChordsToVectorsSlide.jsx`
- `slides/IntroNarrativeSlides.jsx`
- `docs/Presentacion-defensa/imagenes-utiles-externas/README.md`
- `docs/Presentacion-defensa/imagenes-utiles-externas/17-music-concepts-mapi.png`
- `docs/Presentacion-defensa/imagenes-utiles-externas/21-rugosidad.png`
