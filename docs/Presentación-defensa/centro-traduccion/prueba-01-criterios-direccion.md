# Prueba 01 - Criterios de direccion para la defensa

## Lectura de la intencion

La defensa no debe profundizar en detalles tecnicos por defecto. La narracion visual debe hacer comprensible por que el acorde necesita conservar informacion perceptual sin convertir cada diapositiva en una clase de programacion, estadistica o reduccion dimensional.

El centro de gravedad queda asi:

1. Percepcion musical: que oye el cuerpo, que conserva el oido, que se pierde al colapsar intervalos.
2. Estructura intervalar: el acorde como red interna de distancias, no solo como etiqueta.
3. Perfil de rugosidad 12D: no solo cuanta rugosidad hay, sino donde esta distribuida.
4. Validacion Ridge: el argumento fuerte del articulo MAPI revisado.
5. Reduccion dimensional: herramienta de lectura y consistencia, no protagonista epistemica.

## Correcciones incorporadas al criterio visual

- La portada debe usar el titulo institucional exacto de la tesis: "Modelo computacional para la exploracion de acordes en la composicion musical".
- La portada debe mencionar autor, programa, director, codirector y universidad.
- La portada debe ser sobria y de solo texto.
- Las referencias importantes pueden aparecer como titulos/libros/articulos en momentos puntuales, siempre que no compitan con la claridad diagramatica.
- La musica debe poder sonar cuando el concepto lo pida; no es adorno, es evidencia perceptual.

## Prueba auditiva propuesta

Recurso narrativo: melodia reconocible y version con intervalos complementarios.

Intencion:
Mostrar que si se colapsan o reemplazan intervalos por sus complementarios, la identidad perceptual de una melodia puede desfigurarse. Esto prepara la intuicion de por que la tesis conserva 12 bins y no reduce prematuramente a 6 clases interválicas.

Implementacion posible:

- Usar `presentation/audio.js`, que ya genera tonos complejos con `H=6` y `delta=0.88`.
- Crear una funcion `playMelody(midis)` y otra `playComplementMelody(midis)`.
- Representar visualmente dos lineas melodicas:
  - original: contorno reconocible.
  - complementaria: misma maquinaria formal, identidad perceptual rota.

Riesgo:
No afirmar que la melodia sea el objeto de estudio principal. Usarla solo como demostracion auditiva de que conservar intervalos importa para la percepcion.

## Relectura MAPI para orientar resultados

El articulo final MAPI desplaza el enfasis:

- Antes: la proyeccion MDS podia parecer el resultado central.
- Ahora: el resultado fuerte es que el perfil de rugosidad 12D predice mejor consonancia humana que la rugosidad escalar.
- MDS aparece como chequeo exploratorio de consistencia: familias cercanas, inversiones distinguibles, separacion por niveles de rugosidad.

Decision para la defensa:
Las slides de resultados deben priorizar el contraste `1D scalar roughness` vs `12D roughness profile + Ridge`. La reduccion dimensional se explica despues, como mapa de exploracion.

## Fuentes leidas en esta prueba

- `docs/Presentación-defensa/UNAL Design System/uploads/guion_visual_65_diapositivas.md`
- `docs/Presentación-defensa/UNAL Design System/uploads/guion_defensa_tesis_maestro.md`
- `docs/Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024_entregada/0000.tex`
- `docs/Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024_entregada/MAPI_4_2026_final/main.tex`
- `docs/marco_teorico_research/iteracion_3_batimientos_rugosidad.md`
- `docs/marco_teorico_research/iteracion_4_modelo_sethares.md`
- `docs/Presentación-defensa/UNAL Design System/presentation/audio.js`
- `audio.py`

## Primer criterio de aceptacion

Una diapositiva pasa esta prueba si puede explicarse oralmente con una frase corta y si su visual responde a una pregunta: que debe ver o escuchar el jurado para entender esta idea sin leer un parrafo.

## Prueba 02 - Bloque introductorio diagramado

Diapositivas intervenidas:

- 2: pregunta de investigacion, convertida en flujo `acorde -> oido -> espacio navegable`.
- 3: foco musical, separando melodia, ritmo y armonia para ubicar el acorde como objeto perceptual.
- 4: demostracion auditiva, melodia frente a intervalos complementarios con botones de reproduccion.
- 5: decision conceptual, contraste entre colapsar a 6 equivalencias y preservar 12 direcciones.

Fuentes y materiales consultados para esta prueba:

- `docs/Presentacion-defensa/UNAL Design System/presentation/audio.js`
- `docs/Presentacion-defensa/UNAL Design System/slides/AnimatedPianoSlide.jsx`
- `docs/Presentacion-defensa/UNAL Design System/slides/AnimatedVectorSlide.jsx`
- `docs/Presentacion-defensa/UNAL Design System/slides/AnimatedPCSetSlide.jsx`
- `docs/Presentacion-defensa/imagenes-utiles-externas/17-music-concepts-mapi.png`

Hipotesis narrativa:
Antes de mostrar formulas o resultados, el jurado debe sentir que perder direccion intervalar cambia la experiencia musical. Desde ahi se justifica que el modelo conserve un perfil de rugosidad de 12 dimensiones y que el mapa venga despues como forma de exploracion, no como argumento principal.

Chequeo de calidad:
Las cuatro diapositivas nuevas fueron revisadas en navegador local. La diapositiva 4 tiene dos controles de audio; si el gesto auditivo resulta demasiado extremo, se puede ajustar el registro sin cambiar la tesis visual.
