# Modulo de inspiracion visual antes de programar

## Leccion incorporada

Las diapositivas nuevas no deben reemplazar material existente ni reducir la densidad narrativa que ya funcionaba. Cada propuesta nueva entra primero como piloto, comparada contra las diapositivas modelo, y solo despues se integra en el orden final.

## Regla de conservacion

- Nunca sustituir una diapositiva existente sin dejarla visible en el deck.
- Si una idea nueva compite con una diapositiva buena, se agrega como variante posterior.
- La decision final se toma viendo ambas versiones en el navegador, no por intuicion en codigo.

## Modulo 1 - Imagen imprimible de referencia

Antes de programar una diapositiva importante, generar una composicion estatica tipo lamina impresa. Esa imagen funciona como boceto de disenador: permite contar elementos, revisar jerarquia, detectar vacios y comparar contra el estilo de las diapositivas modelo.

Prompt base:

```text
Crear una maqueta editorial imprimible 16:9 para una diapositiva academica de defensa de tesis UNAL.
Tema: [idea central de la diapositiva].
Estilo: sobrio, diagramatico, alto detalle visual, muy poco texto, sistema verde/naranja/negro sobre fondo claro.
Debe incluir muchos elementos pequenos coordinados, como capas que puedan aparecer animadas: diagramas, flechas, puntos, lineas, etiquetas minimas, escalas y pequenos estados intermedios.
No hacer una landing page, no usar tarjetas decorativas, no usar gradientes ni fondos ornamentales.
La diapositiva debe poder leerse como lamina impresa y tambien convertirse luego en HTML/CSS/SVG animado.
```

Salida esperada:

- Una imagen de referencia.
- Conteo aproximado de elementos visibles.
- Lista de grupos animables.
- Riesgos de legibilidad.
- Decision: programar, rehacer boceto o combinar con una diapositiva existente.

## Modulo 2 - Metrica de densidad narrativa

Una diapositiva de alta calidad para este deck no se mide solo por verse limpia. Tambien debe tener suficientes objetos visuales para sostener una explicacion oral rica.

Checklist:

- Claim oral en una frase.
- Objeto visual principal.
- Al menos 4 grupos visuales secundarios cuando la idea sea introductoria o conceptual.
- Al menos 5 apariciones o estados animables cuando la diapositiva explique un mecanismo.
- Poco texto, pero no poca informacion.
- Relacion clara con una fuente del repositorio, imagen seleccionada o codigo reproducible.

## Modulo 3 - Traduccion a codigo

Despues del boceto, la programacion debe reconstruir la lamina con elementos nativos del deck:

- React para estructura.
- SVG para diagramas precisos.
- CSS para ritmo visual y animacion.
- `deck-step` para apariciones progresivas.
- Audio solo cuando el argumento dependa de percepcion.

## Modulo 4 - Comparacion con modelos

Antes de reemplazar o mover una diapositiva, comparar contra:

- diapositivas originales 2 a 9;
- `docs/Presentacion-defensa/imagenes-utiles-externas/17-music-concepts-mapi.png`;
- imagenes utiles seleccionadas en `docs/Presentacion-defensa/imagenes-utiles-externas/`;
- capturas o referencias que el usuario marque como modelo de estilo.

## Modulo 5 - Integracion

Una nueva slide solo pasa al deck principal si:

- no desaparece una slide existente que aun puede servir;
- mejora claridad, densidad o ritmo;
- conserva el sistema institucional;
- tiene una version animable por capas;
- el usuario puede verla junto a la anterior.
