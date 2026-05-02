# Universidad Nacional de Colombia — Design System

## Overview

This design system captures the visual identity and slide presentation conventions of **Universidad Nacional de Colombia (UNAL)**, Colombia's largest and most prestigious public university. The materials provided are academic slide decks used for research presentations — specifically from the **Facultad de Ciencias, Departamento de Matemáticas - Matemáticas Aplicadas**.

The presentation shown covers *"Computational Exploration of a Musical Chord Space: Una representación perceptual para acordes musicales"*, demonstrating the full range of UNAL's slide design language.

### Sources Provided
- `uploads/escudo y tapa primera vista.png` — Cover slide (dark green, escudo centered)
- `uploads/portada colores y marcos.png` — Title/subtitle slide with type specimen
- `uploads/otro ejemplo de estilo.png` — Typography specification slide (Ancízar Serif + Ancízar Sans)
- `uploads/esto no es una diapo estatica...png` — Animated diagram slide (piano keyboard)
- `uploads/otro ejemplo de neustro uso del espacio y de las graficas.png` — Schematic flow diagram
- `uploads/otro ejemplo de flechas y nuestro uso del espacio y de los esquemas.png` — Arrow/node diagram with mascot
- `uploads/el tipo de flechas y distribucion de espacio casi sin texto esquemas suficientes.png` — Split-panel orange/white layout
- `uploads/el tipo de elementos diveritdos que usamos para contar historia.png` — LaTeX + circle diagram with warning bug
- `uploads/el estilo de nuestros diagramas.png` — Piano + circle of fifths diagram
- `uploads/cierre.png` — Closing slide (dark green, italic "Gracias")

No Figma links or codebase were provided. Design system inferred entirely from slide screenshots.

---

## CONTENT FUNDAMENTALS

### Language & Tone
- **Bilingual**: Spanish primary for institution/department credits and closing slides; English for research/academic content titles. Both languages appear freely within the same deck.
- **Academic register**: Formal, precise, technical language. No colloquialisms. Mathematical notation is first-class content.
- **Casing**: Title case for English headers; sentence case for Spanish headers. Subtitles in **ALL CAPS** with wide letter-spacing (Ancízar Sans style).
- **Emoji**: Never used. Zero emoji.
- **Punctuation**: Standard academic Spanish/English. Em dashes (–) used, not hyphens.
- **Formulas**: LaTeX-style math rendering is embedded directly in slides as image or rendered text. Equations are treated as primary narrative elements, not footnotes.
- **Copy density**: **Very low text density**. Slides rely on diagrams, arrows, color-coding, and spatial layout to carry meaning. A complete slide can have fewer than 10 words.
- **Humor**: Subtle and academic. A cartoon bug (bicho/insecto) with a sombrero is used as a recurring mascot. Warning triangles with bug icons signal "watch out!" moments in the narrative.
- **Voice**: Third-person academic for body content. The presenter's voice is in the diagrams themselves — the slides are visual aids, not teleprompters.

### Specific examples
- "Nota base - Inversión" (Slide title — minimal, factual)
- "¿Por qué ocurre esto?" (One question on an entire diagram slide)
- "Gracias" (Entire closing slide — just this one word, italic, centered)
- "ESPACIADO INTERLETRA SEPARADO – 14pt" (Subtitle in caps, spaced)

---

## VISUAL FOUNDATIONS

### Colors
| Name | Hex | Usage |
|---|---|---|
| UNAL Green | `#1B7A3E` | Primary background (covers, closings), titles, accents |
| UNAL Green Light | `#2DC56E` | Bottom accent lines on slides |
| UNAL Gold | `#E8A020` | Top-right horizontal accent line; decorative |
| UNAL Orange | `#E8610A` | Content highlight panel backgrounds (split layouts) |
| UNAL Beige | `#F0EBE0` | Footer strip on content slides |
| White | `#FFFFFF` | Primary content slide background |
| Near Black | `#1A1A1A` | Default body text |
| Diagram Blue | `#1E6BB8` | Arrow and element color in diagrams |
| Diagram Red | `#C0392B` | Highlight dots, error markers, emphasis |
| Diagram Green | `#27AE60` | Secondary diagram element color |
| Diagram Gray | `#888888` | Secondary arrows, less-important connections |

### Typography
**UNAL uses its own proprietary typeface family: Ancízar** (designed for Universidad Nacional de Colombia).
- **Ancízar Serif Extrabold** — slide titles, ~22pt. High-contrast, classical Italian-style serif.
- **Ancízar Sans** — subtitles/labels, uppercase, wide letter-spacing, ~14pt.
- **Body/content**: Clean serif (appears similar to Georgia or Computer Modern for LaTeX content). Diagrams use a neutral sans (Arial-like) for labels.

> ⚠️ **Font substitution**: Ancízar fonts are proprietary to UNAL and not publicly available. This design system uses **Playfair Display** (for Ancízar Serif) and **Raleway** (for Ancízar Sans) from Google Fonts as close substitutes. Ask the university for original `.ttf` or `.otf` files to replace these.

### Slide Layout Structure
Every slide follows a consistent chrome system:
- **Top strip**: Page number top-left. Thin gold horizontal line top-right (decorative accent, ~60% width).
- **Bottom strip**: Thin green accent lines bottom-left (two short parallel lines). UNAL escudo bottom-right corner (small). Department name bottom-left in small italic green text.
- **Footer background**: Sandy beige bar across bottom on content slides.
- **Content area**: White. Generous whitespace. Left-aligned titles in green.

### Backgrounds
- **Cover / Closing**: Full-bleed dark UNAL Green (`#1B7A3E`). Logo centered. Gold + green accent lines at edges.
- **Content slides**: White. No background texture, no images behind content.
- **Split-panel slides**: Vertical split — white left, UNAL Orange right. Title in orange panel, white text.
- **No gradients**. No background photography. No textures.

### Diagrams & Schematics
- Diagrams ARE the content. Heavy use of:
  - Curved and straight arrows (black, with arrowhead)
  - Color-coded parallel flows (red/blue/green for different vectors)
  - Dashed border boxes for formulas and definitions
  - Dotted-line grid separators between diagram zones
  - Column labels below content areas (e.g. "Codificación | Diversidad | Percepción | Visualización")
- Piano keyboard illustrations (black/white keys, numbered, with colored dots marking notes)
- Circle of fifths diagrams (wheel with note names)
- Binary vector grids

### Animation Style
- Slides are **designed to be animated** — elements appear progressively aligned with the speaker's narrative. Layout is spatial; elements enter in sequence to guide attention.
- No decorative animations. All motion is purposeful/narrative.
- CSS transitions if recreating: fade-in or slide-in, not bounces. Easing: ease-in-out.
- Animated slides must use the shared `window.useDeckStep(maxStep, slideId)` hook from `slides/deck-step.js`. Do not attach independent global keyboard listeners inside slide components.
- Each animated slide owns only its internal reveal steps while active; hidden slides must never advance. New slides reset to step 0 when they become active, and held/repeated key presses are ignored until keyup so a slide cannot arrive already fully revealed.

#### Animated Slide Contract
1. Add the slide as a direct `<section id="slide-example">` child of `<deck-stage>`.
2. Render the React component into that same section.
3. Inside the component, call `const step = window.useDeckStep(maxStep, 'slide-example');`.
4. Reveal elements with deterministic gates such as `opacity: step >= 2 ? 1 : 0`.
5. Let `deck-stage` handle slide-to-slide navigation once `step === maxStep`.

Minimal pattern:

```jsx
const ExampleSlide = () => {
  const step = window.useDeckStep(3, 'slide-example');

  return (
    <SlideChrome pageNum={10}>
      <h1>Idea central</h1>
      <div style={{ opacity: step >= 1 ? 1 : 0 }}>Primer elemento</div>
      <div style={{ opacity: step >= 2 ? 1 : 0 }}>Segundo elemento</div>
      <div style={{ opacity: step >= 3 ? 1 : 0 }}>Cierre visual</div>
    </SlideChrome>
  );
};
```

### Arrows
- Curved arc arrows for note relationships (black, clean arrowhead)
- Straight horizontal arrows for flow diagrams (colored per flow)
- Diamond-ended measurement lines (green) for interval spans
- Warning triangle + bug icon for conceptual caution moments

### Cards / Containers
- Dashed-border rectangles for formula definitions
- Solid thin-border rectangles for content groupings
- No rounded corners on boxes. Sharp/square everywhere.
- No drop shadows. Borders only.

### Iconography
See ICONOGRAPHY section below.

### Spacing
- Generous internal margins. Content never touches the slide edges.
- Diagrams take up 60–80% of slide area.
- Text is secondary to visual schema.

### Corner Radii
- **Zero**. All boxes, panels, containers are square-cornered.

### Imagery
- No photography.
- Illustrations are functional diagrams (piano, circle, binary grids).
- Mascot: cartoon bug (beetle-like creature) wearing a sombrero. Appears in "caution" narrative moments. Hand-drawn/clipart style.

---

## ICONOGRAPHY

- **No icon font or icon system** is used.
- **No SVG icon library**.
- **No emoji**.
- The primary "icon" is the **UNAL Escudo** (shield/crest) — institutional logo. Used bottom-right on every content slide, centered on full-green slides.
- **Warning bug**: A cartoon beetle/bug in a warning triangle. Used as a narrative "caution!" marker. Clipart style. Saved in `assets/`.
- **Sombrero bug mascot**: Same beetle, now wearing a sombrero. Appears in diagram slides to add personality. Saved in `assets/`.
- All diagram icons are drawn geometrically with CSS/SVG lines and circles — not from any icon library.

Assets located in: `assets/`

---

## INDEX

```
/
├── README.md                    ← This file
├── SKILL.md                     ← Skill definition for Claude Code
├── colors_and_type.css          ← CSS custom properties: colors + typography
├── assets/
│   ├── escudo-green.png         ← UNAL shield logo (white on green bg)
│   └── cover-bg.png             ← Full green cover slide background
├── preview/
│   ├── colors-primary.html      ← Primary color swatches
│   ├── colors-diagram.html      ← Diagram color palette
│   ├── type-display.html        ← Ancízar Serif specimen
│   ├── type-sans.html           ← Ancízar Sans specimen
│   ├── type-scale.html          ← Full type scale
│   ├── slide-chrome.html        ← Slide frame / chrome system
│   ├── slide-layouts.html       ← Layout variants
│   ├── arrows-connectors.html   ← Arrow and connector styles
│   ├── diagram-elements.html    ← Diagram components
│   └── spacing-tokens.html      ← Spacing system
└── slides/
    ├── index.html               ← Interactive slide deck demo
    ├── deck-stage.js            ← Deck shell: slide navigation, scaling, print layout
    ├── deck-step.js             ← Reusable active-slide step controller for animations
    ├── CoverSlide.jsx           ← Full-green cover
    ├── TitleSlide.jsx           ← White title + subtitle
    ├── ContentSlide.jsx         ← Standard content layout
    ├── DiagramSlide.jsx         ← Schematic/diagram layout
    ├── SplitSlide.jsx           ← Orange/white split panel
    └── ClosingSlide.jsx         ← Full-green closing
```
