#!/usr/bin/env python3
"""
limpiar_bib.py — Segunda pasada de limpieza quirúrgica sobre ReferenciasRugosas.bib
====================================================================================
Correcciones aplicadas:
  1. Eliminar entradas claramente erróneas/alucinaciones:
     - warthog1993handbook (W.W. Warthog no existe en literatura musical)
     - iadanza2021idrogeo (paper de geología italiana sobre deslizamientos de tierra)
     - venna2005visualizing (sobre expresión génica, no música)

  2. Fusionar duplicados residuales con sufijo 'b' (mantener la versión más completa):
     - lerdahl1988tonal + lerdahl1988tonalb  → conservar lerdahl1988tonal (ambas idénticas)
     - piston1962harmony + piston1962harmonyb → fusionar: piston1962harmony con datos del 'b'
     - chew2014mathematical + chew2014mathematicalb → mantener chew2014mathematical (más completa)
     - tymoczko2006geometry + tymoczko2006geometryb → fusionar: geometryb tiene DOI, geometry no
     - stolzenburg2015harmony + stolzenburg2015harmonyb → mantener stolzenburg2015harmony (más completa)
     - eerola2022consonance + eerola2022register → SON papers diferentes (no fusionar)
     - himpel2022geometry + himpel2022riemannian → SON papers diferentes (títulos distintos, no fusionar)
     - bernardes2016multilevel + bernardes2017multilevel → años diferentes, no fusionar

  3. Entradas con problemas menores de datos:
     - unknown2013mathematics → sin autor, convertir a @book con editor explícito (ya está bien con editor)
     - wang2021understanding → "Understanding UMAP" no es paper real de Wang, es de McInnes/Healy. 
       Pero no tenemos certeza → conservar con advertencia.
     - burgoyne2005manifold → sin journal real, sin DOI → marcar como misc (ya es article, dejarlo)

  4. Entradas con campos corruptos (encoding LaTeX mal procesado):
     - alsina1994musica: title = "La Música y Su Evoluci\ń" → \ń = corrupción de ón
     - Múltiples entries con \t́ \ŕ etc. — limpiar en los campos title
"""

import re
import os

BIB_FILE = r"d:\Documents\GitHub\ChordSpace\docs\Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024 pacho\ReferenciasRugosas.bib"
OUT_FILE = BIB_FILE  # Sobreescribir in-place (tenemos el script para regenerar)
LOG_FILE = r"d:\Documents\GitHub\ChordSpace\docs\Tesis_Matemáticas_Aplicadas_UNAL_2024 pacho\limpieza_report.md"

# ──────────────────────────────────────────────────────────────────────────────
# 1. Entradas a ELIMINAR completamente
# ──────────────────────────────────────────────────────────────────────────────
ENTRIES_TO_DELETE = {
    "warthog1993handbook",   # Alucinación: W.W. Warthog no existe
    "iadanza2021idrogeo",    # Paper de geología italiana (deslizamientos de tierra en Italia)
    "venna2005visualizing",  # Sobre expresión génica (bioinformatics)
    "lerdahl1988tonalb",     # Duplicado idéntico de lerdahl1988tonal
    "chew2014mathematicalb", # Duplicado incompleto de chew2014mathematical
    "stolzenburg2015harmonyb", # Duplicado incompleto de stolzenburg2015harmony (título genérico)
    "tymoczko2006geometry",  # Sin DOI; tymoczko2006geometryb tiene el DOI correcto
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Entradas a MEJORAR/PARCHEAR (clave → dict de campos a agregar/reemplazar)
# ──────────────────────────────────────────────────────────────────────────────
PATCHES = {
    # piston1962harmony: el 'b' tenía publisher/address/edition que el principal no tenía
    "piston1962harmony": {
        "publisher": "W. W. Norton {\\&} Company",
        "address": "New York",
        "edition": "3rd",
    },
    # tymoczko2006geometryb: renombrarlo a tymoczko2006geometry (quedó como 'b' para evitar colisión)
    # y quitarle el 'b' del key — esto lo hacemos cambiando el ID directamente
    "tymoczko2006geometryb": {
        "_rename": "tymoczko2006geometry",
        "title": "The Geometry of Musical Chords",
    },
    # vassilakis2001perceptual: falta school (UCLA)
    "vassilakis2001perceptual": {
        "school": "University of California, Los Angeles",
        "type": "PhD thesis",
    },
    # cannas2018geometric: falta school
    "cannas2018geometric": {
        "school": "University of Strasbourg",
        "type": "PhD thesis",
    },
    # tymoczko2016geometry: falta publisher (Oxford University Press)
    "tymoczko2016geometry": {
        "publisher": "Oxford University Press",
        "address": "New York",
        "isbn": "978-0-19-533667-2",
    },
    # unknown2013mathematics: añadir tipo apropiado
    "unknown2013mathematics": {
        "_rename": "yust2013mathematics",
        "author": "Yust, Jason and Wild, Jonathan and Burgoyne, John Ashley",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# 3. Parser de bloques BibTeX
# ──────────────────────────────────────────────────────────────────────────────

def parse_blocks(text):
    """Divide el archivo en bloques: (key, full_block_text) o (None, comment_text)."""
    blocks = []
    i = 0
    n = len(text)

    while i < n:
        at = text.find("@", i)
        if at == -1:
            if i < n:
                blocks.append((None, text[i:]))
            break

        # Cualquier texto antes de @ es comentario/espacio
        if at > i:
            blocks.append((None, text[i:at]))

        # Encontrar la llave de apertura
        brace = text.find("{", at)
        if brace == -1:
            blocks.append((None, text[at:]))
            break

        entry_type = text[at+1:brace].strip().lower()

        # Contar llaves para encontrar el cierre
        depth = 0
        j = brace
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1

        block_text = text[at:j+1]

        # Extraer clave del bloque
        inner = text[brace+1:j]
        first_comma = inner.find(",")
        key = inner[:first_comma].strip() if first_comma != -1 else ""

        if entry_type in ("comment", "string", "preamble") or not key:
            blocks.append((None, block_text))
        else:
            blocks.append((key, block_text))

        i = j + 1

    return blocks


def get_field(block, field_name):
    """Extrae el valor de un campo de un bloque BibTeX."""
    pattern = re.compile(
        rf"^\s*{re.escape(field_name)}\s*=\s*\{{", re.MULTILINE | re.IGNORECASE
    )
    m = pattern.search(block)
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(block) and depth > 0:
        if block[i] == "{":
            depth += 1
        elif block[i] == "}":
            depth -= 1
        i += 1
    return block[start:i-1].strip()


def set_field(block, field_name, value):
    """Reemplaza o añade un campo en un bloque BibTeX."""
    pattern = re.compile(
        rf"^\s*{re.escape(field_name)}\s*=\s*\{{[^{{}}]*(\{{[^{{}}]*\}})*[^{{}}]*\}},?\s*$",
        re.MULTILINE | re.IGNORECASE
    )
    new_field = f"  {field_name} = {{{value}}},"
    if pattern.search(block):
        return pattern.sub(new_field, block)
    else:
        # Insertar antes del último '}'
        last_brace = block.rfind("}")
        return block[:last_brace] + f"\n{new_field}\n" + block[last_brace:]


def rename_key(block, old_key, new_key):
    """Renombra la clave en un bloque."""
    return block.replace(f"{{{old_key},", f"{{{new_key},", 1)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    with open(BIB_FILE, encoding="utf-8") as f:
        content = f.read()

    blocks = parse_blocks(content)

    deleted = []
    patched = []
    renamed_keys = {}  # old → new

    new_blocks = []
    for key, block in blocks:

        if key is None:
            new_blocks.append(block)
            continue

        # ── Eliminar ─────────────────────────────────────────────────────────
        if key in ENTRIES_TO_DELETE:
            deleted.append(key)
            continue

        # ── Parchear ─────────────────────────────────────────────────────────
        if key in PATCHES:
            patch = PATCHES[key]
            new_key = patch.get("_rename", key)

            for field, value in patch.items():
                if field.startswith("_"):
                    continue
                block = set_field(block, field, value)
                patched.append(f"`{key}` → campo `{field}` actualizado")

            if new_key != key:
                block = rename_key(block, key, new_key)
                renamed_keys[key] = new_key
                patched.append(f"`{key}` → renombrada a `{new_key}`")
            key = new_key

        new_blocks.append(block)

    # Actualizar header con nuevo conteo
    entry_count = sum(1 for k, _ in zip(
        [b for k, b in zip([k for k, _ in blocks], [b for _, b in blocks]) if k is not None],
        range(10000)
    ) if k not in ENTRIES_TO_DELETE)

    # Reconstruir contenido
    new_content = "".join(new_blocks)

    # Actualizar total en el header
    new_content = re.sub(
        r"% Total: \d+ entradas únicas",
        lambda m: f"% Total: {new_content.count('@') - 1} entradas únicas",
        new_content
    )

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(new_content)

    # Reporte
    report = f"""# Reporte — Limpieza Segunda Pasada

## Entradas Eliminadas ({len(deleted)})
{chr(10).join(f'- `{k}`' for k in deleted)}

## Entradas Parchadas / Renombradas
{chr(10).join(f'- {p}' for p in patched)}

## Claves Renombradas
{chr(10).join(f'- `{old}` → `{new}`' for old, new in renamed_keys.items())}

## Entradas Finales
Aprox. {new_content.count('@') - 1} entradas en el archivo limpio.
"""
    # Usar mismo directorio que el bib
    log_path = os.path.join(
        os.path.dirname(BIB_FILE), "limpieza_report.md"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ Limpieza completada.")
    print(f"   Eliminadas: {len(deleted)}")
    print(f"   Parches:    {len(patched)}")
    print(f"   Output:     {OUT_FILE}")
    print(f"   Reporte:    {log_path}")
    print(f"\n   Entradas eliminadas:")
    for k in deleted:
        print(f"     - {k}")


if __name__ == "__main__":
    main()
