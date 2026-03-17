import re
tex_file = r"d:\Documents\GitHub\ChordSpace\docs\Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024 pacho\01Seccion01.tex"
bib_file = r"d:\Documents\GitHub\ChordSpace\docs\Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024 pacho\ReferenciasRugosas.bib"

with open(tex_file, "r", encoding="utf-8") as f:
    tex = f.read()

with open(bib_file, "r", encoding="utf-8") as f:
    bib = f.read()

cites = set()
for match in re.finditer(r'\\cite\{([^}]+)\}', tex):
    keys = [k.strip() for k in match.group(1).split(',')]
    cites.update(keys)

bib_keys = set()
for match in re.finditer(r'@\w+\s*\{([^,]+)', bib):
    bib_keys.add(match.group(1).strip())

missing = cites - bib_keys

with open("out.txt", "w", encoding="utf-8") as f:
    f.write(f"Missing: {missing}\n")
    f.write(f"Total found: {len(cites)}\n")
