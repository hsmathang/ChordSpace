import sys
import re

tex_file = r"d:\Documents\GitHub\ChordSpace\docs\Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024 pacho\01Seccion01.tex"
bib_file = r"d:\Documents\GitHub\ChordSpace\docs\Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024 pacho\ReferenciasRugosas.bib"

with open(r"d:\Documents\GitHub\ChordSpace\out.txt", "w", encoding="utf-8") as out:
    try:
        tex_content = open(tex_file, "r", encoding="utf-8").read()
        bib_content = open(bib_file, "r", encoding="utf-8").read()
        
        cites = set()
        for match in re.finditer(r'\\cite\{([^}]+)\}', tex_content):
            keys = [k.strip() for k in match.group(1).split(',')]
            cites.update(keys)
        
        bib_keys = set()
        for match in re.finditer(r'@\w+\s*\{([^,]+)', bib_content):
            bib_keys.add(match.group(1).strip())
        
        missing = cites - bib_keys
        out.write(f"Missing citations: {missing}\n")
        out.write(f"Total found citations: {len(cites)}\n")
    except Exception as e:
        out.write(str(e))
