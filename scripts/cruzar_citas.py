import os
import re

tex_dir = r"d:\Documents\GitHub\ChordSpace\docs\Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024 pacho"
bib_file = os.path.join(tex_dir, "ReferenciasRugosas.bib")
report_file = os.path.join(tex_dir, "reporte_citas_rotas.txt")

with open(bib_file, "r", encoding="utf-8") as f:
    bib_content = f.read()
bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_content))

keys_cited = set()
cite_pattern = re.compile(r"\\(?:cite|citet|citep|citeA)[a-zA-Z]*\{([^}]+)\}")

for filename in os.listdir(tex_dir):
    if filename.endswith(".tex"):
        with open(os.path.join(tex_dir, filename), "r", encoding="utf-8") as f:
            for line in f:
                for match in cite_pattern.findall(line):
                    for k in match.split(","):
                        keys_cited.add(k.strip())

missing_keys = keys_cited - bib_keys

with open(report_file, "w", encoding="utf-8") as out:
    out.write(f"=== ANÁLISIS DE CITAS ===\n")
    out.write(f"Citas en TEX: {len(keys_cited)}\n")
    out.write(f"Claves en BIB: {len(bib_keys)}\n")
    out.write(f"Citas rotas: {len(missing_keys)}\n\n")
    for k in sorted(missing_keys):
        out.write(f"{k}\n")

print(f"Citas rotas: {len(missing_keys)}")
print("Archivo generado ok.")
