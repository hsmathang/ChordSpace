import os, sys
bib_path = os.path.join(
    r'd:\Documents\GitHub\ChordSpace\docs',
    'Tesis_Maestr\u00eda_Matem\u00e1ticas_Aplicadas_UNAL_2024 pacho',
    'ReferenciasRugosas.bib'
)
with open(bib_path, 'rb') as f:
    txt = f.read().decode('utf-8', 'ignore').lower()

# Keys suggested by NotebookLM for oraciones 18-28
keys = {
    'mcdermott2016': 'mcdermott',
    'milne2023': 'milne',
    'terhardt1974': 'terhardt',
    'parncutt2011': 'parncutt',
    'eerola2021': 'eerola',
    'roberts1986': 'roberts',
    'leman2000': 'leman',
    'bernardes2016': 'bernardes',
    'sethares1993': 'sethares',
    'sethares2005': 'sethares, w. a. (2005)',
    'harrison2020': 'harrison',
    'gaulhiac_harmonic': 'gaulhiac',
    'hekmati2021': 'hekmati',
    'cambouropoulos2016': 'cambouro',
    'cook2009': 'cook',
    'mcleod2021': 'mcleod',
}

print("=== VERIFICACION DE CLAVES EN ReferenciasRugosas.bib ===\n")
for label, fragment in keys.items():
    status = "✅ SI EXISTE" if fragment in txt else "❌ FALTA"
    print(f"  {status}  — buscando '{fragment}' → clave sugerida: {label}")
