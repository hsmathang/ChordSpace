import os

file_path = r"c:\Users\SANTIAGO\Documents\GitHub\ChordSpace\docs\thesis\capitulo_metodologia.tex"

replacements = {
    "Milne2023": "milneEvidenceUniversalAssociation2023",
    "Masina2022": "masinaDyadsConsonanceDissonance2022",
    "Masina2023": "masinaTriadsConsonanceDissonance2023",
    "MasinaLoPresti2024": "masinaTriadsConsonanceDissonance2024",
    "Masina2024": "masinaTriadsConsonanceDissonance2024",
    "Lazzari2023": "lazzariPitchclass2vecSymbolicMusic2023",
    "Himpel2022": "himpelGeometryMusicPerception2022",
    "Tymoczko2006": "tymoczkoGeometryMusicalChords2006a",
    "Chew2014": "chewMathematicalComputationalModeling2014",
    "DeBerardinis2023": "deberardinisHarmonicMemoryKnowledge2023",
    "Burgoyne2005": "burgoyneVISUALIZATIONLOWDIMENSIONAL",
    "NavarroCaceres2020": "navarro-caceresComputationalModelTonal2020",
    "Stolzenburg2015": "stolzenburgHarmonyPerceptionPeriodicity2015",
    "McDermott2016": "mcdermottIndifferenceDissonanceNative2016",
    "Karystinaios2022": "karystinaiosCADENCEDETECTIONSYMBOLIC2022",
    "DeHaas2011": "dehaasComparingApproachesSimilarity2011a",
    "DeHaas2013": "dehaasGeometricalDistanceMeasure2013",
    "bibliography{referencias_metodologia}": "bibliography{referencias_metodologia_vf}"
}

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

for old, new in replacements.items():
    content = content.replace(old, new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Replacements completed.")
