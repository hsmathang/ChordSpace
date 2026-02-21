import os

file_path = r"c:\Users\SANTIAGO\Documents\GitHub\ChordSpace\docs\thesis\capitulo_metodologia.tex"

replacements = {
    "(Forte, 1973)": r"\cite{AllenForteSTRUCTURE}",
    "Forte (1973)": r"\cite{AllenForteSTRUCTURE}",
    "(Sethares, 1993)": r"\cite{setharesLocalConsonanceRelationship1993}",
    "Sethares (1993)": r"\cite{setharesLocalConsonanceRelationship1993}",
    "(Sethares, 1993; Plomp & Levelt, 1965)": r"\cite{setharesLocalConsonanceRelationship1993, Plomp1965}",
    "Plomp y Levelt (1965)": r"\cite{Plomp1965}",
    "Plomp & Levelt (1965)": r"\cite{Plomp1965}",
    "Harrison y Pearce (2020)": r"\cite{harrisonRepresentingHarmonyComputational2020}",
    "Harrison & Pearce (2020)": r"\cite{harrisonRepresentingHarmonyComputational2020}",
    "(Harrison & Pearce, 2020)": r"\cite{harrisonRepresentingHarmonyComputational2020}",
    "Milne et al. (2023)": r"\cite{milneEvidenceUniversalAssociation2023}",
    "(Milne et al., 2023)": r"\cite{milneEvidenceUniversalAssociation2023}",
    "(Masina, 2022)": r"\cite{masinaDyadsConsonanceDissonance2022}",
    "Himpel (2022)": r"\cite{himpelGeometryMusicPerception2022}",
    "Lazzari et al. (2023)": r"\cite{lazzariPitchclass2vecSymbolicMusic2023}",
    "(Amari, 2016)": r"\cite{Amari2016}",
    "(Endres & Schindelin, 2003)": r"\cite{Endres2003}",
    "Burgoyne y Saul (2005)": r"\cite{burgoyneVISUALIZATIONLOWDIMENSIONAL}",
    "(Burgoyne & Saul, 2005)": r"\cite{burgoyneVISUALIZATIONLOWDIMENSIONAL}",
    "(Tymoczko, 2011)": r"\cite{tymoczkoGeometryMusicHarmony2011}",
    "Tymoczko (2011)": r"\cite{tymoczkoGeometryMusicHarmony2011}",
    "(Borg & Groenen, 2005)": r"\cite{Borg2005}",
    "(Cox & Cox, 2001)": r"\cite{Cox2001}",
    "(Venna & Kaski, 2006)": r"\cite{Venna2006}",
    "(McInnes et al., 2018)": r"\cite{McInnes2018}",
    "(Coenen & Pearce, 2019)": r"\cite{Coenen2019}",
    "(Wang et al., 2021)": r"\cite{Wang2021}",
    "(Micchi et al., 2020)": r"\cite{Micchi2020}",
    "bibliography{referencias_metodologia}": "bibliography{referencias_metodologia_vf}"
}

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

for old, new in replacements.items():
    content = content.replace(old, new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Replacements completed.")
