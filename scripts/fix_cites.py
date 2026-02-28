import os
import re

tex_dir = r"d:\Documents\GitHub\ChordSpace\docs\Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024 pacho"

replacements = {
    r"Tymoczko2016GeometryOM": "tymoczko2016geometry",
    r"setharesLocalConsonanceRelationship1993": "sethares1993local",
    r"Plomp1965": "plomp1965tonal",
    r"AllenForteSTRUCTURE": "forte1973structure",
    r"Borg2005": "borg2005modern",
    r"Venna2006": "venna2006local",
    r"McInnes2018": "mcinnes2018umap",
    r"milneEvidenceUniversalAssociation2023": "milne2023evidence",
    r"harrisonRepresentingHarmonyComputational2020": "harrison2020representing",
    r"masinaDyadsConsonanceDissonance2022": "masina2022dyads",
    r"Cohn1997": "cohn1997neo",
    r"Eerola2022": "eerola2022music",
    r"Bowling2018": "bowling2018vocal",
    r"Bowling2018VocalSimilarity": "bowling2018vocal",
    r"himpelGeometryMusicPerception2022": "himpel2022geometry",
    r"himpel": "himpel2022geometry",
    r"callenderGeneralizedVoiceLeadingSpaces2008": "callender2008generalized",
    r"Yust2022": "yust2022geometry",
    r"Krumhansl1990": "krumhansl1990cognitive",
    r"tymoczkoGeometryMusicalChords2006": "tymoczko2006geometry",
    r"lewin1987generalized": "lewin1987generalized",
    r"helmholtz1875sensations": "helmholtz1875sensations",
    r"mcdermott2016indifference": "mcdermott2016indifference",
    r"huang2016chordripple": "huang2016chordripple",
    r"bernardes2016multi": "bernardes2016multi",
    r"DeCastroKorgi2010": "decastrokorgi2010",
    r"plomp1965tonal": "plomp1965tonal",
}

for filename in os.listdir(tex_dir):
    if filename.endswith(".tex"):
        filepath = os.path.join(tex_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        def sub_cite(match):
            keys = match.group(1).split(",")
            new_keys = []
            for k in keys:
                k_strip = k.strip()
                new_keys.append(replacements.get(k_strip, k_strip))
            return match.group(0).replace(match.group(1), ", ".join(new_keys))
        
        cite_pattern = re.compile(r"\\(?:cite|citet|citep|citeA)[a-zA-Z]*\{([^}]+)\}")
        new_content = cite_pattern.sub(sub_cite, content)

        if content != new_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Fixed citations in {filename}")
