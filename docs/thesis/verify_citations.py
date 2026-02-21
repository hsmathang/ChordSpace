import re

tex_file = r"c:\Users\SANTIAGO\Documents\GitHub\ChordSpace\docs\thesis\capitulo_metodologia.tex"
bib_file = r"c:\Users\SANTIAGO\Documents\GitHub\ChordSpace\docs\thesis\referencias_metodologia_vf.bib"

# Extract keys from tex
with open(tex_file, 'r', encoding='utf-8') as f:
    tex_content = f.read()

cite_keys = []
# Match \cite{key}, \cite{key1, key2}, \citep{...}, \citet{...}
matches = re.findall(r'\\cite[a-z]*\{([^}]+)\}', tex_content)
for m in matches:
    keys = [k.strip() for k in m.split(',')]
    cite_keys.extend(keys)

unique_keys = sorted(list(set(cite_keys)))

# Extract keys from bib
bib_keys = []
with open(bib_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.startswith('@') and '{' in line:
            # Entry start: @article{key,
            parts = line.split('{', 1)
            if len(parts) > 1:
                key_part = parts[1].split(',')[0].strip()
                bib_keys.append(key_part)

bib_keys_set = set(bib_keys)

missing_keys = []
for k in unique_keys:
    if k not in bib_keys_set:
        missing_keys.append(k)

print(f"Found {len(unique_keys)} unique citation keys in .tex")
print(f"Found {len(bib_keys)} keys in .bib")

if missing_keys:
    print("MISSING KEYS:")
    for k in missing_keys:
        print(f" - {k}")
else:
    print("SUCCESS: All keys found in bibliography.")
