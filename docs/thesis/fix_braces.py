with open('capitulo_metodologia.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip_until = -1
for i, line in enumerate(lines):
    if i < skip_until:
        continue
    if 'decorate, decoration={brace' in line:
        # Skip this line and the next (continuation) line
        if i + 1 < len(lines) and lines[i+1].strip().startswith('('):
            # This is the first of a pair; replace with simple colored bar
            if 'red!60' in line:
                new_lines.append('\\draw[red!60, very thick] (0, -0.5) -- (2, -0.5);\r\n')
                new_lines.append('\\node[font=\\scriptsize, text=red!60, below] at (1, -0.5) {Batimiento};\r\n')
            elif 'orange!60' in line:
                new_lines.append('\\draw[orange!60, very thick] (2, -0.5) -- (5, -0.5);\r\n')
                new_lines.append('\\node[font=\\scriptsize, text=orange!60, below] at (3.5, -0.5) {Rugosidad};\r\n')
            elif 'green!60!black' in line:
                new_lines.append('\\draw[green!60!black, very thick] (5, -0.5) -- (10, -0.5);\r\n')
                new_lines.append('\\node[font=\\scriptsize, text=green!60!black, below] at (7.5, -0.5) {Separacion limpia};\r\n')
            skip_until = i + 2  # skip the continuation line
            continue
    elif '% --- Zones ---' in line:
        new_lines.append('% --- Zones (colored bars + labels) ---\r\n')
        continue
    new_lines.append(line)

with open('capitulo_metodologia.tex', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Done. {len(lines)} -> {len(new_lines)} lines")
