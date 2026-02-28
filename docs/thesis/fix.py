with open('capitulo_metodologia.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

replacement = [
    "% --- Zones (colored bars + labels) ---\n",
    "\\draw[red!60, very thick] (0, -0.5) -- (2, -0.5);\n",
    "\\node[font=\\scriptsize, text=red!60, below] at (1, -0.5) {Batimiento};\n",
    "\\draw[orange!60, very thick] (2, -0.5) -- (5, -0.5);\n",
    "\\node[font=\\scriptsize, text=orange!60, below] at (3.5, -0.5) {Rugosidad};\n",
    "\\draw[green!60!black, very thick] (5, -0.5) -- (10, -0.5);\n",
    "\\node[font=\\scriptsize, text=green!60!black, below] at (7.5, -0.5) {Separación limpia};\n"
]

# We are replacing lines 170 to 176 (index 170 to 176 inclusive, which is 170:177 in slice)
# Let's verify that index 170 is "% --- Zones ---"
if "% --- Zones ---" in lines[170]:
    lines = lines[:170] + replacement + lines[177:]
    with open('capitulo_metodologia.tex', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("Replaced successfully.")
else:
    print("Index mismatch. Found:", lines[170])
