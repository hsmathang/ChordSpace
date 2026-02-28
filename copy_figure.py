import shutil
src = r"C:\Users\Admin\.gemini\antigravity\brain\f5050e3f-9b64-4045-ab22-b49961bc48e4\pipeline_psicoacustico_metodologia_1771996567029.png"
dst = r"d:\Documents\GitHub\ChordSpace\docs\Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024 (1)\00Figuras\pipeline_conceptual.png"
shutil.copy2(src, dst)
print("Copiado con exito a:", dst)
