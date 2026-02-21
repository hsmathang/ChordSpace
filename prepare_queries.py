import re
import json

INPUT_FILE = r"c:\Users\SANTIAGO\Documents\GitHub\ChordSpace\RESPUESTAS_RECOPILADAS.md"
OUTPUT_FILE = r"c:\Users\SANTIAGO\Documents\GitHub\ChordSpace\queries_to_run.json"

# Notebook IDs
NB_PSICO = "27d02df9-0405-4ae0-b1d5-58675f73cc49" # PDFs_Tesis_Psicoacustica
NB_MATH = "cf0745ab-abf7-4b86-a5cf-48b192f603be"  # Tesis_ChordSpace_Math
NB_HARM = "8dedc0d4-9af1-482a-b779-e9733609414a"  # PDFs_Tesis_Armonia
NB_COMP = "14fa63f0-279b-4348-b50e-5d350542b25b"  # PDFs_Tesis_Computacion_ML

def assign_notebook(text):
    text_lower = text.lower()
    
    # Psicoacustica (High Priority for Ch 3)
    if any(k in text_lower for k in ["sethares", "plomp", "levelt", "rugosidad", "roughness", "disonancia", "banda crítica", "frecuencia", "timbre", "auditivo", "fisiolo", "sensorial"]):
        return NB_PSICO
    
    # Math & Geometry
    if any(k in text_lower for k in ["conjunto", "espacio", "vector", "distancia", "métrica", "topología", "álgebra", "grupo", "anillo", "definición", "lema", "proposición", "axioma", "simplex", "euclidiana", "coseno", "jensen"]):
        return NB_MATH
        
    # ML & Computation
    if any(k in text_lower for k in ["clasificación", "clustering", "mds", "umap", "tsne", "t-sne", "isomap", "algoritmo", "python", "librería", "computacional", "feature", "pca", "aprendizaje", "learning"]):
        return NB_COMP

    # Harmony
    if any(k in text_lower for k in ["acorde", "armonía", "tonal", "sustitución", "funcional", "jazz", "forte", "pc-set", "inversión", "voz", "conducción", "musical", "intervalo"]):
        return NB_HARM
        
    return NB_PSICO # Default

def parse_markdown():
    with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Regex to find each block
    # Pattern looks for ## RESP-XXX ... ### Consulta FORMULADA ... ```text ... ```
    pattern = re.compile(r"## (RESP-\d+|RESP-\d+ \([^\)]+\)): (.+?)\n\n\*\*Ubicación:\*\* (.+?)\n\*\*Tipo:\*\* (.+?)\n\n.*?### Consulta FORMULADA para NotebookLM\n```text\n(.*?)\n```", re.DOTALL)
    
    matches = pattern.findall(content)
    queries = []
    
    for match in matches:
        resp_id = match[0]
        title = match[1]
        location = match[2]
        q_type = match[3]
        query_text = match[4].strip()
        
        notebook_id = assign_notebook(query_text + " " + title)
        
        queries.append({
            "id": resp_id,
            "title": title,
            "query": query_text,
            "notebook_id": notebook_id
        })
        
    print(f"Found {len(queries)} queries.")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parse_markdown()
