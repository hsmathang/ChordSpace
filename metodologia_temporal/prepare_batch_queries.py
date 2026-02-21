import json
import re

# Notebook IDs
NB_MATH = "cf0745ab-abf7-4b86-a5cf-48b192f603be"
NB_PSICO = "27d02df9-0405-4ae0-b1d5-58675f73cc49"
NB_HARM = "8dedc0d4-9af1-482a-b779-e9733609414a"
NB_ML = "14fa63f0-279b-4348-b50e-5d350542b25b"
NB_DIMRED = "43913228-e430-45cb-9489-c3b27904f02c"

MANUAL_OVERRIDES = {
    "Q-008": "Explica positivamente qué noción de identidad de acorde se utiliza cuando se rechaza la equivalencia de PC-sets (Forte). ¿Qué propiedades se preservan y cuáles se discriminan?",
    "Q-009": "Justificación para usar 12 bins (intervalos complementarios) en lugar de 6 bins (IC vector) en la representación de acordes. Relación con la rugosidad psicoacústica.",
    "Q-041": "Discute las decisiones críticas de diseño mencionadas en la sección 3.8.4, especificando alternativas descartadas y justificaciones.",
    "Q-012": "Definir concisamente 'Banda Crítica Auditiva' y 'Disonancia Sensorial' y su relación con la rugosidad en acordes.",
}

def get_notebook_id(q, text):
    sec = q.get('seccion', '')
    text_lower = text.lower()
    
    # Hardcoded/Specific logic
    if "3.2" in sec: return NB_PSICO
    if "rugosidad" in text_lower or "sethares" in text_lower or "plomp" in text_lower: return NB_PSICO
        
    if "3.5" in sec or "mds" in text_lower or "umap" in text_lower or "t-sne" in text_lower: return NB_DIMRED
    if "3.7" in sec: return NB_DIMRED

    if "3.3" in sec: return NB_ML
    if "3.6" in sec: return NB_ML
    if "3.4" in sec: return NB_ML
        
    if "3.1" in sec:
        if "acorde" in text_lower or "voicing" in text_lower or "pc-set" in text_lower or "intervalo" in text_lower or "3.1.4" in sec:
            return NB_HARM
        return NB_MATH
        
    if "3.0" in sec: return NB_MATH
    if "3.8" in sec:
        if "rugosidad" in text_lower: return NB_PSICO
        return NB_MATH

    return NB_MATH

def main():
    with open(r"c:\Users\SANTIAGO\Documents\GitHub\ChordSpace\metodologia_temporal\preguntas_identificadas_CORREGIDO.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    output_list = []
    
    for q in data['preguntas']:
        q_id = q.get('id', 'UNKNOWN')
        
        # Get query text
        if q_id in MANUAL_OVERRIDES:
            query = MANUAL_OVERRIDES[q_id]
        else:
            query = q.get('pregunta', '')
            if not query:
                query = q.get('comentario_santimath', '')
                
        if not query or query.strip() == "trabajo para el notebooklm":
             # Fallback if override missed
             if q_id == "Q-008": query = MANUAL_OVERRIDES["Q-008"]
             else:
                 print(f"Warning: Empty query for {q_id}")
                 continue

        nb_id = get_notebook_id(q, query)
        
        citation_req = "\n\nREQUISITO OBLIGATORIO DE CITACIÓN:\n- Responde citando ÚNICAMENTE las FUENTES ORIGINALES.\n- FORMATO: Autor, A. (Año). Título. Editorial. Páginas."
        final_query = query + citation_req
        
        output_list.append({
            "id": q_id,
            "notebook_id": nb_id,
            "query": final_query,
            "original_section": q.get('seccion', '')
        })
        
    with open(r"c:\Users\SANTIAGO\Documents\GitHub\ChordSpace\metodologia_temporal\final_queries_ready.json", "w", encoding="utf-8") as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)
        
    print(f"Generated {len(output_list)} queries in final_queries_ready.json")

if __name__ == "__main__":
    main()
