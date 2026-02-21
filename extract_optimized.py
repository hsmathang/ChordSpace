import re
import json

# Input file path
INPUT_FILE = r"c:\Users\SANTIAGO\Documents\GitHub\ChordSpace\metodologia_temporal\ESTRUCTURA_MATEMATICA_DETALLADA.md"

# Output files
JSON_OUTPUT = "respuestas_documentadas.json"
MD_OUTPUT = "RESPUESTAS_RECOPILADAS.md"

# Keywords for Type A comments (Investigate)
TYPE_A_KEYWORDS = [
    "trabajo para el notebooklm", "hay que explicar qué es", "que dice la literatura",
    "hay que buscar referencias", "habrá que justificar", "que tan robusto",
    "hay trabajos que", "Notebooklm deberia", "como podemos probar",
    "que tiene que ver con", "como interfiere o apoya", "excelente pregunta", "gran pregunta",
    "revisar la bibliografia", "buscar referentes", "que dicen las referencias",
    "validar que es matemáticamente correcto", "justificarlo con base en teoria"
]

# Keywords for Type B comments (Ignore)
TYPE_B_KEYWORDS = [
    "la notación no es clara", "hay que poner un ejemplo", "estamos usando la palabra",
    "hay que definirlo al lector", "decidir si va en este capítulo", "Ya definimos",
    "asumimos que el lector", "o no sé si va en", "algo más de detalle habrá que decir",
    "dónde colocar", "o no?", "creo que la notacion", "no se si estas proposiciones",
    "revisar la notacion", "falta detalle", "acaso no hay mas"
]

def classify_comment(text):
    text_lower = text.lower()
    
    # Priority check: explicit "notebooklm" mention usually means Type A
    if "notebooklm" in text_lower:
        return "TIPO_A"

    # Check for Type B keywords first (False Negatives prevention)
    for kw in TYPE_B_KEYWORDS:
        if kw.lower() in text_lower:
            return "TIPO_B"
            
    # Check for Type A keywords
    for kw in TYPE_A_KEYWORDS:
        if kw.lower() in text_lower:
            return "TIPO_A"
            
    # Default fallback: If it asks "que?" "como?" or mention "referencias", treat as A
    if "?" in text or "referencia" in text_lower or "literatura" in text_lower:
        return "TIPO_A"
        
    return "TIPO_B" # Default to ignore if unsure and not explicitly technical

def extract_items(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readlines()

    items = []
    santimath_count = 0
    type_a_count = 0
    type_b_count = 0
    native_count = 0
    
    current_section = "Unknown"
    
    for i, line in enumerate(content):
        line = line.strip()
        
        # Track sections
        if line.startswith("#"):
            current_section = line
            
        # 1. Extract santimath comments
        if "Nota de santimath:" in line:
            santimath_count += 1
            match = re.search(r"Nota de santimath:\s*(.*)", line, re.IGNORECASE)
            if match:
                comment_text = match.group(1).strip("() ")
                classification = classify_comment(comment_text)
                
                if classification == "TIPO_A":
                    type_a_count += 1
                    items.append({
                        "id": f"RESP-{len(items)+1:03d}",
                        "tipo": "comentario_santimath",
                        "ubicacion": f"{current_section}, linea {i+1}",
                        "texto_original": comment_text,
                        "clasificacion": "TIPO_A",
                        "notebooks_consultados": ["PDFs_Tesis_Armonia", "PDFs_Tesis_ChordSpace_Math"], # Default suggestion
                        "consulta_enviada": f"{comment_text}\n\nREQUISITO OBLIGATORIO DE CITACIÓN:\n- Responde citando ÚNICAMENTE las FUENTES ORIGINALES (papers, libros, capítulos) que consultaste\n- NO cites \"NotebookLM\" como fuente\n- Incluye para CADA fuente mencionada:\n  * Autor(es) completo(s)\n  * Título completo\n  * Año de publicación\n  * Páginas específicas (si aplica)\n  * Editorial o Journal\n\nFORMATO DE CADA CITA:\nAutor, A. (Año). Título del trabajo. Editorial/Journal. Páginas: X-Y.\n\nEJEMPLO:\nSethares, W. A. (2005). Tuning, Timbre, Spectrum, Scale (2nd ed.). Springer. Páginas: 45-67.\nPlomp, R., & Levelt, W. J. M. (1965). Tonal consonance and critical bandwidth. The Journal of the Acoustical Society of America, 38(4), 548-560.\n\nIMPORTANTE: Si respondes con información de múltiples fuentes, lista TODAS las referencias al final de tu respuesta.",
                        "respuesta_notebooklm": "[PENDIENTE: Consultar manualmente]",
                        "fuentes_originales": []
                    })
                else:
                    type_b_count += 1

        # 2. Extract Native Questions
        # Look for questions that are NOT santimath comments
        if "?" in line and "Nota de santimath" not in line:
            # Check if it's likely a question line (starts with - or is inside a text block)
            # Split by '?' to catch multiple questions in one line
            segments = line.split('?')
            for seg in segments:
                seg = seg.strip()
                if len(seg) > 5 and ( "¿" in seg or seg.strip().lower().startswith(("que", "como", "cuando", "donde", "por que", "cual", "quien", "es ", "seria ", "podria ")) ):
                     # Re-append '?' for readability
                     question_text = seg + "?"
                     native_count += 1
                     items.append({
                        "id": f"RESP-{len(items)+1:03d}",
                        "tipo": "pregunta_nativa",
                        "ubicacion": f"{current_section}, linea {i+1}",
                        "texto_original": question_text.strip("- "),
                        "clasificacion": "NATIVA",
                        "notebooks_consultados": ["PDFs_Tesis_Armonia", "PDFs_Tesis_ChordSpace_Math"],
                        "consulta_enviada": f"{question_text.strip('- ')}\n\nREQUISITO OBLIGATORIO DE CITACIÓN:\n- Responde citando ÚNICAMENTE las FUENTES ORIGINALES (papers, libros, capítulos) que consultaste\n- NO cites \"NotebookLM\" como fuente\n- Incluye para CADA fuente mencionada:\n  * Autor(es) completo(s)\n  * Título completo\n  * Año de publicación\n  * Páginas específicas (si aplica)\n  * Editorial o Journal\n\nFORMATO DE CADA CITA:\nAutor, A. (Año). Título del trabajo. Editorial/Journal. Páginas: X-Y.\n\nEJEMPLO:\nSethares, W. A. (2005). Tuning, Timbre, Spectrum, Scale (2nd ed.). Springer. Páginas: 45-67.\nPlomp, R., & Levelt, W. J. M. (1965). Tonal consonance and critical bandwidth. The Journal of the Acoustical Society of America, 38(4), 548-560.\n\nIMPORTANTE: Si respondes con información de múltiples fuentes, lista TODAS las referencias al final de tu respuesta.",
                        "respuesta_notebooklm": "[PENDIENTE: Consultar manualmente]",
                        "fuentes_originales": []
                    })
            
    return items, santimath_count, type_a_count, type_b_count, native_count

def generate_json(items, stats):
    output_data = {
        "metadata": {
            "paso_0_mcp": {
                "conexion_exitosa": True, 
                "nota": "Conectado pero sin recursos PDFs_Tesis_* expuestos. Requiere consulta manual.",
                "notebooks_encontrados": [],
                "total_notebooks": 0
            },
            "inventario": {
                "total_comentarios_santimath_raw": stats[0],
                "comentarios_tipo_a_investigacion": stats[1],
                "comentarios_tipo_b_omitidos": stats[2],
                "total_preguntas_nativas": stats[3],
                "total_consultas_realizadas": len(items)
            },
            "fecha_consulta": "2026-02-15"
        },
        "respuestas_documentadas": items
    }
    
    with open(JSON_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    return output_data

def generate_markdown(data):
    md_content = f"""# Respuestas Documentadas para Capítulo 3 - Metodología

**Fecha:** {data['metadata']['fecha_consulta']}
**Total comentarios santimath (Raw):** {data['metadata']['inventario']['total_comentarios_santimath_raw']}
**Comentarios Tipo A (Investigados):** {data['metadata']['inventario']['comentarios_tipo_a_investigacion']}
**Comentarios Tipo B (Omitidos):** {data['metadata']['inventario']['comentarios_tipo_b_omitidos']}
**Total preguntas nativas:** {data['metadata']['inventario']['total_preguntas_nativas']}
**Total consultas formuladas:** {data['metadata']['inventario']['total_consultas_realizadas']}

---
"""

    for entry in data['respuestas_documentadas']:
        notebooks = ", ".join(entry['notebooks_consultados'])
        md_content += f"""
## {entry['id']}: {entry['texto_original'][:60]}...

**Ubicación:** {entry['ubicacion']}  
**Tipo:** {entry['tipo']} ({entry['clasificacion']})  
**Notebooks SUGERIDOS:** {notebooks}

### Texto Original
> "{entry['texto_original']}"

### Consulta FORMULADA para NotebookLM
```text
{entry['consulta_enviada']}
```

### Respuesta de NotebookLM
{entry['respuesta_notebooklm']}

### Fuentes Citadas
*(Pendiente de consulta manual)*

---
"""
    with open(MD_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(md_content)

if __name__ == "__main__":
    items, s_count, a_count, b_count, n_count = extract_items(INPUT_FILE)
    stats = (s_count, a_count, b_count, n_count)
    
    print(f"Extraction stats:")
    print(f"Total santimath: {s_count}")
    print(f"Type A (Keep): {a_count}")
    print(f"Type B (Drop): {b_count}")
    print(f"Native Qs: {n_count}")
    print(f"Total Items: {len(items)}")
    
    json_data = generate_json(items, stats)
    generate_markdown(json_data)
    print("Files generated successfully.")
