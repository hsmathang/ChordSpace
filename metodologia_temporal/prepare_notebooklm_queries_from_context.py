import json
from datetime import date
from pathlib import Path

INPUT_PATH = Path("metodologia_temporal/preguntas_CONTEXTUALIZADAS.json")
OUTPUT_PATH = Path("metodologia_temporal/notebooklm_queries_contextualizadas.json")

PERSONA_AND_RULES = """ROL:
Eres mi asesor de tesis (maestría en matemáticas aplicadas) para el proyecto ChordSpace.

REGLAS:
1) Usa SOLO los documentos de ESTE notebook. Prohibido conocimiento externo.
2) No inventes: si no hay evidencia suficiente, dilo y pide el PDF/fuente faltante.
3) Cita fuentes originales (papers/libros) cuando existan en el notebook.
4) Incluye páginas o sección del documento para cada afirmación importante.
5) Entrega BibTeX válido por fuente.
6) Escribe en español académico (breve, denso, orientado a metodología).
"""

GLOBAL_CONTEXT = """CONTEXTO GLOBAL (ChordSpace / Metodología):
- Objetivo: espacio de representación de acordes para explorar/sugerir sustituciones por similitud sonora.
- Dominio: MIDI n∈{0..127}, 12‑TET (A4=440 Hz), f(n)=440·2^((n-69)/12).
- Acorde: tupla estrictamente creciente (sin unísonos MIDI); identidad sensible a registro/voicing (no PC-sets).
- Feature: rugosidad/disonancia sensorial (Plomp–Levelt + Sethares), tonos complejos con parciales armónicos (H=6, δ=0.88).
- Representación: Φ_raw∈R_{≥0}^{12} por clase de intervalo, sin colapsar complementarios (intervalo 0→índice 11).
- Pipeline: población→Φ_raw→normalización→distancia ρ→matriz D→embedding 2D (MDS/UMAP/…)→evaluación y supuestos.
"""

OUTPUT_CONTRACT = """FORMATO DE SALIDA (OBLIGATORIO):
1) RESPUESTA PARA INSERTAR (máx. 180–250 palabras)
   - Lista para pegar en tesis (metodología).
   - Usa citas tipo \\cite{claveBib}.

2) AFIRMACIONES + EVIDENCIA (5–10 ítems)
   - Cada ítem: afirmación breve + (Fuente, año, páginas/ubicación).

3) BIBTEX (mínimo 3 fuentes, máximo 7)
   - Para cada fuente: un bloque BibTeX.
   - Incluye pages cuando aplique; si no hay páginas, explica por qué.

4) NOTAS DEL ASESOR (3–6 bullets)
   - Objeciones típicas de jurado + cómo mitigarlas.

5) PREGUNTAS DE SEGUIMIENTO (0–3)
   - Solo si falta info crítica en los documentos.
"""

TOPIC_NOTEBOOK_HINTS = {
    "psicoacustica": [
        "PDF_thesis_Psicoacustica",
        "PDFs_Tesis_Psicoacustica",
        "Psicoacustica",
        "Roughness",
        "Sethares",
        "Plomp",
    ],
    "math": [
        "PDF_thesis_Math",
        "PDFs_Tesis_Math",
        "Math",
        "Matem",
        "Topology",
        "Metric",
        "ChordSpace",
    ],
    "armonia": [
        "PDF_thesis_Armonia",
        "PDFs_Tesis_Armonia",
        "Armonia",
        "Harmony",
        "Forte",
        "Voicing",
    ],
    "reduccion_dimensionalidad": [
        "PDF_thesis_Dimensionalidad",
        "PDFs_Tesis_Dimensionalidad",
        "Dimensionalidad",
        "DimRed",
        "UMAP",
        "MDS",
        "t-SNE",
        "Manifold",
    ],
}


def build_query(question_reformulada: str) -> str:
    return (
        PERSONA_AND_RULES
        + "\n"
        + GLOBAL_CONTEXT
        + "\nCONTEXTO/PREGUNTA:\n"
        + question_reformulada.strip()
        + "\n\n"
        + OUTPUT_CONTRACT
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    items = data["preguntas_contextualizadas"]

    queries = []
    skips = []

    for q in items:
        q_id = q["id"]
        q_type = q["tipo"]

        if q_type == "B":
            skips.append(
                {
                    "id": q_id,
                    "tipo": "B",
                    "ubicacion_original": q.get("ubicacion_original", ""),
                    "pregunta_original": q.get("pregunta_original", ""),
                    "razon_skip": q.get("razon_skip", ""),
                    "accion_sugerida": q.get("accion_sugerida", ""),
                }
            )
            continue

        topic = q.get("tema_notebook_sugerido", "math")
        hints = TOPIC_NOTEBOOK_HINTS.get(topic, TOPIC_NOTEBOOK_HINTS["math"])
        query_text = build_query(q["pregunta_reformulada_para_notebooklm"])

        queries.append(
            {
                "id": q_id,
                "tipo": "A",
                "tema": topic,
                "notebook_hints": hints,
                "query": query_text,
                "ubicacion_original": q.get("ubicacion_original", ""),
                "seccion_completa": q.get("seccion_completa", ""),
                "para_redactor": q.get("para_redactor", {}),
            }
        )

    out = {
        "metadata": {
            "archivo_preguntas": str(INPUT_PATH).replace("\\", "/"),
            "fecha": date.today().isoformat(),
            "total": len(items),
            "total_tipo_a": len(queries),
            "total_tipo_b": len(skips),
        },
        "queries": queries,
        "skips": skips,
    }

    OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(queries)} query packets to {OUTPUT_PATH}")
    print(f"Wrote {len(skips)} skips to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
