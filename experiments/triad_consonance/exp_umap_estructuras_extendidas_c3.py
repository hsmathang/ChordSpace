"""
exp_umap_estructuras_extendidas_c3.py
=====================================
Versión del experimento de estructuras extendidas (díadas, tríadas, 7mas, 9nas)
adaptado explícitamente para usar UMAP con las mismas figuras, símbolos y "look an feel"
que el visualizador base de la tesis (exp_utils.py).

Guarda resultados en una subcarpeta de:
  experiments/triad_consonance/outputs/estructuras_extendidas_c3_umap
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_utils import build_entries, run_experiment

OUTPUT_DIR = Path(__file__).parent / "outputs" / "estructuras_extendidas_c3_umap"
JSONL_FILE = Path(__file__).parent / "poblacion_extendida_c3.jsonl"

def build_raw_from_jsonl():
    """Lee el JSON local exportado previamente y extrae los datos puros"""
    chords_raw = []
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            nombre = row['code']
            intervals = row['interval'] 
            notas = row['notes_abs_json']
            freqs = [440.0 * 2**((m - 69) / 12.0) for m in notas]
            chords_raw.append((nombre, intervals, freqs, notas))
    return chords_raw

def main():
    if not JSONL_FILE.exists():
        print(f"[ERROR] Archivo de población dictada no encontrada: {JSONL_FILE}")
        sys.exit(1)
        
    print("[INFO] Cargando la población de 81 estructuras extendidas desde JSONL...")
    chords_raw = build_raw_from_jsonl()
    print(f"[INFO] Población: {len(chords_raw)} acordes.")

    # Convertir a ChordEntry y vectores usando la rigurosa perclass_alpha0_75
    entries, X = build_entries(chords_raw, proposal="perclass_alpha0_75")

    # Mapeo de Categorías vs Símbolos
    facets = []
    symbols = []
    
    for e in entries:
        nombre = e.identity_name
        tipo = nombre.split("-")[0]
        
        # Asignación de categorías IDÉNTICA al originario
        if "Diada" in tipo:
            facets.append("Díadas")
            symbols.append("diamond-open")
        elif tipo == "Maj":
            facets.append("Maj")
            symbols.append("triangle-up-open")
        elif tipo == "Min":
            facets.append("Min")
            symbols.append("triangle-down-open")
        elif tipo == "Dim":
            facets.append("Dim")
            symbols.append("circle-open")
        elif tipo == "Aug":
            facets.append("Aug")
            symbols.append("cross-open")
        else:
            facets.append("Otros Acordes")
            symbols.append("circle")  # Punto CON relleno

    print(f"[INFO] Lanzando pipeline UMAP a la carpeta {OUTPUT_DIR}...")
    
    # 1. Pipeline vista única original
    metrics_single = run_experiment(
        entries=entries,
        X=X,
        metric="euclidean",
        output_dir=OUTPUT_DIR,
        experiment_name="Acordes_Extendidas_C3_UMAP",
        scatter_title="Estructuras Extendidas Ancladas en C3 (Proyección UMAP)",
        n_init=1,  # Irrelevante para UMAP pero requerido en firma
        scatter_mode="faceted",
        facet_labels=facets,
        facet_symbols=symbols,
        facet_layout="single_overview",
        label_font_size=9,
        show_labels=False,
        reducer="umap"
    )
    
    print(f"\n[DONE] Acordes proyectados usando UMAP con éxito. Resultados en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
