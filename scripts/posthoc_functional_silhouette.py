"""
Script post-hoc para calcular Silhouette Score sobre particiones funcionales.

Uso:
    python scripts/posthoc_functional_silhouette.py <population.jsonl> <escenario_idx>

Este script NO modifica el pipeline. Lee los archivos de salida de un
experimento ya ejecutado y calcula Silhouette sobre:
  1. Cualidad armónica (Mayor / Menor / Disminuido / Aumentado / Otro)
  2. Función armónica (Tónica / Subdominante / Dominante) — solo diatónicos

Requisitos: numpy, pandas, scikit-learn, scipy
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import silhouette_score
from collections import Counter

# ── Pitch class helpers ──────────────────────────────────────────────────────

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Intervalos que definen cada cualidad de tríada (desde la fundamental)
QUALITY_INTERVALS = {
    "Mayor":       frozenset([0, 4, 7]),
    "Menor":       frozenset([0, 3, 7]),
    "Disminuido":  frozenset([0, 3, 6]),
    "Aumentado":   frozenset([0, 4, 8]),
    "sus4":        frozenset([0, 5, 7]),
    "sus2":        frozenset([0, 2, 7]),
}

# Funciones armónicas en Do Mayor (diatónico)
# I=C, ii=Dm, iii=Em, IV=F, V=G, vi=Am, vii°=Bdim
DIATONIC_FUNCTIONS_C = {
    # (root_pc, quality) -> función
    (0, "Mayor"):      "T",   # I  = C Major  → Tónica
    (2, "Menor"):      "S",   # ii = D minor  → Subdominante
    (4, "Menor"):      "T",   # iii= E minor  → Tónica
    (5, "Mayor"):      "S",   # IV = F Major  → Subdominante
    (7, "Mayor"):      "D",   # V  = G Major  → Dominante
    (9, "Menor"):      "T",   # vi = A minor  → Tónica
    (11, "Disminuido"): "D",  # vii°= B dim   → Dominante
}


def midi_to_pc(midi_note: int) -> int:
    """Convierte nota MIDI a pitch class (0-11)."""
    return midi_note % 12


def get_quality(intervals_from_root: frozenset) -> str:
    """Determina la cualidad de una tríada dados sus intervalos desde la raíz."""
    for name, pattern in QUALITY_INTERVALS.items():
        if intervals_from_root == pattern:
            return name
    return "Otro"


def classify_chord(notes_abs: List[int]) -> Dict[str, Optional[str]]:
    """
    Clasifica un acorde por cualidad y función armónica.
    
    Args:
        notes_abs: Lista de notas MIDI absolutas, ordenadas ascendentemente.
    
    Returns:
        Dict con 'quality', 'function', 'root_name', 'root_pc'
    """
    if len(notes_abs) < 3:
        return {"quality": "Díada", "function": None, "root_name": None, "root_pc": None}
    
    # Tomar las primeras 3 notas (para tríadas)
    triad_notes = sorted(notes_abs[:3])
    root_pc = midi_to_pc(triad_notes[0])
    
    # Calcular intervalos desde la raíz (en pitch classes)
    intervals = frozenset(midi_to_pc(n - triad_notes[0]) for n in triad_notes)
    
    quality = get_quality(intervals)
    
    # Función armónica (solo si es diatónico en Do Mayor)
    func = DIATONIC_FUNCTIONS_C.get((root_pc, quality), None)
    
    return {
        "quality": quality,
        "function": func,
        "root_name": NOTE_NAMES[root_pc],
        "root_pc": root_pc,
    }


def load_population(jsonl_path: str) -> pd.DataFrame:
    """Carga la población desde un archivo JSONL."""
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


def extract_notes_abs(row) -> Optional[List[int]]:
    """Extrae notas absolutas MIDI de una fila de la población."""
    # Intentar distintos campos que podrían contener las notas
    for field in ["notes_abs", "notes_midi", "midi_notes"]:
        val = row.get(field)
        if val is not None:
            if isinstance(val, str):
                val = json.loads(val)
            return [int(n) for n in val]
    
    # Si tiene frequencies, derivar MIDI de ellas
    freqs = row.get("frequencies")
    if freqs is not None:
        if isinstance(freqs, str):
            freqs = json.loads(freqs)
        # f = 440 * 2^((n-69)/12)  →  n = 69 + 12 * log2(f/440)
        return [int(round(69 + 12 * np.log2(f / 440.0))) for f in freqs]
    
    return None


def compute_silhouette_for_partition(
    embedding: np.ndarray,
    labels: np.ndarray,
    partition_name: str
) -> Optional[float]:
    """Calcula Silhouette para una partición dada, con manejo de errores."""
    mask = labels != "None"
    labels_clean = labels[mask]
    emb_clean = embedding[mask]
    
    unique = np.unique(labels_clean)
    if len(unique) < 2:
        print(f"  ⚠️  {partition_name}: solo {len(unique)} grupo(s), necesita ≥2. Skipping.")
        return None
    
    counts = Counter(labels_clean)
    print(f"  📊 {partition_name} — distribución: {dict(counts)}")
    
    try:
        sil = silhouette_score(emb_clean, labels_clean)
        return float(sil)
    except Exception as e:
        print(f"  ❌ Error calculando Silhouette para {partition_name}: {e}")
        return None


def main():
    """
    Uso interactivo: si no se proporcionan argumentos, solicita la ruta.
    
    También puede usarse como módulo:
        from posthoc_functional_silhouette import classify_chord, compute_silhouette_for_partition
    """
    
    if len(sys.argv) < 2:
        print("=" * 70)
        print("SILHOUETTE POST-HOC: Cualidad y Función Armónica")
        print("=" * 70)
        print()
        print("Este script necesita DOS inputs:")
        print("  1. El archivo .jsonl de la población (se guarda como archivo temporal)")
        print("  2. Un archivo .npy o .csv con el embedding 2D")
        print()
        print("ALTERNATIVA: puedes usarlo como módulo de Python.")
        print()
        print("  Ejemplo de uso manual en Python:")
        print("  >>> import numpy as np")
        print("  >>> from scripts.posthoc_functional_silhouette import classify_chord")
        print("  >>> classify_chord([60, 64, 67])  # Do Mayor")
        print("  {'quality': 'Mayor', 'function': 'T', 'root_name': 'C', 'root_pc': 0}")
        print()
        
        # Demo rápido
        print("── Demo: Clasificación de tríadas diatónicas de Do Mayor ──")
        print()
        diatonic_triads = [
            ([60, 64, 67], "I   Do Mayor"),
            ([62, 65, 69], "ii  Re menor"),
            ([64, 67, 71], "iii Mi menor"),
            ([65, 69, 72], "IV  Fa Mayor"),
            ([67, 71, 74], "V   Sol Mayor"),
            ([69, 72, 76], "vi  La menor"),
            ([71, 74, 77], "vii° Si disminuido"),
        ]
        
        print(f"  {'Grado':<22} {'Cualidad':<14} {'Función':<8} {'Raíz'}")
        print(f"  {'─' * 22} {'─' * 14} {'─' * 8} {'─' * 6}")
        for notes, name in diatonic_triads:
            result = classify_chord(notes)
            func_label = {"T": "Tónica", "S": "Subdom.", "D": "Dominante"}.get(result["function"], "—")
            print(f"  {name:<22} {result['quality']:<14} {func_label:<8} {result['root_name']}")
        
        print()
        print("── Guía de interpretación del Silhouette Score ──")
        print()
        print("  Rango        │ Interpretación")
        print("  ─────────────┼──────────────────────────────────────────")
        print("  0.71 – 1.00  │ Estructura fuerte: clusters bien separados")
        print("  0.51 – 0.70  │ Razonable: clusters distinguibles")
        print("  0.26 – 0.50  │ Estructura débil: overlap considerable")
        print("  0.00 – 0.25  │ Sin estructura clara")
        print("  < 0.00       │ Los puntos están en el grupo equivocado")
        print()
        print("  Para CUALIDAD armónica: esperar 0.15–0.40 sería un buen resultado.")
        print("  Para FUNCIÓN armónica: incluso 0.10–0.25 sería significativo,")
        print("  porque significaría que la rugosidad 'sabe' algo sobre funciones")
        print("  sin ninguna información teórica ni contextual.")
        return
    
    # Si se proporcionan argumentos, ejecutar análisis completo
    pop_path = sys.argv[1]
    
    print(f"Cargando población: {pop_path}")
    df = load_population(pop_path)
    print(f"  → {len(df)} acordes cargados")
    
    # Clasificar cada acorde
    classifications = []
    for _, row in df.iterrows():
        notes = extract_notes_abs(row)
        if notes is not None:
            classifications.append(classify_chord(notes))
        else:
            classifications.append({"quality": "Unknown", "function": None, "root_name": None, "root_pc": None})
    
    df_class = pd.DataFrame(classifications)
    
    print(f"\n── Distribución de cualidades ──")
    print(df_class["quality"].value_counts().to_string())
    
    print(f"\n── Distribución de funciones (solo diatónicos en Do Mayor) ──")
    func_counts = df_class["function"].value_counts()
    print(func_counts.to_string())
    
    # Si hay embedding disponible como segundo argumento
    if len(sys.argv) >= 3:
        emb_path = sys.argv[2]
        print(f"\nCargando embedding: {emb_path}")
        if emb_path.endswith(".npy"):
            embedding = np.load(emb_path)
        else:
            embedding = pd.read_csv(emb_path).values
        
        print(f"  → shape: {embedding.shape}")
        
        # Silhouette por cualidad
        labels_quality = df_class["quality"].values
        sil_quality = compute_silhouette_for_partition(embedding, labels_quality, "Cualidad")
        
        # Silhouette por función
        labels_func = df_class["function"].fillna("None").values
        sil_func = compute_silhouette_for_partition(embedding, labels_func, "Función")
        
        print(f"\n{'=' * 50}")
        print(f"  Silhouette (Cualidad):  {sil_quality:.4f}" if sil_quality is not None else "  Silhouette (Cualidad):  N/A")
        print(f"  Silhouette (Función):   {sil_func:.4f}" if sil_func is not None else "  Silhouette (Función):   N/A")
        print(f"{'=' * 50}")
    else:
        print("\n💡 Para calcular Silhouette, pasa también el embedding como segundo argumento:")
        print(f"   python {sys.argv[0]} {pop_path} <embedding.npy>")


if __name__ == "__main__":
    main()
