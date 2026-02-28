"""
Experimento reproducible: comparacion de vecinos para sus4 en dos espacios.

Espacio A (N=69):
    Coleccion canonica anclada en C3 (diadas, triadas, tetradas, nonales).

Espacio B (N=156):
    Poblacion combinatoria estructural (octavas 3-4, cardinalidades 2-3),
    filtrada con max_internal_interval <= 12.

En ambos casos se usa:
    - propuesta: perclass_alpha0_75
    - distancia: euclidean en el espacio original (12D)
    - consulta: sus4 = [0,5,7] (intervalos [5,2])
    - criterio de vecindad: misma cardinalidad (triadas)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_utils import build_entries
from services.combinatorial_generator import generate_combinatorial_chords
from services.population_filter import filter_dataframe
from tools.data_access import ChordFilters

OUT_PATH = ROOT / "experiments" / "exp_sustitucion_sus4_espacios_results.txt"
BASE_MIDI = 48  # C3
TOP_K = 8


def midi_to_freq(note: int) -> float:
    return 440.0 * 2 ** ((note - 69) / 12.0)


def parse_interval(raw: object) -> List[int]:
    if isinstance(raw, list):
        return [int(v) for v in raw]
    text = str(raw).strip("{}[]() ")
    if not text:
        return []
    return [int(v) for v in text.split(",") if str(v).strip()]


def get_inversions(name: str, semitones: Sequence[int], base: int = BASE_MIDI):
    out = []
    semitones = list(int(v) for v in semitones)
    for inv in range(len(semitones)):
        if inv == 0:
            inv_name = f"{name}-fund"
            notes = [base + s for s in semitones]
        else:
            inv_name = f"{name}-inv{inv}"
            shifted = sorted(semitones[inv:] + [s + 12 for s in semitones[:inv]])
            notes = [base + (s - shifted[0]) for s in shifted]
        intervals = [notes[i + 1] - notes[i] for i in range(len(notes) - 1)]
        freqs = [midi_to_freq(n) for n in notes]
        out.append((inv_name, intervals, freqs, notes))
    return out


def topk_same_cardinality(
    dist_matrix: np.ndarray,
    cardinalities: Sequence[int],
    query_idx: int,
    *,
    k: int = TOP_K,
    farthest: bool = False,
) -> List[Tuple[int, float]]:
    q_card = int(cardinalities[query_idx])
    candidates = [i for i, c in enumerate(cardinalities) if i != query_idx and int(c) == q_card]
    ranked = sorted(
        ((i, float(dist_matrix[query_idx, i])) for i in candidates),
        key=lambda item: item[1],
        reverse=farthest,
    )
    return ranked[:k]


def build_space_a() -> Tuple[List[Tuple[str, List[int], List[float], List[int]]], Dict[str, int]]:
    chords_raw: List[Tuple[str, List[int], List[float], List[int]]] = []

    for name, semitones in [
        ("Diada-m2", [0, 1]),
        ("Diada-M2", [0, 2]),
        ("Diada-m3", [0, 3]),
        ("Diada-M3", [0, 4]),
        ("Diada-P4", [0, 5]),
        ("Diada-TT", [0, 6]),
        ("Diada-P5", [0, 7]),
        ("Diada-m6", [0, 8]),
        ("Diada-M6", [0, 9]),
        ("Diada-m7", [0, 10]),
        ("Diada-M7", [0, 11]),
        ("Diada-P8", [0, 12]),
    ]:
        notes = [BASE_MIDI + s for s in semitones]
        intervals = [notes[i + 1] - notes[i] for i in range(len(notes) - 1)]
        freqs = [midi_to_freq(n) for n in notes]
        chords_raw.append((name, intervals, freqs, notes))

    for name, semitones in [
        ("Maj", [0, 4, 7]),
        ("Min", [0, 3, 7]),
        ("Dim", [0, 3, 6]),
        ("Aug", [0, 4, 8]),
        ("sus2", [0, 2, 7]),
        ("sus4", [0, 5, 7]),
    ]:
        chords_raw.extend(get_inversions(name, semitones))

    for name, semitones in [
        ("Maj7", [0, 4, 7, 11]),
        ("Min7", [0, 3, 7, 10]),
        ("Dom7", [0, 4, 7, 10]),
        ("m7b5", [0, 3, 6, 10]),
        ("dim7", [0, 3, 6, 9]),
        ("mM7", [0, 3, 7, 11]),
    ]:
        chords_raw.extend(get_inversions(name, semitones))

    for name, semitones in [
        ("Maj9", [0, 4, 7, 11, 14]),
        ("Dom9", [0, 4, 7, 10, 14]),
        ("Min9", [0, 3, 7, 10, 14]),
    ]:
        chords_raw.extend(get_inversions(name, semitones))

    query = {"sus4-fund": next(i for i, item in enumerate(chords_raw) if item[0] == "sus4-fund")}
    return chords_raw, query


def build_space_b() -> Tuple[List[Tuple[str, List[int], List[float], List[int]]], List[Dict[str, object]], Dict[str, int]]:
    raw_df = generate_combinatorial_chords(
        alphabet=list(range(12)),
        octave_min=3,
        octave_max=4,
        cardinalities=[2, 3],
        structural_mode=True,
    )
    filters = ChordFilters(
        cardinalities=[2, 3],
        max_internal_interval=12,
        interval_mode="exact",
        include_pitch_classes=[],
    )
    df = filter_dataframe(raw_df, filters).reset_index(drop=True)

    chords_raw = []
    meta: List[Dict[str, object]] = []
    query_idx = -1

    for i, row in df.iterrows():
        notes_abs = [int(v) for v in json.loads(row["notes_abs_json"])]
        intervals = parse_interval(row["interval"])
        code = str(row["code"])
        label = f"{code} [{','.join(str(v) for v in intervals)}]"
        freqs = [midi_to_freq(n) for n in notes_abs]
        struct = [int(v) for v in row["__struct_semitones"]]

        chords_raw.append((label, intervals, freqs, notes_abs))
        meta.append({"code": code, "intervals": intervals, "struct": struct, "n": int(row["n"])})

        if struct == [0, 5, 7]:
            query_idx = i

    if query_idx < 0:
        raise RuntimeError("No se encontro la estructura [0,5,7] (sus4) en el espacio B.")

    return chords_raw, meta, {"sus4-struct": query_idx}


def format_ranked(
    ranked: Iterable[Tuple[int, float]],
    names: Sequence[str],
    meta: Sequence[Dict[str, object]] | None = None,
) -> List[str]:
    lines: List[str] = []
    for pos, (idx, dist) in enumerate(ranked, 1):
        if meta is None:
            extra = ""
        else:
            struct = meta[idx].get("struct")
            extra = f" struct={struct}"
        lines.append(f"{pos:>2}. {names[idx]:<16} d={dist:0.3f}{extra}")
    return lines


def main() -> None:
    # Espacio A
    space_a_raw, space_a_query = build_space_a()
    entries_a, xa = build_entries(space_a_raw, proposal="perclass_alpha0_75")
    da = squareform(pdist(xa, metric="euclidean"))
    cards_a = [e.n_notes for e in entries_a]
    names_a = [e.identity_name for e in entries_a]
    q_a = space_a_query["sus4-fund"]

    near_a = topk_same_cardinality(da, cards_a, q_a, k=TOP_K, farthest=False)
    far_a = topk_same_cardinality(da, cards_a, q_a, k=TOP_K, farthest=True)

    # Espacio B
    space_b_raw, meta_b, space_b_query = build_space_b()
    entries_b, xb = build_entries(space_b_raw, proposal="perclass_alpha0_75")
    db = squareform(pdist(xb, metric="euclidean"))
    cards_b = [e.n_notes for e in entries_b]
    names_b = [e.identity_name for e in entries_b]
    q_b = space_b_query["sus4-struct"]

    near_b = topk_same_cardinality(db, cards_b, q_b, k=TOP_K, farthest=False)
    far_b = topk_same_cardinality(db, cards_b, q_b, k=TOP_K, farthest=True)

    out_lines: List[str] = []
    out_lines.append("EXP_SUS4_ESPACIOS\n")
    out_lines.append("=================\n\n")
    out_lines.append("Parametros comunes:\n")
    out_lines.append("- descriptor: perclass_alpha0_75 (12D)\n")
    out_lines.append("- distancia: euclidean en espacio original\n")
    out_lines.append("- criterio de vecinos: misma cardinalidad (triadas)\n")
    out_lines.append("- consulta: sus4 = [0,5,7] / intervalos [5,2]\n\n")

    out_lines.append(f"Espacio A (N={len(entries_a)})\n")
    out_lines.append("- catalogo canonico anclado en C3\n")
    out_lines.append("- consulta: sus4-fund\n")
    out_lines.append("Top 8 cercanos:\n")
    out_lines.extend(line + "\n" for line in format_ranked(near_a, names_a))
    out_lines.append("Top 8 lejanos:\n")
    out_lines.extend(line + "\n" for line in format_ranked(far_a, names_a))
    out_lines.append("\n")

    out_lines.append(f"Espacio B (N={len(entries_b)})\n")
    out_lines.append("- generacion combinatoria estructural, octavas=3-4, cardinalidades=[2,3], max_internal_interval<=12\n")
    out_lines.append(f"- consulta: {names_b[q_b]} struct={meta_b[q_b]['struct']}\n")
    out_lines.append("Top 8 cercanos:\n")
    out_lines.extend(line + "\n" for line in format_ranked(near_b, names_b, meta_b))
    out_lines.append("Top 8 lejanos:\n")
    out_lines.extend(line + "\n" for line in format_ranked(far_b, names_b, meta_b))

    OUT_PATH.write_text("".join(out_lines), encoding="utf-8")
    print(f"[OK] Resultados guardados en: {OUT_PATH}")


if __name__ == "__main__":
    main()
