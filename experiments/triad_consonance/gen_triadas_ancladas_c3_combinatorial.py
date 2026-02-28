"""
gen_triadas_ancladas_c3_combinatorial.py
----------------------------------------
Genera triadas por combinatoria con raiz fija (ancladas) en C3 (MIDI 48).

Definicion usada:
  acorde = [C3, C3+i, C3+j] con 1 <= i < j <= max_semitones

Conteo teorico:
  n = cantidad de intervalos candidatos (1..max_semitones)
  total_triadas = C(n,2)

Ejemplos:
  - max_semitones=12  -> n=12 -> C(12,2)=66 (incluye C4 como tercera nota posible)
  - max_semitones=11  -> n=11 -> C(11,2)=55 (sin octava)
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path


def midi_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def build_records(base_midi: int, max_semitones: int) -> list[dict]:
    interval_candidates = list(range(1, max_semitones + 1))
    pairs = list(combinations(interval_candidates, 2))

    rows: list[dict] = []
    for idx, (i, j) in enumerate(pairs, start=1):
        notes_abs = [base_midi, base_midi + i, base_midi + j]
        intervals = [notes_abs[1] - notes_abs[0], notes_abs[2] - notes_abs[1]]
        row = {
            "id": idx,
            "code": f"triad_{i}_{j}",
            "interval": intervals,
            "frequencies": [midi_to_freq(m) for m in notes_abs],
            "notes_abs_json": notes_abs,
            "__root_midi": base_midi,
            "__anchor_note": "C3",
            "__generator": "combinatorial_fixed_root",
            "__semitones_from_root": [0, i, j],
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera triadas ancladas por combinatoria y las exporta a JSONL."
    )
    parser.add_argument(
        "--base-midi",
        type=int,
        default=48,
        help="Nota raiz en MIDI (default: 48 = C3).",
    )
    parser.add_argument(
        "--max-semitones",
        type=int,
        default=12,
        help="Maximo desplazamiento en semitonos sobre la raiz (default: 12).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "poblacion_triadas_ancladas_c3_combinatorial.jsonl",
        help="Ruta de salida JSONL.",
    )
    args = parser.parse_args()

    if args.max_semitones < 2:
        raise ValueError("max_semitones debe ser >= 2 para formar triadas.")

    rows = build_records(args.base_midi, args.max_semitones)
    expected = comb(args.max_semitones, 2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] Archivo generado: {args.out}")
    print(f"[INFO] max_semitones={args.max_semitones} -> C({args.max_semitones},2)={expected}")
    print(f"[INFO] Filas exportadas: {len(rows)}")


if __name__ == "__main__":
    main()

