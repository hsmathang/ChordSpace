"""
exp_orden_triadas_rtotal_anclado.py
===================================
Experimento reproducible para evaluar el orden de rugosidad total (R_total)
en cinco cualidades triadicas:
  - Mayor, Menor, Sus4, Disminuida, Aumentada

Dos escenarios (ambos anclados a C3 = MIDI 48):
  1) Base: voicing fundamental.
  2) Robusto: promedio de R_total sobre fund, inv1 e inv2 (ancladas).

Salidas:
  - outputs/orden_triadas_rtotal_anclado/triadas_rtotal_base.csv
  - outputs/orden_triadas_rtotal_anclado/triadas_rtotal_robusto.csv
  - outputs/orden_triadas_rtotal_anclado/triadas_rtotal_resumen.csv
  - outputs/orden_triadas_rtotal_anclado/tabla_latex_orden_triadas_rtotal.tex
  - outputs/orden_triadas_rtotal_anclado/reporte_orden_triadas_rtotal.txt
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
import sys
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pre_process import Acorde, ModeloSetharesVec


BASE_MIDI = 48  # C3
OUT_DIR = Path(__file__).parent / "outputs" / "orden_triadas_rtotal_anclado"

QUALITIES: List[Tuple[str, List[int]]] = [
    ("Mayor", [0, 4, 7]),
    ("Menor", [0, 3, 7]),
    ("Sus4", [0, 5, 7]),
    ("Disminuida", [0, 3, 6]),
    ("Aumentada", [0, 4, 8]),
]


def midi_to_freq(midi_note: int) -> float:
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))


def r_total_for_notes(model: ModeloSetharesVec, notes_abs: List[int]) -> float:
    notes = sorted(notes_abs)
    intervals = [notes[i + 1] - notes[i] for i in range(len(notes) - 1)]
    freqs = [midi_to_freq(m) for m in notes]
    chord = Acorde(name="tmp", intervals=intervals, frequencies=freqs)
    _, total = model.calcular(chord)
    return float(total)


def build_anchored_voicings(base_midi: int, semitones_from_root: List[int]) -> Dict[str, List[int]]:
    # Fundamental
    fund = [base_midi + s for s in semitones_from_root]

    # First inversion anchored again to base_midi
    inv1_tmp = [semitones_from_root[1], semitones_from_root[2], semitones_from_root[0] + 12]
    shift1 = inv1_tmp[0]
    inv1 = [base_midi + (n - shift1) for n in inv1_tmp]

    # Second inversion anchored again to base_midi
    inv2_tmp = [semitones_from_root[2], semitones_from_root[0] + 12, semitones_from_root[1] + 12]
    shift2 = inv2_tmp[0]
    inv2 = [base_midi + (n - shift2) for n in inv2_tmp]

    return {"fund": fund, "inv1": inv1, "inv2": inv2}


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_latex_table(sorted_rows: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        "\\caption[Orden de rugosidad triadica (anclado)]"
        "{Rugosidad total $R_{\\text{total}}$ en cinco cualidades triadicas, "
        "comparando dos escenarios anclados a C3: (i) voicing fundamental "
        "(base) y (ii) promedio sobre fundamental, primera y segunda inversion "
        "(robusto). La tabla esta ordenada de menor a mayor $R_{\\text{total}}$ "
        "en el escenario robusto.}"
    )
    lines.append("\\label{tab:orden_triadas_rtotal_anclado}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l c c}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Cualidad triadica} & "
        "\\textbf{$R_{\\text{total}}$ base} & "
        "\\textbf{$R_{\\text{total}}$ robusto} \\\\"
    )
    lines.append("\\midrule")
    for row in sorted_rows:
        lines.append(
            f"{row['quality']} & "
            f"{row['r_total_base']:.4f} & "
            f"{row['r_total_robusto']:.4f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = ModeloSetharesVec(config={})

    base_rows: List[Dict[str, object]] = []
    robust_rows: List[Dict[str, object]] = []
    summary: Dict[str, Dict[str, object]] = {}

    for quality, semitones in QUALITIES:
        voicings = build_anchored_voicings(BASE_MIDI, semitones)

        r_base = r_total_for_notes(model, voicings["fund"])
        r_fund = r_total_for_notes(model, voicings["fund"])
        r_inv1 = r_total_for_notes(model, voicings["inv1"])
        r_inv2 = r_total_for_notes(model, voicings["inv2"])
        r_rob = float(mean([r_fund, r_inv1, r_inv2]))

        base_rows.append({"quality": quality, "r_total": r_base})
        robust_rows.append(
            {
                "quality": quality,
                "r_total_fund": r_fund,
                "r_total_inv1": r_inv1,
                "r_total_inv2": r_inv2,
                "r_total_mean": r_rob,
            }
        )
        summary[quality] = {"quality": quality, "r_total_base": r_base, "r_total_robusto": r_rob}

    base_sorted = sorted(base_rows, key=lambda x: float(x["r_total"]))
    robust_sorted = sorted(robust_rows, key=lambda x: float(x["r_total_mean"]))

    rank_base = {row["quality"]: i + 1 for i, row in enumerate(base_sorted)}
    rank_rob = {row["quality"]: i + 1 for i, row in enumerate(robust_sorted)}

    summary_rows: List[Dict[str, object]] = []
    for quality in summary:
        item = summary[quality]
        item["rank_base"] = rank_base[quality]
        item["rank_robusto"] = rank_rob[quality]
        summary_rows.append(item)

    summary_sorted = sorted(summary_rows, key=lambda x: int(x["rank_robusto"]))

    write_csv(
        OUT_DIR / "triadas_rtotal_base.csv",
        ["quality", "r_total"],
        base_sorted,
    )
    write_csv(
        OUT_DIR / "triadas_rtotal_robusto.csv",
        ["quality", "r_total_fund", "r_total_inv1", "r_total_inv2", "r_total_mean"],
        robust_sorted,
    )
    write_csv(
        OUT_DIR / "triadas_rtotal_resumen.csv",
        ["quality", "r_total_base", "r_total_robusto", "rank_base", "rank_robusto"],
        summary_sorted,
    )

    latex_table = build_latex_table(summary_sorted)
    (OUT_DIR / "tabla_latex_orden_triadas_rtotal.tex").write_text(latex_table, encoding="utf-8")

    order_base = " < ".join([str(row["quality"]) for row in base_sorted])
    order_rob = " < ".join([str(row["quality"]) for row in robust_sorted])
    report_lines = [
        "Orden de rugosidad (menor a mayor) - escenario base:",
        order_base,
        "",
        "Orden de rugosidad (menor a mayor) - escenario robusto anclado:",
        order_rob,
        "",
        "NOTA: menor R_total implica menor friccion interparcial.",
    ]
    (OUT_DIR / "reporte_orden_triadas_rtotal.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("[OK] Experimento completado.")
    print(f"[OK] Salidas en: {OUT_DIR}")
    print(f"[OK] Orden base:    {order_base}")
    print(f"[OK] Orden robusto: {order_rob}")


if __name__ == "__main__":
    main()
