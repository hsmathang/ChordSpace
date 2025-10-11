"""
Runner mínimo del laboratorio sin notebooks.

Objetivo
- Ejecutar un experimento de extremo a extremo (modelo → distancias → reducción → visualización)
  usando un conjunto pequeño de acordes sintéticos (sin DB) y dejar artefactos reproducibles
  en una carpeta de salida.

Uso
  python -m tools.run_lab --model Sethares --metric cosine --reduction MDS --out outputs/demo

Salida
- embeddings.npy, distances.npy, report.txt
- fig_scatter.html, fig_heatmap.html, fig_shepard.html
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from lab import LaboratorioAcordes
from pre_process import Acorde, ModeloSethares, ModeloEuler, ModeloArmonicosCriticos, \
    PonderacionConsonancia, PonderacionImportanciaPerceptual, PonderacionCombinada
from visualization import visualizar_scatter_density, visualizar_heatmap, graficar_shepard


def build_synthetic_chords() -> List[Acorde]:
    """Crea un pequeño set de acordes (sin DB) con intervalos típicos.

    - Triada mayor (4,3), menor (3,4), disminuida (3,3)
    - Algunas tétradas para variedad
    """
    chords: List[Acorde] = []
    specs = [
        ("C_maj", [4, 3]),
        ("A_min", [3, 4]),
        ("B_dim", [3, 3]),
        ("G7", [4, 3, 3]),
        ("Dm7", [3, 4, 3]),
        ("Cmaj7", [4, 3, 4]),
        ("Csus2", [2, 5]),
        ("Csus4", [5, 2]),
    ]
    for name, ivs in specs:
        chords.append(Acorde(name=name, intervals=ivs))
    # Añadir variaciones transpuestas (mismas estructuras)
    for t in [2, 5, 7, 9]:
        for base in [
            ("T_maj", [4, 3]),
            ("T_min", [3, 4]),
            ("T_dim", [3, 3]),
        ]:
            chords.append(Acorde(name=f"{base[0]}_{t}", intervals=base[1]))
    return chords


def pick_model(name: str):
    if name.lower() == "sethares":
        return ModeloSethares(config={})
    if name.lower() == "euler":
        return ModeloEuler(config={})
    if name.lower() in ("armonicoscriticos", "armónicos críticos", "armonicos_criticos"):
        return ModeloArmonicosCriticos(config={})
    return ModeloSethares(config={})


def pick_ponderacion(name: str | None):
    if not name or name == "ninguna":
        return None
    if name == "consonancia":
        return PonderacionConsonancia()
    if name == "importancia":
        return PonderacionImportanciaPerceptual()
    if name == "combinada":
        return PonderacionCombinada(PonderacionConsonancia(), PonderacionImportanciaPerceptual())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Sethares")
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--reduction", default="MDS")
    ap.add_argument("--ponderacion", default="ninguna")
    ap.add_argument("--out", type=Path, default=Path("outputs/demo"))
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    acordes = build_synthetic_chords()
    lab = LaboratorioAcordes(acordes)
    modelo = pick_model(args.model)
    pond = pick_ponderacion(args.ponderacion)

    res = lab.ejecutar_experimento(modelo, pond, metrica=args.metric, reduccion=args.reduction)

    # Persistir artefactos
    np.save(out_dir / "embeddings.npy", res.embeddings)
    np.save(out_dir / "distances.npy", res.matriz_distancias)

    # Guardar visualizaciones a HTML (sin depender de motores extra)
    fig_sc = visualizar_scatter_density(res.embeddings, acordes, res.X_original)
    fig_hm = visualizar_heatmap(res.matriz_distancias, acordes)
    fig_sh = graficar_shepard(res.embeddings, res.matriz_distancias)
    fig_sc.write_html(out_dir / "fig_scatter.html")
    fig_hm.write_html(out_dir / "fig_heatmap.html")
    fig_sh.write_html(out_dir / "fig_shepard.html")

    # Reporte mínimo
    with (out_dir / "report.txt").open("w", encoding="utf-8") as f:
        f.write(f"Modelo: {res.nombre_modelo}\n")
        for k, v in res.metricas.items():
            if v is None:
                continue
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")

    print(f"Artefactos guardados en: {out_dir}")


if __name__ == "__main__":
    main()

