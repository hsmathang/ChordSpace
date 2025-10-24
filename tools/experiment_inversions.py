"""
Experiment runner: poblaciones A/B/C y modo conjunto (joint) para MDS/UMAP.

Flujo basico:
  1) Cargar acordes desde DB usando una constante SQL definida en config.py.
  2) Construir poblacion segun tipo:
        A: sin inversiones
        B: inversiones ancladas a 0 e intersectadas con DB
        C: inversiones sinteticas (sin interseccion)
  3) Ejecutar ModeloSethares y reducir con metrica euclidiana (MDS o UMAP).
  4) Guardar artefactos (CSV, NPY, HTML) en carpeta de salida.

Modo conjunto (joint): especificar varias poblaciones via --pops para embebecer la union.
  Formato: --pops A:QUERY_NAME --pops B:QUERY_NAME [--pops C:QUERY_NAME]
  Se guarda labels.csv para mapear cada punto a su poblacion.

Uso rapido:
  - Unica poblacion (tipo B por defecto):
      python -m tools.experiment_inversions --type B --query QUERY_CHORDS_WITH_NAME --out outputs/demo
  - Conjunto (joint) de dos poblaciones (A y B):
      python -m tools.experiment_inversions --pops A:QUERY_CHORDS_WITH_NAME --pops B:QUERY_CHORDS_WITH_NAME --out outputs/joint_AB
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from config import config_db, QUERY_CHORDS_WITH_NAME
try:
    from chordcodex.model import QueryExecutor  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback to local helper
    from synth_tools import QueryExecutor  # type: ignore
from synth_tools import make_inversions_df, DB_TAG, calculate_row
from pre_process import (
    ChordAdapter,
    ModeloSetharesVec,
    ModeloEuler,
    ModeloArmonicosCriticos,
    PonderacionConsonancia,
    PonderacionImportanciaPerceptual,
    PonderacionCombinada,
)
from lab import LaboratorioAcordes
from visualization import (
    visualizar_scatter_density,
    visualizar_heatmap,
    graficar_shepard,
)
from tools.query_registry import resolve_query_sql


def load_chords_from_db(query_const: str = "QUERY_CHORDS_WITH_NAME") -> pd.DataFrame:
    qe = QueryExecutor(**config_db)
    sql = resolve_query_sql(query_const)
    return qe.as_pandas(sql)


def acordes_from_df(df: pd.DataFrame) -> List[ChordAdapter]:
    acordes: List = []
    for _, row in df.iterrows():
        acordes.append(ChordAdapter.from_csv_row(row))
    return acordes


def _extract_notes_abs_from_row(row: pd.Series) -> List[int]:
    # Preferir notes_abs_json si existe
    raw = row.get("notes_abs_json")
    if isinstance(raw, (str, bytes)) and raw:
        try:
            import json
            return [int(x) for x in json.loads(raw)]
        except Exception:
            pass
    # Caso alterno: reconstruir por interval
    intervals = row.get("interval")
    if isinstance(intervals, str):
        # Posgres puede venir como '{4,3}' o '[4,3]'
        intervals = intervals.strip("{}[]")
        if intervals:
            intervals = [int(x.strip()) for x in intervals.split(',') if x.strip()]
        else:
            intervals = []
    elif isinstance(intervals, list):
        intervals = [int(x) for x in intervals]
    else:
        intervals = []
    notes = [0]
    for step in intervals:
        notes.append(notes[-1] + int(step))
    return notes


def _rotate_plus12(notes_abs: List[int], k: int) -> List[int]:
    if k <= 0:
        return list(notes_abs)
    head = [n + 12 for n in notes_abs[:k]]
    tail = list(notes_abs[k:])
    return tail + head


def _transport_to_zero(notes_abs: List[int]) -> List[int]:
    mn = min(notes_abs) if notes_abs else 0
    return [n - mn for n in notes_abs]


def build_inversions_anchored_db(df_src: pd.DataFrame, include_original: bool = True) -> pd.DataFrame:
    """Genera inversiones ancladas a 0 y selecciona solo las que existen en DB.

    Flujo:
      - Para cada fila de df_src, extraer alturas absolutas y generar rotaciones k=1..n-1.
      - Rotar +12, transportar a 0 (restar mÃ­nimo) y descartar fuera de [0..24].
      - Calcular fila DB-like vÃ­a calculate_row para obtener abs_mask_int.
      - Hacer intersecciÃ³n con DB: traer filas cuyo abs_mask_int estÃ© en el conjunto.
      - Si include_original: concatenar originales.
    """
    masks = []
    n_gen = 0
    n_oob = 0
    show_first_oob = True
    for _, row in df_src.iterrows():
        notes = _extract_notes_abs_from_row(row)
        for k in range(1, len(notes)):
            rot = _rotate_plus12(notes, k)
            anch = _transport_to_zero(rot)
            if not anch or min(anch) < 0 or max(anch) > 24:
                n_oob += 1
                if show_first_oob:
                    print(f"[debug] inversión descartada k={k} rot={rot} transportada={anch}")
                    show_first_oob = False
                continue
            try:
                # calculate_row retorna tuple con abs_mask_int en la pos conocida
                res = calculate_row(tuple(sorted(int(n) for n in anch)))
                # Indices segÃºn fallback: (..., abs_mask_int, abs_mask_hex, notes_abs_json)
                abs_mask_int = int(res[-3])
                masks.append(abs_mask_int)
                n_gen += 1
            except Exception:
                continue

    masks = sorted(set(masks))
    qe = QueryExecutor(**config_db)
    if masks:
        # Usar ANY para traer filas coincidentes por abs_mask_int
        sql = "SELECT * FROM chords WHERE abs_mask_int = ANY(%s)"
        df_inv_db = qe.as_pandas(sql, (masks,))
    else:
        df_inv_db = pd.DataFrame()

    if include_original:
        cols = df_inv_db.columns.tolist() if not df_inv_db.empty else df_src.columns.tolist()
        base = df_src[cols] if all(c in df_src.columns for c in cols) else df_src
        out = pd.concat([base, df_inv_db], ignore_index=True)
    else:
        out = df_inv_db

    # Reporte breve
    print(f"Inversiones generadas(anchored): {n_gen}, fuera de rango descartadas: {n_oob}, en_DB: {len(df_inv_db)}")
    return out


def _parse_pop_spec(spec: str) -> tuple[str, str]:
    """Parsea una especificacion 'A:QUERY' o solo 'A'. Retorna (tipo, query_const)."""
    parts = spec.split(":", 1)
    ptype = parts[0].strip().upper()
    if ptype not in {"A", "B", "C"}:
        raise ValueError(f"Tipo de poblacion invalido: {ptype}")
    qname = parts[1].strip() if len(parts) == 2 and parts[1].strip() else "QUERY_CHORDS_WITH_NAME"
    return ptype, qname


def _build_population(ptype: str, query_const: str) -> pd.DataFrame:
    df_src = load_chords_from_db(query_const)
    if ptype == "A":
        return df_src.copy()
    if ptype == "B":
        return build_inversions_anchored_db(df_src, include_original=True)
    return make_inversions_df(df_src, tag=DB_TAG, include_original=True)


def _collect_pops_specs(args: argparse.Namespace) -> list[str]:
    pops_specs: list[str] = []
    if getattr(args, "pops", None):
        pops_specs.extend(args.pops)
    if getattr(args, "pops_csv", None):
        pops_specs.extend([s.strip() for s in str(args.pops_csv).split(",") if s.strip()])
    if getattr(args, "pops_file", None):
        pops_file: Path = args.pops_file
        if pops_file.exists():
            try:
                content = pops_file.read_text(encoding="utf-8")
            except Exception:
                content = pops_file.read_text(encoding="latin-1")
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pops_specs.append(line)
    seen = set()
    ordered: list[str] = []
    for spec in pops_specs:
        spec_clean = spec.strip()
        if not spec_clean or spec_clean in seen:
            continue
        ordered.append(spec_clean)
        seen.add(spec_clean)
    return ordered


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/inversions_experiment"))
    ap.add_argument("--type", choices=["A", "B", "C"], default="B",
                    help="A: sin inversiones; B: inversiones ancladas a 0 e intersectadas con DB; C: inversiones sintéticas (sin intersección)")
    ap.add_argument("--query", default="QUERY_CHORDS_WITH_NAME",
                    help="Nombre de constante en config.py o consulta guardada")
    ap.add_argument("--reduction", choices=["MDS", "UMAP"], default="MDS")
    ap.add_argument(
        "--metric",
        default="euclidean",
        choices=["euclidean", "cosine", "cityblock", "chebyshev", "custom"],
        help="Métrica de distancia para la matriz (por defecto: euclidean).",
    )
    ap.add_argument(
        "--model",
        default="Sethares",
        choices=["Sethares", "SetharesVec", "Euler", "ArmonicosCriticos"],
        help="Modelo de rugosidad a utilizar.",
    )
    ap.add_argument(
        "--ponderation",
        default="ninguna",
        choices=["ninguna", "consonancia", "importancia", "combinada"],
        help="Ponderación a aplicar sobre el vector de rugosidad.",
    )
    ap.add_argument("--pops", action="append", default=None,
                    help="Modo conjunto: varias poblaciones 'A:QUERY'. Repetible.")
    ap.add_argument("--pops-csv", dest="pops_csv", default=None,
                    help="Lista separada por comas de pops, ej: A:QUERY1,B:QUERY2")
    ap.add_argument("--pops-file", dest="pops_file", type=Path, default=None,
                    help="Archivo de texto con especificaciones por línea (comentarios con #)")
    return ap


def run_experiment_with_args(
    args: argparse.Namespace,
    df_override: Optional[pd.DataFrame] = None,
    descriptor: Optional[str] = None,
) -> dict:
    out_dir = Path(getattr(args, "out", Path("outputs/inversions_experiment"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    override_used = df_override is not None

    pops_specs = [] if override_used else _collect_pops_specs(args)

    if pops_specs:
        combined_frames: list[pd.DataFrame] = []
        for spec in pops_specs:
            ptype, qname = _parse_pop_spec(spec)
            label = f"{ptype}:{qname}"
            print(f"Construyendo poblacion conjunta: {label}")
            df_p = _build_population(ptype, qname).copy()
            df_p["__population__"] = label
            combined_frames.append(df_p)

        if combined_frames:
            df_pop_raw = pd.concat(combined_frames, ignore_index=True)
        else:
            df_pop_raw = pd.DataFrame()

        if not df_pop_raw.empty:
            if "abs_mask_int" in df_pop_raw.columns:
                df_pop_raw["__dedupe_key__"] = df_pop_raw["abs_mask_int"]
            elif "notes_abs_json" in df_pop_raw.columns:
                df_pop_raw["__dedupe_key__"] = df_pop_raw["notes_abs_json"]
            else:
                df_pop_raw["__dedupe_key__"] = (
                    df_pop_raw.get("code", "").astype(str) + "|" +
                    df_pop_raw.get("interval", "").astype(str)
                )

            before = len(df_pop_raw)
            df_pop = df_pop_raw.drop_duplicates(subset="__dedupe_key__", keep="first").copy()
            removed = before - len(df_pop)
            if removed > 0:
                print(f"Advertencia: se eliminaron {removed} duplicados al combinar poblaciones.")

            lab_df = pd.DataFrame({
                "population": df_pop["__population__"].tolist(),
                "id": df_pop["id"].tolist() if "id" in df_pop.columns else [None] * len(df_pop),
                "code": df_pop["code"].tolist() if "code" in df_pop.columns else [None] * len(df_pop),
            })
            lab_df.to_csv(out_dir / "labels.csv", index=False, encoding="utf-8")
            df_pop.to_csv(out_dir / "union_population.csv", index=False, encoding="utf-8")
            drop_cols = [c for c in ("__population__", "__dedupe_key__") if c in df_pop.columns]
            if drop_cols:
                df_pop = df_pop.drop(columns=drop_cols)
        else:
            df_pop = df_pop_raw
    else:
        if override_used:
            df_pop = df_override.copy()
        else:
            query_name = getattr(args, "query", None) or QUERY_CHORDS_WITH_NAME
            df_src = load_chords_from_db(query_name)
            run_type = getattr(args, "type", None) or "B"
            if run_type == "A":
                df_pop = df_src.copy()
            elif run_type == "B":
                df_pop = build_inversions_anchored_db(df_src, include_original=True)
            else:
                df_pop = make_inversions_df(df_src, tag=DB_TAG, include_original=True)
                df_pop.to_csv(out_dir / "inversions_synthetic.csv", index=False, encoding="utf-8")

    if df_pop is None or df_pop.empty:
        raise ValueError("La población resultante está vacía. Verifica la selección o consulta utilizada.")

    if override_used:
        df_pop = df_pop.reset_index(drop=True)
    elif pops_specs:
        df_pop = df_pop.reset_index(drop=True)
    else:
        df_pop = df_pop.reset_index(drop=True)

    acordes = acordes_from_df(df_pop)
    lab = LaboratorioAcordes(acordes)
    # Seleccionar modelo
    model_key = str(getattr(args, "model", "Sethares")).strip()
    if model_key in {"Sethares", "SetharesVec"}:
        modelo = ModeloSetharesVec(config={})
    elif model_key == "Euler":
        modelo = ModeloEuler(config={})
    else:
        modelo = ModeloArmonicosCriticos(config={})

    # Seleccionar ponderación
    pond_key = str(getattr(args, "ponderation", "ninguna")).strip().lower()
    if pond_key == "consonancia":
        ponderacion = PonderacionConsonancia()
    elif pond_key == "importancia":
        ponderacion = PonderacionImportanciaPerceptual()
    elif pond_key == "combinada":
        ponderacion = PonderacionCombinada(PonderacionConsonancia(), PonderacionImportanciaPerceptual())
    else:
        ponderacion = None

    # Métrica de distancia
    metric = str(getattr(args, "metric", "euclidean")).strip().lower()
    res = lab.ejecutar_experimento(modelo, ponderacion=ponderacion, metrica=metric, reduccion=args.reduction)

    np.save(out_dir / "embeddings.npy", res.embeddings)
    np.save(out_dir / "distances.npy", res.matriz_distancias)

    if pops_specs:
        experiment_descriptor = " | ".join(pops_specs)
        ttl = f"{args.reduction} ({metric}) - joint"
    elif override_used:
        experiment_descriptor = descriptor or "Selección manual"
        ttl = f"{args.reduction} ({metric}) - {experiment_descriptor}"
    else:
        experiment_descriptor = f"Tipo {getattr(args, 'type', 'B')} | Query {getattr(args, 'query', QUERY_CHORDS_WITH_NAME)}"
        ttl = f"{args.reduction} ({metric}) - tipo {getattr(args, 'type', 'B')} - {getattr(args, 'query', QUERY_CHORDS_WITH_NAME)}"
    fig_sc = visualizar_scatter_density(res.embeddings, acordes, res.X_original, title=ttl)
    fig_hm = visualizar_heatmap(res.matriz_distancias, acordes, title="Matriz de Distancias")
    fig_sh = graficar_shepard(res.embeddings, res.matriz_distancias, title="Grafico de Shepard")
    fig_sc.write_html(out_dir / "scatter.html")
    fig_hm.write_html(out_dir / "heatmap.html")
    fig_sh.write_html(out_dir / "shepard.html")

    with (out_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Experimento: inversiones\n")
        f.write(f"Población: {experiment_descriptor}\n")
        f.write(f"Modelo: {res.nombre_modelo}\n")
        f.write(f"Ponderación: {pond_key}\n")
        f.write(f"Métrica: {metric}\n")
        f.write(f"Reducción: {args.reduction}\n")
        f.write(f"Total acordes: {len(acordes)}\n")
        for k, v in res.metricas.items():
            if v is None:
                continue
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")
        f.write(f"Embeddings: {res.embeddings.shape}\n")
        f.write(f"Distancias: {res.matriz_distancias.shape}\n")

    print("Listo. Artefactos en:", out_dir)
    return {
        "output_dir": out_dir,
        "pops": pops_specs,
        "experiment_descriptor": experiment_descriptor,
        "resultado": res,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_experiment_with_args(args)


if __name__ == "__main__":
    main()
