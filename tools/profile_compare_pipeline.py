"""
Perfilador reproducible para el pipeline de comparación de propuestas.

Este módulo permite medir, de forma controlada, el coste temporal de cada
etapa clave del proceso:
    1. Descarga de la población desde la base de datos.
    2. Construcción de histogramas Sethares y tensores auxiliares.
    3. Preprocesamiento asociado a una propuesta (normalización, suavizado, etc.).
    4. Construcción de la matriz de distancias/discordancias (pdist).
    5. Reducción dimensional (MDS/UMAP/TSNE/ISOMAP).

El objetivo académico es identificar cuellos de botella al escalar el tamaño
de la población. Para facilitar el análisis se imprime un resumen ordenado
por etapas y, opcionalmente, se genera un informe de cProfile con los 30
frames más costosos.

Uso típico::

    python -m tools.profile_compare_pipeline \\
        --population-query QUERY_CHORDS_4_NOTES_SAMPLE_75 \\
        --proposal perclass_alpha0_75 \\
        --metric euclidean \\
        --reduction MDS \\
        --seeds 42 \\
        --profile-stats stats_mds.prof

El script reutiliza internamente las mismas funciones que `compare_proposals`
para garantizar que las mediciones sean representativas del pipeline real.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from tools.compare_proposals import (
    AVAILABLE_REDUCTIONS,
    PREPROCESSORS,
    aggregate_seed_results,
    compute_embeddings,
    evaluate_mixture_error,
    evaluate_nn_hits,
    load_chords,
    metric_distance,
    parse_seed_list,
    stack_hist,
    summarise_embedding_metrics,
)


@dataclass
class StageTiming:
    """Contenedor simple para registrar la duración de una etapa."""

    label: str
    seconds: float


def _time_stage(label: str, func, *args, **kwargs):
    """Ejecuta `func` midiendo tiempo de pared y retorna (resultado, StageTiming)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    t1 = time.perf_counter()
    return result, StageTiming(label, t1 - t0)


def _format_timings(stages: Sequence[StageTiming]) -> str:
    total = sum(stage.seconds for stage in stages)
    lines = ["\n=== Perfil de etapas (segundos) ==="]
    for stage in stages:
        pct = (stage.seconds / total * 100) if total > 0 else 0.0
        lines.append(f"{stage.label:<28} {stage.seconds:8.3f}s ({pct:5.1f} %)")
    lines.append(f"{'TOTAL':<28} {total:8.3f}s (100.0 %)")
    return "\n".join(lines)


def _profile_pipeline(args: argparse.Namespace) -> None:
    """Ejecuta el pipeline de interés registrando tiempos y métricas."""
    timings: List[StageTiming] = []

    # 1) Descarga de población.
    (entries, entries_timing) = _time_stage(
        "Carga desde DB",
        load_chords,
        args.dyads_query,
        args.triads_query,
        args.sevenths_query,
    )
    timings.append(entries_timing)
    print(f"[perfil] población cargada: {len(entries)} acordes.")

    # 2) Construcción de histogramas/tensores.
    (stack_tuple, stack_timing) = _time_stage("Stack hist/pairs", stack_hist, entries)
    timings.append(stack_timing)
    hist, totals, counts, pairs, _notes = stack_tuple

    seeds = parse_seed_list(args.seeds) or [args.seed]

    # 3) Preprocesamiento de la propuesta.
    preproc_id = args.proposal
    if preproc_id not in PREPROCESSORS:
        raise ValueError(f"Propuesta desconocida: {preproc_id}")
    preproc_label, preproc_func, preproc_kwargs = PREPROCESSORS[preproc_id]
    (preproc_res, preproc_timing) = _time_stage(
        f"Preproc ({preproc_label})", preproc_func, hist, counts=counts, pairs=pairs, **preproc_kwargs
    )
    timings.append(preproc_timing)
    feature_matrix, simplex_matrix = preproc_res

    # 4) Construcción de la matriz de distancias.
    (condensed, pdist_timing) = _time_stage(
        f"pdist ({args.metric})",
        metric_distance,
        args.metric,
        feature_matrix,
        simplex_matrix,
    )
    timings.append(pdist_timing)

    dist_matrix = None
    figures_payload: Dict[str, object] = {}
    per_seed_rows: List[Dict[str, Optional[float]]] = []

    # 5) Reducciones solicitadas (se mide individualmente cada una).
    for reduction in args.reductions:
        reduction = reduction.upper()
        if reduction not in AVAILABLE_REDUCTIONS:
            raise ValueError(f"Reducción no soportada: {reduction}")

        label = f"Reducción {reduction}"
        (embedding, red_timing) = _time_stage(
            label,
            compute_embeddings,
            condensed,
            reduction,
            seeds[0],
            base_matrix=feature_matrix,
            n_jobs=args.n_jobs,
            deterministic=args.execution_mode != "parallel",
            mds_n_init=args.mds_n_init,
        )
        timings.append(red_timing)

        if dist_matrix is None:
            dist_matrix = condensed if reduction == "MDS" else None

        dist_square = None
        if reduction in {"MDS", "TSNE", "UMAP"}:
            dist_square = metric_distance("euclidean", embedding, embedding)
            dist_square = None  # evitar basura; mantenemos variable para futuras extensiones

        nn_top1, nn_top2 = evaluate_nn_hits(condensed, entries, simplex_matrix)
        mix_mean, mix_max = evaluate_mixture_error(simplex_matrix, entries)
        metrics_summary = summarise_embedding_metrics(feature_matrix, embedding, condensed)
        per_seed_rows.append(
            {
                "reduction": reduction,
                "seed": seeds[0],
                "nn_hit_top1": nn_top1,
                "nn_hit_top2": nn_top2,
                "mixture_l1_mean": mix_mean,
                "mixture_l1_max": mix_max,
                **metrics_summary,
            }
        )
        figures_payload[reduction] = embedding

    print(_format_timings(timings))
    if per_seed_rows:
        summary = aggregate_seed_results(per_seed_rows, seeds)
        print("\n=== Métricas agregadas (una semilla) ===")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key:<22}: {value: .6f}")
            else:
                print(f"{key:<22}: {value}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Perfilador del pipeline de compare_proposals.")
    ap.add_argument(
        "--dyads-query",
        default="QUERY_DYADS_REFERENCE",
        help="Consulta o SQL para díadas (por defecto: catálogo de referencia).",
    )
    ap.add_argument(
        "--triads-query",
        default="QUERY_CHORDS_3_NOTES_ALL",
        help="Consulta o SQL para tríadas (por defecto: conjunto completo).",
    )
    ap.add_argument(
        "--sevenths-query",
        default="QUERY_CHORDS_4_NOTES_SAMPLE_25",
        help="Consulta o SQL para séptimas (por defecto: muestra 25%).",
    )
    ap.add_argument(
        "--proposal",
        default="perclass_alpha0_75",
        help="Identificador de propuesta (debe existir en PREPROCESSORS).",
    )
    ap.add_argument(
        "--metric",
        default="euclidean",
        help="Métrica para la matriz de distancias (euclidean, cosine, etc.).",
    )
    ap.add_argument(
        "--reductions",
        default="MDS",
        help="Lista separada por comas de reducciones a perfilar (MDS,UMAP,TSNE,ISOMAP).",
    )
    ap.add_argument(
        "--execution-mode",
        choices=["deterministic", "parallel"],
        default="deterministic",
        help="Controla parámetros de reproducibilidad (seeds y n_jobs).",
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Número de núcleos usados dentro de la reducción (1 recomendado para MDS).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla única para la reducción (si --seeds no se especifica).",
    )
    ap.add_argument(
        "--seeds",
        default="",
        help="Lista separada por comas de semillas (solo la primera se usa en este perfil).",
    )
    ap.add_argument(
        "--mds-n-init",
        type=int,
        default=4,
        help="Número de inicializaciones de MDS (consistente con compare_proposals).",
    )
    ap.add_argument(
        "--profile-stats",
        default=None,
        help="Ruta para guardar un volcado cProfile; si no se da, no se genera.",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    args.reductions = [r.strip().upper() for r in args.reductions.split(",") if r.strip()]

    profiler: Optional[cProfile.Profile]
    if args.profile_stats:
        profiler = cProfile.Profile()
        profiler.enable()
        _profile_pipeline(args)
        profiler.disable()
        profiler.dump_stats(args.profile_stats)
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        stats.print_stats(30)
        print("\n=== Resumen cProfile (top 30 por tiempo acumulado) ===")
        print(stream.getvalue())
    else:
        _profile_pipeline(args)


if __name__ == "__main__":
    main()

