"""Comparación visual de reducciones dimensionales para una propuesta fija.

Genera un reporte HTML donde cada reducción seleccionada se muestra en
paralelo (tarjetas) para las métricas elegidas.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import squareform

from tools import compare_proposals as cp


AVAILABLE_REDUCTIONS = cp.AVAILABLE_REDUCTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara un único método de normalización a través de varias reducciones dimensionales."
    )
    parser.add_argument("--proposal", required=True, help="Identificador de la propuesta (ej. perclass_alpha1).")
    parser.add_argument(
        "--metrics",
        default="euclidean",
        help="Lista separada por comas de métricas (euclidean, cosine, js, hellinger, etc.).",
    )
    parser.add_argument(
        "--reductions",
        default="MDS,UMAP",
        help="Lista separada por comas de métodos de reducción (MDS,UMAP,…).",
    )
    parser.add_argument(
        "--dyads-query",
        default="QUERY_DYADS_REFERENCE",
        help="Consulta o SQL para díadas (se usa como población base).",
    )
    parser.add_argument(
        "--triads-query",
        default="",
        help="Consulta opcional para tríadas (se concatena si se proporciona).",
    )
    parser.add_argument(
        "--sevenths-query",
        default="",
        help="Consulta opcional para séptimas (se concatena si se proporciona).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla base para reducciones deterministas (usada cuando --seeds está vacío).",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Lista separada por comas de semillas para promediar resultados.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Carpeta de salida (por defecto outputs/compare_reductions/<timestamp>).",
    )
    return parser.parse_args()


def _normalise_list(csv: str) -> List[str]:
    return [item.strip() for item in csv.split(',') if item.strip()]


def _target_preproc_ids(proposal: str, scenarios: Sequence[Dict[str, object]]) -> List[str]:
    prefix = proposal.strip().lower()
    ids = {
        s["preproc_id"]
        for s in scenarios
        if isinstance(s.get("name"), str) and s["name"].split("|")[0].strip() == prefix
    }
    if not ids:
        raise SystemExit(f"Propuesta desconocida: '{proposal}'.")
    return sorted(ids)


def _prepare_color_panels(
    scenario_name: str,
    preproc_id: str,
    totals: np.ndarray,
    totals_adj: np.ndarray,
    pairs: np.ndarray,
    existing_counts: np.ndarray,
    base_matrix: np.ndarray,
    adjusted_vectors: np.ndarray,
    entries: List[cp.ChordEntry],
    reduction: str,
    figures: List[Tuple[str, go.Figure]],
    embedding: np.ndarray,
    color_modes: Sequence[Tuple[str, Optional[float]]],
) -> None:
    fig_title = f"{reduction} — {scenario_name}"
    for mode, exponent in color_modes:
        vals, ctitle = _apply_color_mode(
            mode,
            exponent,
            totals,
            totals_adj,
            pairs,
            existing_counts,
        )
        key = mode if exponent is None else f"{mode}_{int(round(exponent * 100)):03d}"
        fig = cp.build_scatter_figure(
            embedding=embedding,
            entries=entries,
            color_values=vals,
            pair_counts=pairs,
            type_counts=existing_counts,
            vectors=base_matrix,
            adjusted_vectors=adjusted_vectors,
            title=fig_title,
            is_proposal=(preproc_id != "identity"),
            color_title=ctitle,
        )
        figures.append((f"{scenario_name}||{reduction}||{key}", fig))


def _apply_color_mode(
    mode: str,
    exponent: Optional[float],
    totals_raw: np.ndarray,
    totals_adjusted: np.ndarray,
    pairs_arr: np.ndarray,
    types_arr: np.ndarray,
) -> Tuple[np.ndarray, str]:
    mode = mode.lower()
    if mode == "pair_exp":
        exp = exponent if exponent is not None else 1.0
        denom = cp._safe_denominator(pairs_arr, subtract=cp.COLOR_PER_PAIR_SUBTRACT)
        denom = np.power(denom, exp)
        if not np.isclose(cp.COLOR_DEN_EXPONENT, 1.0):
            denom = np.power(denom, cp.COLOR_DEN_EXPONENT)
        vals = totals_raw / denom
        title = f"Total/Pares^{cp._format_exp(exp)}"
    elif mode == "types_exp":
        exp = exponent if exponent is not None else 1.0
        denom = cp._safe_denominator(types_arr, subtract=cp.COLOR_PER_EXISTING_SUBTRACT)
        denom = np.power(denom, exp)
        if not np.isclose(cp.COLOR_DEN_EXPONENT, 1.0):
            denom = np.power(denom, cp.COLOR_DEN_EXPONENT)
        vals = totals_adjusted / denom
        title = f"Total ajustado/Tipos^{cp._format_exp(exp)}"
    elif mode == "raw_total":
        vals = totals_raw.copy()
        title = "Total bruto"
    else:
        raise ValueError(f"Modo de color no soportado: {mode}")

    if not np.isclose(cp.COLOR_OUTPUT_EXPONENT, 1.0):
        vals = np.power(np.clip(vals, 0.0, None), cp.COLOR_OUTPUT_EXPONENT)
    return vals, title


def build_reduction_report(
    proposal: str,
    metric_results: Dict[str, List[Dict[str, object]]],
    figures: List[Tuple[str, go.Figure]],
    output_path: Path,
    seeds: Sequence[int],
) -> None:
    figure_map = {key: fig for key, fig in figures}
    sections: List[str] = []
    table_rows: List[Dict[str, object]] = []

    for metric, rows in metric_results.items():
        table_rows.extend(rows)
        metric_info = cp.METRIC_INFO.get(metric, {"title": metric.upper(), "casual": "", "technical": ""})
        body: List[str] = [
            f"<section><h2>{metric_info['title']} ({metric.upper()})</h2>",
            f"<div class='metric-desc'><p>{metric_info['casual']}</p><p><em>Detalle técnico:</em> {metric_info['technical']}</p></div>",
            "<div class='reduction-grid'>",
        ]
        for row in rows:
            body.append(render_reduction_card(row, figure_map))
        body.append("</div></section>")
        sections.append("".join(body))

    if table_rows:
        df = pd.DataFrame(table_rows)
        display_df = df[
            [
                "reduction",
                "metric",
                "stress_mean",
                "stress_std",
                "trustworthiness_mean",
                "trustworthiness_std",
                "mixture_l1_mean_mean",
                "mixture_l1_mean_std",
                "seeds",
            ]
        ].copy()
        display_df["Stress"] = display_df.apply(
            lambda row: cp.format_value_with_std(row["stress_mean"], row["stress_std"]), axis=1
        )
        display_df["Trustworthiness"] = display_df.apply(
            lambda row: cp.format_value_with_std(row["trustworthiness_mean"], row["trustworthiness_std"]), axis=1
        )
        display_df["Mixture L1"] = display_df.apply(
            lambda row: cp.format_value_with_std(row["mixture_l1_mean_mean"], row["mixture_l1_mean_std"]), axis=1
        )
        display_df["Semillas"] = display_df["seeds"].apply(cp.format_seed_list)
        display_df = display_df.rename(
            columns={
                "reduction": "Reducción",
                "metric": "Métrica",
            }
        )
        display_df = display_df[["Reducción", "Métrica", "Stress", "Trustworthiness", "Mixture L1", "Semillas"]]
        table_html = display_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    else:
        table_html = "<p>No se generaron resultados.</p>"

    seeds_text = cp.format_seed_list(seeds)
    prop_info = cp.PROPOSAL_INFO.get(proposal.lower(), {"title": proposal})
    html_content = f"""
<!DOCTYPE html>
<html lang='es'>
<head>
  <meta charset='utf-8'/>
  <title>Comparación de Reducciones Dimensionales</title>
  <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; margin-bottom: 30px; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: center; }}
    h1 {{ margin-bottom: 12px; }}
    h2 {{ margin-top: 28px; }}
    section {{ margin-bottom: 40px; }}
    .metric-desc {{ background: #f0f4ff; padding: 12px 16px; border-radius: 10px; margin-bottom: 16px; }}
    .reduction-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 24px; }}
    .plot-card {{ background: #f8f9fb; padding: 14px; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
    .card-header {{ margin: 0 0 6px 0; font-size: 1.02rem; }}
    .metrics-line {{ font-size: 0.9rem; color: #444; margin: 0 0 8px 0; }}
    .color-controls {{ margin: 8px 0 10px 0; display: flex; gap: 16px; align-items: center; }}
    .color-controls label {{ font-size: 0.9rem; color: #333; }}
    .color-controls input[disabled] {{ opacity: 0.5; cursor: not-allowed; }}
  </style>
</head>
<body>
  <h1>Comparación de Reducciones Dimensionales</h1>
  <p><strong>Propuesta:</strong> {prop_info.get('title', proposal)} ({proposal}) &nbsp;·&nbsp; <strong>Semillas:</strong> {seeds_text}</p>
  <h3>Métricas globales</h3>
  {table_html}
  {''.join(sections)}
  <script>
    (function() {{
      document.querySelectorAll('.plot-card').forEach(card => {{
        const sid = card.dataset.sid || '';
        const controls = card.querySelector('.color-controls');
        const panels = card.querySelectorAll('.subtab-panels .subtab-panel');
        const modeSel = card.querySelector(`#${{sid}}-mode`);
        const expSlider = card.querySelector(`#${{sid}}-exp`);
        const expVal = card.querySelector(`#${{sid}}-exp-val`);
        const defaultMode = controls ? controls.dataset.defaultMode || 'pair_exp' : 'pair_exp';
        const defaultExp = controls ? parseFloat(controls.dataset.defaultExp || '1') : 1;
        if (modeSel && !modeSel.value) modeSel.value = defaultMode;
        if (expSlider && !expSlider.value) expSlider.value = defaultExp.toFixed(2);

        function showPanel() {{
          const mode = modeSel ? modeSel.value : defaultMode;
          let optionDefaultExp = defaultExp;
          if (modeSel) {{
            const opt = modeSel.options[modeSel.selectedIndex];
            if (opt && opt.dataset.defaultExp) {{
              const candidate = parseFloat(opt.dataset.defaultExp);
              if (Number.isFinite(candidate)) optionDefaultExp = candidate;
            }}
          }}
          const needsExp = (mode === 'pair_exp' || mode === 'types_exp');
          let exp = needsExp ? (expSlider ? parseFloat(expSlider.value) : optionDefaultExp) : optionDefaultExp;
          if (!Number.isFinite(exp)) exp = optionDefaultExp;
          const code = String(Math.round(exp * 100)).padStart(3, '0');
          const target = needsExp ? `${{sid}}-${{mode}}_${{code}}` : `${{sid}}-${{mode}}`;
          panels.forEach(p => {{
            const isTarget = p.id === target;
            p.classList.toggle('active', isTarget);
            p.style.display = isTarget ? 'block' : 'none';
          }});
          if (expSlider) {{
            if (needsExp) {{
              expSlider.removeAttribute('disabled');
              expSlider.value = exp.toFixed(2);
              if (expVal) expVal.textContent = exp.toFixed(2);
            }} else {{
              expSlider.setAttribute('disabled', 'disabled');
              expSlider.value = optionDefaultExp.toFixed(2);
              if (expVal) expVal.textContent = '--';
            }}
          }} else if (expVal) {{
            expVal.textContent = needsExp ? exp.toFixed(2) : '--';
          }}
        }}

        if (modeSel) modeSel.addEventListener('change', () => showPanel());
        if (expSlider) {{
          expSlider.addEventListener('input', () => showPanel());
          expSlider.addEventListener('change', () => showPanel());
        }}
        showPanel();
      }});
    }})();
  </script>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")


def render_reduction_card(row: Dict[str, object], figure_map: Dict[str, go.Figure]) -> str:
    reduction = row.get("reduction", "?")
    metric = row.get("metric", "")
    scenario_name = row.get("scenario", f"{reduction} | {metric}")
    sid = f"red-{reduction.lower()}-{metric.lower()}-{int(row.get('figure_seed', 0))}"

    mode_entries: List[Tuple[str, Optional[float], go.Figure]] = []
    for key, fig in figure_map.items():
        if not key.startswith(f"{scenario_name}||{reduction}||"):
            continue
        suffix = key.split("||")[-1]
        if suffix == "raw_total":
            mode = "raw_total"
            exponent = None
        elif suffix.startswith("pair_exp_"):
            exponent = int(suffix.rsplit("_", 1)[-1]) / 100.0
            mode = "pair_exp"
        elif suffix.startswith("types_exp_"):
            exponent = int(suffix.rsplit("_", 1)[-1]) / 100.0
            mode = "types_exp"
        else:
            continue
        mode_entries.append(((0 if mode == "raw_total" else 1 if mode == "pair_exp" else 2, exponent or 0.0), mode, exponent, suffix, fig))

    mode_entries.sort(key=lambda item: item[0])

    panels: List[str] = []
    default_mode = row.get("default_mode", "pair_exp")
    default_exp = row.get("default_exp", 1.0)
    if mode_entries:
        default_mode = mode_entries[0][1] if default_mode not in {entry[1] for entry in mode_entries} else default_mode
        if default_mode in {entry[1] for entry in mode_entries}:
            for ordering, mode, exponent, suffix, fig in mode_entries:
                key = f"{sid}-{suffix}"
                is_active = (mode == default_mode and (
                    (exponent is None and default_exp is None)
                    or (exponent is not None and abs(exponent - float(default_exp)) < 1e-9)
                ))
                html = go.Figure(fig).to_html(include_plotlyjs="cdn", full_html=False)
                panels.append(
                    f"<div id='{key}' class='subtab-panel{' active' if is_active else ''}' data-mode='{mode}' data-exp='{exponent if exponent is not None else ''}' style='display:{'block' if is_active else 'none'};'>"
                    f"{html}"
                    "</div>"
                )

    stress = cp.format_value_with_std(row.get("stress_mean"), row.get("stress_std"))
    trust = cp.format_value_with_std(row.get("trustworthiness_mean"), row.get("trustworthiness_std"))
    mixture = cp.format_value_with_std(row.get("mixture_l1_mean_mean"), row.get("mixture_l1_mean_std"))
    seeds_text = cp.format_seed_list(row.get("seeds"))

    def _default_exp_for_mode(mode: str) -> float:
        for _, m, exponent, _, _ in mode_entries:
            if m == mode and exponent is not None:
                return exponent
        return float(default_exp if default_exp is not None else 1.0)

    options_html: List[str] = []
    seen_modes: set[str] = set()
    for _, mode, exponent, suffix, _ in mode_entries:
        if mode in seen_modes:
            continue
        seen_modes.add(mode)
        display = {
            "raw_total": "Rugosidad bruta",
            "pair_exp": "Normalización por pares",
            "types_exp": "Normalización por tipos",
        }.get(mode, mode)
        default_exp_mode = exponent if exponent is not None else _default_exp_for_mode(mode)
        options_html.append(
            f"<option value='{mode}' data-default-exp='{default_exp_mode:.2f}'{' selected' if mode == default_mode else ''}>{display}</option>"
        )

    default_exp_str = f"{float(default_exp):.2f}" if default_exp is not None else "0.00"
    default_exp_display = default_exp_str if default_mode in {"pair_exp", "types_exp"} else "--"
    slider_disabled_attr = "" if default_mode in {"pair_exp", "types_exp"} else " disabled"

    controls = (
        f"<div class='color-controls' data-default-mode='{default_mode}' data-default-exp='{default_exp_str}'>"
        f"  <label>Color por:"
        f"    <select id='{sid}-mode'>"
        f"      {''.join(options_html)}"
        f"    </select>"
        f"  </label>"
        f"  <label> Exponente:"
        f"    <input id='{sid}-exp' type='range' min='0' max='1' step='0.05' value='{default_exp_str}'{slider_disabled_attr}/>"
        f"    <span id='{sid}-exp-val'>{default_exp_display}</span>"
        f"  </label>"
        f"</div>"
    )
    panels_html = f"<div class='subtab-panels'>{''.join(panels)}</div>"

    header = f"<div class='card-header'><strong>{reduction}</strong></div>"
    metrics_line = (
        f"<div class='metrics-line'>Stress: {stress} · Trust: {trust} · Mixture L1: {mixture} · Semillas: {seeds_text}</div>"
    )
    return f"<div class='plot-card' data-sid='{sid}'>{header}{metrics_line}{controls}{panels_html}</div>"


def main() -> None:
    args = parse_args()
    metrics_requested = _normalise_list(args.metrics)
    if not metrics_requested:
        metrics_requested = ["euclidean"]
    reductions_requested = [r.upper() for r in _normalise_list(args.reductions) or ["MDS"]]

    entries = cp.load_chords(args.dyads_query, args.triads_query, args.sevenths_query)
    if not entries:
        raise SystemExit("La consulta no devolvió acordes para construir la comparación.")
    hist, totals, counts, pairs, notes = cp.stack_hist(entries)

    scenarios = cp.build_scenarios([args.proposal], metrics_requested)
    target_ids = _target_preproc_ids(args.proposal, scenarios)
    scenarios = [s for s in scenarios if s["preproc_id"] in target_ids]

    seed_list = cp.parse_seed_list(args.seeds)
    if not seed_list:
        seed_list = [args.seed]

    dist_simplex_cache: Dict[str, np.ndarray] = {}
    preproc_cache: Dict[str, np.ndarray] = {}

    results_by_metric: Dict[str, List[Dict[str, object]]] = {}
    figures: List[Tuple[str, go.Figure]] = []

    for scenario in scenarios:
        preproc_id = scenario["preproc_id"]
        metric = scenario["metric"]

        if preproc_id not in dist_simplex_cache:
            preproc_func = scenario["preproc_func"]
            kwargs = scenario["preproc_kwargs"]
            X, simplex = preproc_func(hist, counts=counts, pairs=pairs, **kwargs)
            preproc_cache[preproc_id] = X
            dist_simplex_cache[preproc_id] = simplex
        X = preproc_cache[preproc_id]
        simplex = dist_simplex_cache[preproc_id]

        try:
            dist_condensed = cp.metric_distance(metric, X, simplex)
        except ValueError as exc:
            print(f"[skip] {preproc_id} / {metric}: {exc}")
            continue

        dist_matrix = squareform(dist_condensed)
        base_matrix = X if metric in {"cosine", "euclidean", "l1", "l2", "cityblock", "manhattan"} else simplex

        seed_rows: List[Dict[str, Optional[float]]] = []
        figure_seed: Optional[int] = None
        embeddings_by_reduction: Dict[str, np.ndarray] = {}

        for reduction in reductions_requested:
            reduction_upper = reduction.upper()
            try:
                embedding = cp.compute_embeddings(dist_condensed, reduction_upper, seed_list[0])
            except ValueError as exc:
                print(f"[skip] {preproc_id} / {metric} / {reduction_upper}: {exc}")
                continue
            embeddings_by_reduction[reduction_upper] = embedding

        if not embeddings_by_reduction:
            continue

        for reduction_upper, embedding in embeddings_by_reduction.items():
            per_seed: List[Dict[str, Optional[float]]] = []
            for seed in seed_list:
                embedding_seed = cp.compute_embeddings(dist_condensed, reduction_upper, seed)
                nn_top1, nn_top2 = cp.evaluate_nn_hits(dist_matrix, entries, simplex)
                mix_mean, mix_max = cp.evaluate_mixture_error(simplex, entries)
                metrics_summary = cp.summarise_embedding_metrics(base_matrix, embedding_seed, dist_matrix)
                per_seed.append(
                    {
                        "metric": metric,
                        "reduction": reduction_upper,
                        "preproc_id": preproc_id,
                        "seed": seed,
                        "nn_hit_top1": nn_top1,
                        "nn_hit_top2": nn_top2,
                        "mixture_l1_mean": mix_mean,
                        "mixture_l1_max": mix_max,
                        **metrics_summary,
                    }
                )
                if figure_seed is None:
                    figure_seed = seed

            summary = cp.aggregate_seed_results(per_seed, seed_list)
            summary.update(
                {
                    "metric": metric,
                    "reduction": reduction_upper,
                    "preproc_id": preproc_id,
                    "figure_seed": figure_seed,
                    "scenario": f"{preproc_id} | {metric}",
                }
            )
            results_by_metric.setdefault(metric, []).append(summary)

            vectors_adjusted = preproc_cache[preproc_id]
            totals_adj = np.sum(np.asarray(vectors_adjusted, dtype=float), axis=1)
            existing_counts = np.sum(
                np.asarray(vectors_adjusted, dtype=float) > cp.COLOR_EXISTING_THRESHOLD,
                axis=1,
            ).astype(float)

            color_modes: List[Tuple[str, Optional[float]]] = []
            if preproc_id == "identity":
                color_modes.append(("raw_total", None))
            for exp in cp.COLOR_EXPONENTS:
                color_modes.append(("pair_exp", exp))
                color_modes.append(("types_exp", exp))

            embedding_fig = embeddings_by_reduction[reduction_upper]
            _prepare_color_panels(
                f"{preproc_id} | {metric}",
                preproc_id,
                totals,
                totals_adj,
                pairs,
                existing_counts,
                base_matrix,
                preproc_cache[preproc_id],
                entries,
                seed_list,
                reduction_upper,
            figures,
            embedding_fig,
            color_modes,
        )

    output_dir = cp.ensure_output_dir(args.output)
    report_path = output_dir / "report.html"
    build_reduction_report(args.proposal.strip(), results_by_metric, figures, report_path, seed_list)
    print(f"[ok] Reporte generado en: {report_path}")


if __name__ == "__main__":
    main()
