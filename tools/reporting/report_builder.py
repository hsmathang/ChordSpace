"""Construye el archivo report.html reutilizando la plantilla externa."""

from __future__ import annotations

import html
import math
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

from tools.reporting.contracts import validate_run_metadata
from tools.reporting.utils import (
    compute_rank,
    format_seed_list,
    format_value_with_std,
)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "report_assets"


def _load_asset(name: str) -> str:
    path = ASSETS_DIR / name
    return path.read_text(encoding="utf-8")


try:
    REPORT_TEMPLATE = _load_asset("template.html")
    REPORT_CSS = _load_asset("styles.css")
    REPORT_JS = _load_asset("script.js")
except FileNotFoundError:  # pragma: no cover
    REPORT_TEMPLATE = ""
    REPORT_CSS = ""
    REPORT_JS = ""

METRIC_INFO_FALLBACK: Dict[str, Dict[str, str]] = {
    "cosine": {
        "title": "Cosine",
        "casual": "Mide el angulo entre perfiles; destaca la forma relativa.",
        "technical": "(d(u,v) = 1 - u.v / (||u|| ||v||)).",
    },
    "js": {
        "title": "Jensen-Shannon",
        "casual": "Comparacion simetrica basada en entropia.",
        "technical": "d_JS = sqrt(0.5 KL(p||m) + 0.5 KL(q||m)).",
    },
    "hellinger": {
        "title": "Hellinger",
        "casual": "Distancia probabilistica equilibrada.",
        "technical": "d_H(p,q) = ||sqrt(p) - sqrt(q)||_2 / sqrt(2).",
    },
    "euclidean": {
        "title": "Euclidiana",
        "casual": "Mide separaciones directas punto a punto.",
        "technical": "||u-v||_2.",
    },
    "l1": {
        "title": "Manhattan",
        "casual": "Suma diferencias absolutas por componente.",
        "technical": "||u-v||_1.",
    },
    "cityblock": {
        "title": "Manhattan",
        "casual": "Suma diferencias absolutas por componente.",
        "technical": "||u-v||_1.",
    },
    "manhattan": {
        "title": "Manhattan",
        "casual": "Suma diferencias absolutas por componente.",
        "technical": "||u-v||_1.",
    },
}


def render_report_html(
    metrics_df: pd.DataFrame,
    figures: List[Tuple[str, go.Figure]],
    output_path: Path,
    seeds: Sequence[int],  # noqa: ARG001 - reservado para futuras variantes
    *,
    run_metadata: Optional[Dict[str, Any]] = None,
    metric_info: Optional[Dict[str, Dict[str, str]]] = None,
    highlight_threshold: int = 2000,
    sections_enabled: Optional[Dict[str, bool]] = None,
    heatmap_data: Optional[Dict[str, Any]] = None,
) -> None:
    metric_catalog = metric_info or METRIC_INFO_FALLBACK

    ranked_df = metrics_df.copy()
    include_section = lambda key, default=True: bool(sections_enabled.get(key, default)) if sections_enabled else default
    ranked_df["rank"] = compute_rank(ranked_df)
    ranked_df.sort_values(by="rank", inplace=True)

    table_html = ""
    secondary_table_html = ""
    if include_section("table", True) and not ranked_df.empty:
        def fmt_metric(row: pd.Series, mean_key: str, std_key: str, *, applies: bool = True) -> str:
            if not applies:
                return "N/A"
            return format_value_with_std(row.get(mean_key), row.get(std_key))

        def fmt_hyperparams(value: Any) -> str:
            if isinstance(value, dict):
                items = [f"{k}={v}" for k, v in value.items()]
                return ", ".join(items) if items else "-"
            return html.escape(str(value)) if value else "-"

        display_df = pd.DataFrame(
            {
                "Ranking": ranked_df["rank"],
                "Escenario": ranked_df["scenario"],
                "Reducción": ranked_df.get("reduction", pd.Series(["-"] * len(ranked_df))),
                "Métrica": ranked_df["metric"],
                "Stress": ranked_df.apply(
                    lambda row: fmt_metric(row, "stress_mean", "stress_std", applies=row.get("reduction") in {"MDS", "ISOMAP"}),
                    axis=1
                ),
                "Trustworthiness": ranked_df.apply(
                    lambda row: fmt_metric(row, "trustworthiness_mean", "trustworthiness_std"), axis=1
                ),
                "Continuidad": ranked_df.apply(
                    lambda row: fmt_metric(row, "continuity_mean", "continuity_std"), axis=1
                ),
                "kNN recall": ranked_df.apply(
                    lambda row: fmt_metric(row, "knn_recall_mean", "knn_recall_std"), axis=1
                ),
                "Shepard R²": ranked_df.apply(
                    lambda row: fmt_metric(row, "shepard_r2_mean", "shepard_r2_std"), axis=1
                ),
                "Silhouette": ranked_df.apply(
                    lambda row: fmt_metric(row, "silhouette_mean", "silhouette_std"), axis=1
                ),
                "Davies-Bouldin": ranked_df.apply(
                    lambda row: fmt_metric(row, "davies_bouldin_mean", "davies_bouldin_std"), axis=1
                ),
                "KL (TSNE)": ranked_df.apply(
                    lambda row: fmt_metric(row, "kl_divergence_mean", "kl_divergence_std", applies=row.get("reduction") == "TSNE"),
                    axis=1
                ),
                "Mixture L1": ranked_df.apply(
                    lambda row: fmt_metric(row, "mixture_l1_mean_mean", "mixture_l1_mean_std"), axis=1
                ),
                "Hiperparámetros": ranked_df["hyperparams"].apply(fmt_hyperparams)
                if "hyperparams" in ranked_df.columns
                else "-",
                "Semillas": ranked_df["seeds"].apply(format_seed_list),
            }
        )
        table_html = display_df.to_html(index=False, escape=True)

    if include_section("secondary_metrics", False) and not ranked_df.empty:
        def fmt_secondary(row: pd.Series, key: str) -> str:
            return format_value_with_std(row.get(f"{key}_mean"), row.get(f"{key}_std"))

        card_mean_cols = sorted(
            {
                col[:-5]
                for col in ranked_df.columns
                if col.startswith("knn_hit_card_") and col.endswith("_mean")
            }
        )
        secondary_rows: List[Dict[str, Any]] = []
        for _, row in ranked_df.iterrows():
            sec_row: Dict[str, Any] = {
                "Escenario": row["scenario"],
                "MRRE": fmt_secondary(row, "relative_rank_error"),
                "Var comp1": fmt_secondary(row, "var_ratio_dim1"),
                "Var comp2": fmt_secondary(row, "var_ratio_dim2"),
                "LogReg acc": fmt_secondary(row, "cardinality_logreg_acc"),
                "Memoria (KB)": fmt_secondary(row, "embedding_mem_kb"),
                "Tiempo (s)": (
                    f"{float(row.get('runtime_seconds', 0.0)):.2f}"
                    if row.get("runtime_seconds") is not None
                    else "-"
                ),
            }
            for base in card_mean_cols:
                label = base.split("_")[-1]
                sec_row[f"kNN n={label}"] = fmt_secondary(row, base)
            secondary_rows.append(sec_row)
        secondary_df = pd.DataFrame(secondary_rows)
        secondary_table_html = (
            "<h3>Métricas secundarias</h3>" + secondary_df.to_html(index=False, escape=True)
        )

    # Agrupar figuras por escenario y sección
    figure_groups: Dict[str, Dict[str, go.Figure]] = defaultdict(dict)
    scatter_prefixes = ("raw_total", "pair_exp", "types_exp")
    for title, fig in figures:
        if "||" in title:
            scenario_name, suffix = title.split("||", 1)
        else:
            scenario_name, suffix = title, "raw_total"
        suffix_lower = suffix.lower()
        if suffix_lower.startswith(scatter_prefixes):
            if not include_section("scatter", True):
                continue
        elif suffix_lower == "heatmap":
            if not include_section("heatmap", True):
                continue
            suffix = "heatmap"
        elif suffix_lower == "shepard":
            if not include_section("shepard", True):
                continue
            suffix = "shepard"
        else:
            if not include_section("scatter", True):
                continue
        figure_groups[scenario_name][suffix] = fig

    preferred_order = ["euclidean", "cosine", "js", "hellinger", "l1", "cityblock", "manhattan"]

    def _metric_key(metric: str) -> Tuple[int, str]:
        idx = preferred_order.index(metric) if metric in preferred_order else len(preferred_order)
        return (idx, metric)

    reductions_present = list(dict.fromkeys(ranked_df.get("reduction", pd.Series(["MDS"])).tolist()))

    include_js = True
    def fig_to_html(fig_obj: go.Figure) -> str:
        nonlocal include_js
        if fig_obj is None:
            return "<p>No hay figura disponible.</p>"
        inner = to_html(fig_obj, include_plotlyjs="cdn" if include_js else False, full_html=False)
        include_js = False
        return inner
    subtab_counter = 0

    def render_card(row: Dict[str, Any], *, is_baseline: bool) -> str:
        nonlocal include_js, subtab_counter
        if row is None:
            return ""
        scenario = row.get("scenario", "")
        fig_bucket = figure_groups.get(scenario, {})

        card_highlight_info: Optional[Dict[str, Any]] = None
        panel_entries: List[Tuple[Tuple[int, float], str, Optional[float], str, str]] = []
        heatmap_html = ""
        shepard_html = ""
        for suffix, fig in fig_bucket.items():
            suffix_lower = suffix.lower()
            if suffix_lower == "heatmap":
                heatmap_html = fig_to_html(fig)
                continue
            if suffix_lower == "shepard":
                shepard_html = fig_to_html(fig)
                continue

            if suffix == "raw_total":
                mode = "raw_total"
                exponent = None
                order = (0, 0.0)
            elif suffix.startswith("pair_exp_"):
                try:
                    exponent = int(suffix.rsplit("_", 1)[-1]) / 100.0
                except Exception:
                    continue
                mode = "pair_exp"
                order = (1, float(exponent))
            elif suffix.startswith("types_exp_"):
                try:
                    exponent = int(suffix.rsplit("_", 1)[-1]) / 100.0
                except Exception:
                    continue
                mode = "types_exp"
                order = (2, float(exponent))
            else:
                continue

            if fig is not None:
                if card_highlight_info is None or not bool(card_highlight_info.get("enabled")):
                    meta_obj = getattr(fig, "layout", None)
                    meta_dict = getattr(meta_obj, "meta", None) if meta_obj is not None else None
                    if isinstance(meta_dict, dict):
                        fh = meta_dict.get("familyHighlight")
                        if isinstance(fh, dict):
                            candidate_info = dict(fh)
                            if card_highlight_info is None or candidate_info.get("enabled"):
                                card_highlight_info = candidate_info
                inner_html = fig_to_html(fig)
            else:
                inner_html = "<p>No hay figura disponible.</p>"

            # Usamos el sufijo original tal cual (raw_total, pair_exp_050, ...),
            # porque el JS construye los IDs a partir de ese patrón.
            suffix_label = suffix

            panel_entries.append((order, mode, exponent, suffix_label, inner_html))

        panel_entries.sort(key=lambda item: item[0])
        has_scatter = bool(panel_entries)
        has_auxiliary = bool(heatmap_html or shepard_html)
        if not has_scatter and not has_auxiliary:
            return ""

        controls_html = ""
        panels_block = ""
        inversion_controls_html = ""
        highlight_note_html = ""
        default_mode = "raw_total"
        default_exponent = None

        if has_scatter:
            mode_defaults: Dict[str, float] = {}
            for _, mode, exponent, _, _ in panel_entries:
                if mode not in mode_defaults and exponent is not None:
                    mode_defaults[mode] = exponent

            default_mode = panel_entries[0][1]
            default_exponent = panel_entries[0][2]
            if default_mode != "raw_total":
                found = False
                for _, mode, exponent, _, _ in panel_entries:
                    if mode == "pair_exp" and exponent is not None and abs(exponent - 1.0) < 1e-9:
                        default_exponent = exponent
                        found = True
                        break
                if not found:
                    _, default_mode, default_exponent, _, _ = panel_entries[0]

            mode_labels = {
                "raw_total": "Rugosidad bruta",
                "pair_exp": "Normalizacion por pares",
                "types_exp": "Normalizacion por tipos",
            }

            mode_order: List[str] = []
            for _, mode, _, _, _ in panel_entries:
                if mode not in mode_order:
                    mode_order.append(mode)

            options_html: List[str] = []
            for mode in mode_order:
                label = mode_labels.get(mode, mode)
                selected_attr = " selected" if mode == default_mode else ""
                default_exp_val = mode_defaults.get(mode, 0.0)
                options_html.append(
                    f"<option value='{mode}' data-default-exp='{default_exp_val:.2f}'{selected_attr}>{label}</option>"
                )

            panels_html: List[str] = []
            for _, mode, exponent, suffix, inner_html in panel_entries:
                target_id = f"card-{subtab_counter}-{suffix}"
                is_active = mode == default_mode and (
                    (exponent is None and default_exponent is None)
                    or (
                        exponent is not None
                        and default_exponent is not None
                        and abs(exponent - default_exponent) < 1e-9
                    )
                )
                exp_str = f"{exponent:.2f}" if exponent is not None else ""
                style_attr = " style='display:block;'" if is_active else " style='display:none;'"
                panels_html.append(
                    f"<div id='{target_id}' class='subtab-panel{' active' if is_active else ''}' "
                    f"data-mode='{mode}' data-exp='{exp_str}'{style_attr}>{inner_html}</div>"
                )

            panels_block = "<div class='subtab-panels'>" + "".join(panels_html) + "</div>"
            default_exp_str = f"{default_exponent:.2f}" if default_exponent is not None else "0.00"
            default_exp_display = f"{default_exponent:.2f}" if default_exponent is not None else "--"
            slider_disabled_attr = "" if default_mode in {"pair_exp", "types_exp"} else " disabled"

            controls_html = (
                f"<div class='color-controls' data-default-mode='{default_mode}' data-default-exp='{default_exp_str}'>"
                f"  <label>Color por:"
                f"    <select id='card-{subtab_counter}-mode'>"
                f"      {''.join(options_html)}"
                f"    </select>"
                f"  </label>"
                f"  <label> Exponente:"
                f"    <input id='card-{subtab_counter}-exp' type='range' min='0' max='1' step='0.05' "
                f"value='{default_exp_str}'{slider_disabled_attr}/>"
                f"    <span id='card-{subtab_counter}-exp-val'>{default_exp_display}</span>"
                f"  </label>"
                f"</div>"
            )
            inversion_controls_html = """
        <div class='inversion-controls'>
            <label><input type='checkbox' class='inversion-toggle' data-inversion-type='musical'> Resaltar inversiones musicales</label>
            <label><input type='checkbox' class='inversion-toggle' data-inversion-type='structural'> Resaltar inversiones estructurales</label>
            <label><input type='checkbox' class='substitution-toggle'> Resaltar sustituciones</label>
            <label class='substitution-profile-label'>
                Perfil de sustitucion:
                <select class='substitution-profile' disabled>
                    <option value=''>Cargando...</option>
                </select>
            </label>
        </div>
        """
            highlight_enabled_flag = bool(card_highlight_info and card_highlight_info.get("enabled"))
            if highlight_enabled_flag:
                highlight_note_html = "<p class='highlight-note'>Haz clic sobre un punto para resaltar su familia.</p>"
            else:
                highlight_note_html = "<p class='highlight-note muted'>Sin familias destacadas en este escenario.</p>"
        else:
            reason = (
                "La sección Scatter está desactivada en la GUI."
                if not include_section("scatter", True)
                else "Scatter no generado para este escenario."
            )
            panels_block = f"<div class='subtab-panels empty'><p class='empty-panel'>{reason}</p></div>"
            highlight_enabled_flag = False

        stress = format_value_with_std(row.get("stress_mean"), row.get("stress_std"))
        trust = format_value_with_std(row.get("trustworthiness_mean"), row.get("trustworthiness_std"))
        mixture = format_value_with_std(row.get("mixture_l1_mean_mean"), row.get("mixture_l1_mean_std"))
        seeds_text = format_seed_list(row.get("seeds"))
        badge = "<span class='badge'>Base</span>" if is_baseline else ""
        header = f"<div class='card-header'><strong>{scenario}</strong> {badge}</div>"
        metrics_line = (
            f"<div class='metrics-line'>Stress: {stress} | Trust: {trust} | Mixture L1: {mixture} | Semillas: {seeds_text}</div>"
        )

        card_attrs = [
            f"data-sid='card-{subtab_counter}'",
            f"data-family-highlight='{'1' if highlight_enabled_flag else '0'}'",
        ]
        if card_highlight_info:
            families_detected = int(card_highlight_info.get("families", 0) or 0)
            threshold_limit = int(card_highlight_info.get("threshold", highlight_threshold) or highlight_threshold)
            card_attrs.append(f"data-families='{families_detected}'")
            card_attrs.append(f"data-highlight-threshold='{threshold_limit}'")
        else:
            card_attrs.append(f"data-highlight-threshold='{highlight_threshold}'")

        subtab_counter += 1
        card_attrs_str = " ".join(card_attrs)
        detail_panel = (
            "<div class='detail-panel' "
            "data-default-msg='Haz clic en un punto para ver el detalle completo.'>"
            "Haz clic en un punto para ver el detalle completo."
            "</div>"
        )
        aux_sections = ""
        if heatmap_html:
            aux_sections += (
                "<details class='aux-figure' open data-section='heatmap'>"
                "<summary>Heatmap de distancias</summary>"
                f"<div class='figure-body'>{heatmap_html}</div>"
                "</details>"
            )
        if shepard_html:
            aux_sections += (
                "<details class='aux-figure' open data-section='shepard'>"
                "<summary>Shepard (distancias originales vs reducidas)</summary>"
                f"<div class='figure-body'>{shepard_html}</div>"
                "</details>"
            )

        return (
            f"<div class='plot-card' {card_attrs_str}>{header}{metrics_line}{controls_html}"
            f"{highlight_note_html}{inversion_controls_html}{panels_block}{aux_sections}{detail_panel}</div>"
        )

    outer_headers: List[str] = []
    outer_bodies: List[str] = []
    for ridx, reduction in enumerate(reductions_present):
        active_cls = " active" if ridx == 0 else ""
        outer_headers.append(f"<li class='tab{active_cls}' data-target='tab-red-{ridx}'>{reduction}</li>")
        red_group = ranked_df[ranked_df.get("reduction").fillna("MDS") == reduction]
        metrics_present = list(dict.fromkeys(red_group["metric"].tolist()))
        metrics_sorted = sorted(metrics_present, key=_metric_key)
        inner_headers: List[str] = []
        inner_bodies: List[str] = []
        for midx, metric in enumerate(metrics_sorted):
            metric_data = metric_catalog.get(metric, {"title": metric.upper(), "casual": "", "technical": ""})
            inner_headers.append(
                f"<li class='subtab{' active' if midx == 0 else ''}' data-target='tab-{ridx}-{metric}'>"
                f"{metric_data['title']}</li>"
            )
            group = red_group[red_group["metric"] == metric].sort_values("rank")
            baseline_df = group[group["preproc_id"] == "identity"]
            baseline_row = baseline_df.iloc[0].to_dict() if not baseline_df.empty else None
            proposal_rows = [
                row._asdict() if hasattr(row, "_asdict") else row
                for row in group[group["preproc_id"] != "identity"].to_dict("records")
            ]
            body_parts: List[str] = [
                f"<div id='tab-{ridx}-{metric}' class='tab-panel{' active' if midx == 0 else ''}'>",
                f"<div class='metric-intro'><p>{metric_data['casual']}</p>"
                f"<p><em>Detalle tecnico:</em> {metric_data['technical']}</p></div>",
            ]
            if baseline_row is None and not proposal_rows:
                body_parts.append("<p>No se registraron resultados para esta metrica.</p>")
            elif not proposal_rows:
                base_card = render_card(baseline_row, is_baseline=True)
                body_parts.append(f"<div class='plot-grid'>{base_card}</div>")
            else:
                for pr in proposal_rows:
                    cards = []
                    base_card = render_card(baseline_row, is_baseline=True)
                    if base_card:
                        cards.append(base_card)
                    cards.append(render_card(pr, is_baseline=False))
                    body_parts.append(f"<div class='plot-grid'>{''.join(cards)}</div>")
            body_parts.append("</div>")
            inner_bodies.append("".join(body_parts))
        inner_tabs = (
            "<div class='subtabs'>"
            f"  <ul class='subtab-headers'>{''.join(inner_headers)}</ul>"
            f"  <div class='tab-panels'>{''.join(inner_bodies)}</div>"
            "</div>"
        )
        outer_bodies.append(f"<div id='tab-red-{ridx}' class='tab-panel{active_cls}'>" + inner_tabs + "</div>")

    tabs_html = (
        "<div class='tabs'>"
        f"<ul class='tab-headers'>{''.join(outer_headers)}</ul>"
        f"<div class='tab-panels'>{''.join(outer_bodies)}</div>"
        "</div>"
    )

    metadata_html = ""
    if run_metadata and include_section("metadata", True):
        validate_run_metadata(run_metadata)
        selection = run_metadata.get("selection", {}) if isinstance(run_metadata, dict) else {}
        population = run_metadata.get("population", {}) if isinstance(run_metadata, dict) else {}
        rows_selected = selection.get("rows_selected")
        rows_available = selection.get("rows_available")
        mode_label = selection.get("mode") or "-"
        payload_used = bool(selection.get("payload_path"))
        payload_reason = selection.get("payload_reason")

        selection_rows: list[tuple[str, str]] = []
        if rows_selected is not None and rows_available is not None:
            selection_rows.append(
                ("Seleccion", f"{int(rows_selected)} de {int(rows_available)} acordes")
            )
        selection_rows.append(("Modo de ejecucion", html.escape(str(mode_label))))
        selection_rows.append(("Payload JSON", "si" if payload_used else "no"))
        if payload_reason:
            selection_rows.append(("Motivo payload", html.escape(str(payload_reason))))

        descriptors = population.get("descriptors") or []
        source_rows: list[tuple[str, str]] = []
        for idx, desc in enumerate(descriptors, start=1):
            if not isinstance(desc, dict):
                continue
            mode = desc.get("mode", "desconocido")
            rows = desc.get("rows")
            if mode == "combinatorial":
                combo = desc.get("combinatorial") or {}
                alphabet = combo.get("alphabet") or []
                cards = combo.get("cardinalities") or []
                oct_min = combo.get("octave_min")
                oct_max = combo.get("octave_max")
                structural = combo.get("structural_mode")
                parts = [
                    "Combinatoria",
                    f"{rows} acordes" if rows is not None else "",
                    "alfabeto=" + ", ".join(str(v) for v in alphabet) if alphabet else "",
                    f"octavas={oct_min}-{oct_max}" if oct_min is not None and oct_max is not None else "",
                    "n=" + ", ".join(str(v) for v in cards) if cards else "",
                    "modo estructural=si" if structural else "modo estructural=no",
                ]
            else:
                db = desc.get("database") or {}
                base_q = db.get("base_query") or "<ninguna>"
                pops = db.get("pops_entries") or []
                filt_mode = db.get("filter_mode")
                parts = [
                    "Base de datos",
                    f"{rows} acordes" if rows is not None else "",
                    f"base={base_q}",
                    "poblaciones=" + ", ".join(str(v) for v in pops) if pops else "",
                    f"modo filtros={filt_mode}" if filt_mode else "",
                ]
            filt = desc.get("filters") or {}
            label = filt.get("label")
            if label:
                parts.append(f"filtros={label}")
            clean = [html.escape(str(p)) for p in parts if p]
            if clean:
                source_rows.append((f"Fuente {idx}", " | ".join(clean)))
            detail = filt.get("detail")
            if isinstance(detail, dict):
                detail_lines: List[str] = []
                cards = detail.get("cardinalities")
                if cards:
                    detail_lines.append("Cardinalidades filtradas: " + ", ".join(map(str, cards)))
                span = detail.get("span")
                if isinstance(span, dict):
                    detail_lines.append(
                        f"Span permitido: {span.get('min', '-')}-{span.get('max', '-')}"
                    )
                max_int = detail.get("max_internal_interval")
                if max_int is not None:
                    detail_lines.append(f"Intervalo interno máximo: {max_int}")
                include_pcs = detail.get("include_pitch_classes")
                if include_pcs:
                    detail_lines.append("PCs incluidas: " + ", ".join(map(str, include_pcs)))
                exclude_pcs = detail.get("exclude_pitch_classes")
                if exclude_pcs:
                    detail_lines.append("PCs excluidas: " + ", ".join(map(str, exclude_pcs)))
                patterns = detail.get("interval_patterns")
                if patterns:
                    detail_lines.append("Patrones de intervalo: " + ", ".join(map(str, patterns)))
                interval_mode = detail.get("interval_mode")
                if interval_mode:
                    detail_lines.append(f"Modo de intervalos: {interval_mode}")
                pc_mode = detail.get("pc_mode")
                if pc_mode:
                    detail_lines.append(f"Modo de pitch classes: {pc_mode}")
                if detail.get("scale_expand"):
                    detail_lines.append("Extensión de escala activa")
                if detail_lines:
                    source_rows.append(
                        (f"Fuente {idx} · filtros", "<br>".join(html.escape(line) for line in detail_lines))
                    )

        def _render_meta_table(rows: list[tuple[str, str]]) -> str:
            if not rows:
                return ""
            body = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in rows)
            return "<table class='meta-table'><tbody>" + body + "</tbody></table>"

        metadata_html = (
            "<section class='meta-section'>"
            "<h3>Configuracion de la poblacion</h3>"
            f"{_render_meta_table(selection_rows)}"
            f"{_render_meta_table(source_rows)}"
            "</section>"
        )

    combined_tables = table_html + secondary_table_html

    # Inject Heatmap Data if present
    js_prefix = ""
    if heatmap_data:
        json_payload = json.dumps(heatmap_data, ensure_ascii=False)
        js_prefix = f"window.HEATMAP_DATA = {json_payload};\n"

    final_js = js_prefix + REPORT_JS

    if REPORT_TEMPLATE:
        html_content = (
            REPORT_TEMPLATE.replace("__CSS__", REPORT_CSS)
            .replace("__TABLE_HTML__", combined_tables)
            .replace("__METADATA_HTML__", metadata_html)
            .replace("__TABS_HTML__", tabs_html)
            .replace("__SCRIPT_JS__", final_js)
        )
        html_content = html_content.replace(
            "Comparacion de Propuestas de Rugosidad", "Exploración del espacio de acordes"
        )
    else:
        html_content = (
            "<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'/>"
            "<title>Exploración del espacio de acordes</title></head><body>"
            "<h1>Exploración del espacio de acordes</h1>"
            "<h3>Resumen global</h3>"
            f"{table_html}{metadata_html}{tabs_html}"
            f"<script>{final_js}</script>"
            "</body></html>"
        )

    output_path.write_text(html_content, encoding="utf-8")


__all__ = ["render_report_html"]
