"""Utilities for building Plotly visualisations used across proposal comparisons."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

if TYPE_CHECKING:  # pragma: no cover - hints only
    from tools.compare_proposals import ChordEntry  # circular only for type checking


HighlightConfig = Dict[str, float]


def _to_serialisable(value: Any) -> Any:
    """Convert values to JSON-serialisable representations."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_serialisable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serialisable(v) for k, v in value.items()}
    return value


@dataclass
class FigureSpec:
    """Container for a proposal visualisation.

    Attributes
    ----------
    key:
        Unique key used in HTML and caching (``"scenario||mode"``).
    scenario:
        Scenario name (proposal/metric/reduction combination).
    metric:
        Identifier of the metric used to build the embedding.
    reduction:
        Reduction method (MDS, TSNE, UMAP, ...).
    proposal:
        Normalisation proposal identifier.
    mode:
        Colouring mode (``raw_total``, ``pair_exp`` or ``types_exp``).
    exponent:
        Optional exponent associated with ``pair_exp``/``types_exp``.
    seed:
        Seed used to generate the embedding (if deterministic).
    payload:
        Serialisable Plotly dictionary containing ``data``/``layout``.
    """

    key: str
    scenario: str
    metric: str
    reduction: str
    proposal: str
    mode: str
    exponent: Optional[float]
    seed: Optional[int]
    payload: Dict[str, Any]
    _figure: Optional[go.Figure] = field(default=None, init=False, repr=False)

    def to_figure(self) -> go.Figure:
        """Materialise the Plotly ``Figure`` object for inline rendering."""
        if self._figure is None:
            self._figure = go.Figure(
                data=self.payload.get("data", []),
                layout=self.payload.get("layout", {}),
            )
        return self._figure

    def to_json(self) -> str:
        """Return a Plotly JSON string."""
        return json.dumps(self.payload, cls=PlotlyJSONEncoder)

    def serializable(self) -> Dict[str, Any]:
        """Return metadata + payload ready to dump as JSON."""
        return {
            "key": self.key,
            "scenario": self.scenario,
            "metric": self.metric,
            "reduction": self.reduction,
            "proposal": self.proposal,
            "mode": self.mode,
            "exponent": self.exponent,
            "seed": self.seed,
            "figure": self.payload,
        }


def _format_vec(vec: np.ndarray, *, precision: int = 2, max_len: int = 12) -> str:
    slice_vec = vec[:max_len]
    values = ", ".join(f"{float(v):.{precision}f}" for v in slice_vec)
    if len(vec) > max_len:
        values += ", ..."
    return f"[{values}]"


def build_hover(
    entry: "ChordEntry",
    vector_used: np.ndarray,
    vector_adjusted: np.ndarray,
    color_value: float,
    color_title: str,
    pair_count: int,
    type_count: int,
    *,
    is_proposal: bool,
    family_size: Optional[int] = None,
) -> str:
    acorde = entry.acorde
    intervals = getattr(acorde, "intervals", [])
    tipo = getattr(acorde, "name", "Unknown")
    total = entry.total
    n = entry.n_notes
    identity_label = entry.identity_name if entry.is_named else "Desconocido"
    alias_line = ""
    if entry.identity_aliases:
        alias_line = f"Alias: {', '.join(entry.identity_aliases)}<br>"
    color_line = f"{color_title}: {float(color_value):.4f}<br>"
    pair_line = f"Pares totales (P): {pair_count}<br>"
    type_line = f"Tipos activos (PE): {type_count}<br>"
    family_line = ""
    has_family_id = entry.family_id is not None
    if has_family_id or entry.is_inversion:
        family_label = str(entry.family_id) if has_family_id else "—"
        role = "Inversión" if entry.is_inversion else "Acorde base"
        details: List[str] = []
        if family_size is not None and family_size > 0:
            details.append(f"miembros: {family_size}")
        if entry.is_inversion and entry.inversion_rotation is not None:
            details.append(f"rotación: {entry.inversion_rotation}")
        details_text = (
            f" ({role}{', ' + ', '.join(details) if details else ''})"
            if role or details
            else ""
        )
        family_line = f"Familia: {family_label}{details_text}<br>"
    if is_proposal:
        total_adj = float(np.sum(vector_adjusted))
        return (
            f"Acorde: {tipo}<br>"
            f"Notas: {n}<br>"
            f"Intervalos: {intervals}<br>"
            f"Identidad: {identity_label}<br>"
            f"{alias_line}"
            f"{family_line}"
            f"TotalRug (bruto): {total:.4f}<br>"
            f"TotalRug (ajustado): {total_adj:.4f}<br>"
            f"H bruto: {_format_vec(entry.hist)}<br>"
            f"H ajustado: {_format_vec(vector_adjusted)}<br>"
            f"{color_line}"
            f"{pair_line}"
            f"{type_line}"
        )
    return (
        f"Acorde: {tipo}<br>"
        f"Notas: {n}<br>"
        f"Intervalos: {intervals}<br>"
        f"Identidad: {identity_label}<br>"
        f"{alias_line}"
        f"{family_line}"
        f"TotalRug: {total:.4f}<br>"
        f"{color_line}"
        f"{pair_line}"
        f"{type_line}"
        f"H bruto: {_format_vec(entry.hist)}<br>"
    )


def build_hover_summary(
    entry: "ChordEntry",
    family_size: Optional[int],
    color_value: float,
    color_title: str,
) -> str:
    acorde = entry.acorde
    name = getattr(acorde, "name", None)
    if not name or name == "Unknown":
        name = entry.identity_name or "Acorde"
    intervals = getattr(acorde, "intervals", [])
    interval_label = ""
    try:
        if intervals:
            interval_label = " " + "[" + ",".join(str(int(i)) for i in intervals) + "]"
    except Exception:
        interval_label = ""
    fam_label = family_size if family_size and family_size > 0 else 1
    return (
        f"{name}{interval_label} · {color_title}: {float(color_value):.2f} · "
        f"Familia: {fam_label}"
    )


DEFAULT_HIGHLIGHT: HighlightConfig = {
    "threshold": 2000.0,
    "size_scale": 1.35,
    "size_delta": 3.0,
    "selected_opacity": 0.95,
    "fade_factor": 0.25,
}


def build_scatter_payload(
    embedding: np.ndarray,
    entries: Sequence["ChordEntry"],
    color_values: np.ndarray,
    pair_counts: np.ndarray,
    type_counts: np.ndarray,
    vectors: np.ndarray,
    adjusted_vectors: np.ndarray,
    *,
    title: str,
    color_title: str,
    is_proposal: bool,
    highlight: Optional[HighlightConfig] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    highlight_cfg = dict(DEFAULT_HIGHLIGHT)
    if highlight:
        highlight_cfg.update({k: float(v) for k, v in highlight.items()})

    x = np.asarray(embedding[:, 0], dtype=float)
    y = np.asarray(embedding[:, 1], dtype=float)
    color_values = np.asarray(color_values, dtype=float)
    cmin = float(np.min(color_values)) if len(color_values) else 0.0
    cmax = float(np.max(color_values)) if len(color_values) else 0.0

    total_points = len(entries)
    family_tags: List[str] = []
    family_counts: Dict[str, int] = {}
    highlight_summary: Dict[str, Any] = {
        "enabled": False,
        "threshold": int(highlight_cfg["threshold"]),
        "total_points": total_points,
        "families": 0,
        "size_scale": highlight_cfg["size_scale"],
        "size_delta": highlight_cfg["size_delta"],
        "selected_opacity": highlight_cfg["selected_opacity"],
        "fade_factor": highlight_cfg["fade_factor"],
        "has_inversions": any(getattr(e, "is_inversion", False) for e in entries),
    }

    if total_points:
        def _normalize_family(raw_value: Optional[object], idx: int) -> str:
            if raw_value is None:
                return f"__solo_{idx}"
            if isinstance(raw_value, float) and np.isnan(raw_value):
                return f"__solo_{idx}"
            return str(raw_value)

        for idx, entry in enumerate(entries):
            tag = _normalize_family(getattr(entry, "family_id", None), idx)
            family_tags.append(tag)
            family_counts[tag] = family_counts.get(tag, 0) + 1

        families_with_links = sum(1 for count in family_counts.values() if count > 1)
        highlight_enabled = (
            total_points <= highlight_cfg["threshold"] and families_with_links > 0
        )
        highlight_summary.update(
            {
                "enabled": bool(highlight_enabled),
                "families": int(families_with_links),
            }
        )
    else:
        highlight_enabled = False

    detail_texts: List[str] = []
    summary_texts: List[str] = []
    for idx in range(total_points):
        fam_size = family_counts.get(family_tags[idx], 1)
        detail_texts.append(
            build_hover(
                entries[idx],
                vectors[idx],
                adjusted_vectors[idx],
                float(color_values[idx]),
                color_title,
                int(round(pair_counts[idx])),
                int(round(type_counts[idx])),
                is_proposal=is_proposal,
                family_size=fam_size,
            )
        )
        summary_texts.append(
            build_hover_summary(
                entries[idx],
                fam_size,
                float(color_values[idx]),
                color_title,
            )
        )

    customdata_all = [
        [
            family_tags[i],
            1 if getattr(entries[i], "is_inversion", False) else 0,
            family_counts.get(family_tags[i], 1),
            summary_texts[i],
            detail_texts[i],
        ]
        for i in range(total_points)
    ]

    def _base_marker_params(count: int) -> Tuple[float, float]:
        if count <= 40:
            return 20.0, 0.5
        if count >= 1000:
            return 6.0, 0.2
        frac = (count - 40) / (1000 - 40)
        size = 20.0 - frac * (20.0 - 6.0)
        opacity = 0.5 - frac * (0.5 - 0.2)
        return max(size, 4.0), max(min(opacity, 0.5), 0.2)

    def _highlight_markers(base_size: float, base_opacity: float) -> Tuple[Dict[str, object], Dict[str, object]]:
        selected_size = max(base_size * highlight_cfg["size_scale"], base_size + highlight_cfg["size_delta"])
        selected_opacity = min(1.0, max(highlight_cfg["selected_opacity"], base_opacity + 0.35))
        unselected_opacity = max(0.05, base_opacity * highlight_cfg["fade_factor"])
        selected_marker: Dict[str, object] = {
            "size": selected_size,
            "opacity": selected_opacity,
        }
        unselected_marker: Dict[str, object] = {
            "opacity": unselected_opacity,
        }
        return selected_marker, unselected_marker

    base_size, base_opacity = _base_marker_params(total_points)

    named_groups = {
        "Diadas": [],
        "Triadas": [],
        "Séptimas": [],
        "Extensiones": [],
    }
    unnamed_groups: Dict[str, List[int]] = {
        "3 notas": [],
        "4 notas": [],
        "5 notas": [],
        "Más de 5 notas": [],
    }

    def _classify_named(entry: "ChordEntry") -> Optional[str]:
        if not entry.is_named:
            return None
        n = entry.n_notes
        name = (entry.identity_name or "").lower()
        if n == 2:
            return "Diadas"
        if n == 3:
            return "Triadas"
        if n == 4 and "7" in name:
            return "Séptimas"
        return "Extensiones"

    for idx, entry in enumerate(entries):
        category = _classify_named(entry)
        if category is not None:
            named_groups[category].append(idx)
            continue
        if entry.n_notes == 3:
            unnamed_groups["3 notas"].append(idx)
        elif entry.n_notes == 4:
            unnamed_groups["4 notas"].append(idx)
        elif entry.n_notes == 5:
            unnamed_groups["5 notas"].append(idx)
        else:
            unnamed_groups["Más de 5 notas"].append(idx)

    named_symbol_map = {
        "Diadas": "diamond",
        "Triadas": "triangle-down",
        "Séptimas": "square",
        "Extensiones": "cross",
    }
    unnamed_symbol_map = {
        "3 notas": "triangle-up",
        "4 notas": "x",
        "5 notas": "star",
        "Más de 5 notas": "circle",
    }

    named_size = max(base_size + 2.5, base_size * 1.2, 8.0)
    named_opacity = min(0.9, base_opacity + 0.25)

    traces: List[Dict[str, Any]] = []

    def _build_trace(idxs: Sequence[int], *, label: str, marker: Dict[str, Any]) -> None:
        if not idxs:
            return
        trace: Dict[str, Any] = {
            "type": "scatter",
            "mode": "markers",
            "name": f"{label} ({len(idxs)})",
            "x": x[list(idxs)].tolist(),
            "y": y[list(idxs)].tolist(),
            "marker": marker,
            "text": [summary_texts[i] for i in idxs],
            "customdata": [customdata_all[i] for i in idxs],
            "hovertemplate": "%{text}<extra></extra>",
        }
        if highlight_enabled:
            selected_marker, unselected_marker = _highlight_markers(marker.get("size", base_size), marker.get("opacity", base_opacity))
            trace["selected"] = {"marker": selected_marker}
            trace["unselected"] = {"marker": unselected_marker}
        traces.append(trace)

    for label in ["Diadas", "Triadas", "Séptimas", "Extensiones"]:
        idxs = named_groups[label]
        if not idxs:
            continue
        base_marker = dict(
            symbol=named_symbol_map[label],
            size=named_size,
            color=color_values[idxs].tolist(),
            colorscale="Turbo",
            cmin=cmin,
            cmax=cmax,
            coloraxis="coloraxis",
            opacity=named_opacity,
            line=dict(width=0),
        )
        _build_trace(idxs, label=label, marker=base_marker)

    for label in ["3 notas", "4 notas", "5 notas", "Más de 5 notas"]:
        idxs = unnamed_groups[label]
        if not idxs:
            continue
        base_marker = dict(
            symbol=unnamed_symbol_map[label],
            size=base_size,
            color=color_values[idxs].tolist(),
            colorscale="Turbo",
            cmin=cmin,
            cmax=cmax,
            coloraxis="coloraxis",
            opacity=base_opacity,
            line=dict(width=0),
        )
        _build_trace(idxs, label=label, marker=base_marker)

    meta_payload = {
        "familyHighlight": highlight_summary,
        "colorTitle": color_title,
        "isProposal": bool(is_proposal),
    }
    if meta:
        for key, value in meta.items():
            meta_payload[str(key)] = _to_serialisable(value)

    layout = {
        "title": title,
        "width": 640,
        "height": 420,
        "plot_bgcolor": "white",
        "margin": dict(l=40, r=200, t=64, b=42),
        "showlegend": True,
        "legend": dict(
            orientation="v",
            x=1.22,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="#ccc",
            borderwidth=1,
            font=dict(size=11),
        ),
        "xaxis": dict(visible=False),
        "yaxis": dict(visible=False),
        "coloraxis": dict(
            colorscale="Turbo",
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(title=color_title, thickness=14, len=0.75, x=1.08),
        ),
        "meta": meta_payload,
    }

    return {
        "data": traces,
        "layout": layout,
        "meta": layout["meta"],
    }


def build_board_index(
    figures: Sequence[FigureSpec],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Create hierarchical boards grouped by proposal, metric and reduction."""

    boards: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "by_proposal": {},
        "by_metric": {},
        "by_reduction": {},
    }
    for spec in figures:
        payload = spec.serializable()
        boards["by_proposal"].setdefault(spec.proposal, []).append(payload)
        boards["by_metric"].setdefault(spec.metric, []).append(payload)
        boards["by_reduction"].setdefault(spec.reduction, []).append(payload)

    def _sorted(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            entries,
            key=lambda item: (
                item.get("scenario", ""),
                item.get("mode", ""),
                item.get("exponent") if item.get("exponent") is not None else -1,
            ),
        )

    for group in boards.values():
        for key, values in list(group.items()):
            group[key] = _sorted(values)

    return boards
