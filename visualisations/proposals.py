"""Utilities for building Plotly visualisations used across proposal comparisons."""
from __future__ import annotations

import copy
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


def _build_filter_metadata(
    entries: Sequence["ChordEntry"],
    family_tags: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    """Return filter definitions and per-entry field values."""

    cardinalities = [int(getattr(entry, "n_notes", 0) or 0) for entry in entries]
    unique_cardinalities = sorted(set(cardinalities))

    def _interval_sequence(entry: "ChordEntry") -> Tuple[int, ...]:
        raw = getattr(getattr(entry, "acorde", None), "intervals", []) or []
        return tuple(int(v) for v in raw)

    def _pattern_id(pattern: Tuple[int, ...]) -> str:
        if not pattern:
            return "none"
        return "-".join(str(v) for v in pattern)

    def _pattern_label(pattern: Tuple[int, ...]) -> str:
        if not pattern:
            return "Sin intervalos"
        return "[" + ", ".join(str(v) for v in pattern) + "]"

    interval_sequences = [_interval_sequence(entry) for entry in entries]
    max_internal_values: List[int] = []
    pattern_registry: Dict[str, Dict[str, Any]] = {}
    interval_values: List[str] = []
    for pattern in interval_sequences:
        pid = _pattern_id(pattern)
        interval_values.append(pid)
        info = pattern_registry.setdefault(
            pid,
            {
                "pattern": pattern,
                "label": _pattern_label(pattern),
            },
        )
        info["count"] = info.get("count", 0) + 1
        try:
            max_internal = max(int(step) for step in pattern) if pattern else 0
        except ValueError:
            max_internal = 0
        max_internal_values.append(max_internal)

    def _pitch_classes(entry: "ChordEntry") -> List[int]:
        acorde = getattr(entry, "acorde", None)
        chroma = getattr(acorde, "chroma", None)
        if chroma is not None and len(chroma):
            try:
                active = [int(idx) for idx, val in enumerate(chroma) if val]
                if active:
                    return sorted({pc % 12 for pc in active})
            except Exception:  # pragma: no cover - defensive
                pass
        intervals = getattr(acorde, "intervals", []) or []
        total = 0
        pcs = {0}
        for step in intervals:
            total = (total + int(step)) % 12
            pcs.add(total)
        return sorted(pcs)

    pitch_class_values: List[List[str]] = []
    available_pitch_classes: set[int] = set()
    for entry in entries:
        classes = _pitch_classes(entry)
        for pc in classes:
            available_pitch_classes.add(int(pc))
        pitch_class_values.append([str(int(pc)) for pc in classes])

    filter_definitions: Dict[str, Any] = {
        "cardinality": {
            "label": "Número de notas",
            "options": [
                {
                    "id": str(card),
                    "label": f"{card} nota{'s' if card != 1 else ''}",
                    "default": True,
                }
                for card in unique_cardinalities
            ],
        },
    }

    if max_internal_values:
        max_allowed = int(max(max_internal_values))
        min_allowed = int(min(max_internal_values))
        filter_definitions["maxInternalInterval"] = {
            "label": "Máximo intervalo interno (semitonos)",
            "type": "numeric",
            "min": min_allowed,
            "max": max_allowed,
            "default": max_allowed,
            "step": 1,
        }

    if pattern_registry:
        options = [
            {
                "id": pid,
                "label": data["label"],
                "default": True,
                "count": int(data.get("count", 0)),
                "cardinality": len(data["pattern"]) + 1,
                "patternLength": len(data["pattern"]),
            }
            for pid, data in sorted(
                pattern_registry.items(),
                key=lambda item: (len(item[1]["pattern"]), item[0]),
            )
        ]
        filter_definitions["intervalPattern"] = {
            "label": "Patrones de intervalos",
            "options": options,
        }

    if available_pitch_classes:
        filter_definitions["pitchClass"] = {
            "label": "Pitch classes presentes",
            "options": [
                {
                    "id": str(pc),
                    "label": f"PC {pc}",
                    "default": True,
                }
                for pc in sorted(available_pitch_classes)
            ],
        }

    field_values: Dict[str, List[Any]] = {
        "cardinality": cardinalities,
        "intervalPattern": interval_values,
        "pitchClass": pitch_class_values,
        "maxInternalInterval": max_internal_values,
        "family": list(family_tags),
    }
    return filter_definitions, field_values


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

    filter_definitions, filter_fields = _build_filter_metadata(entries, family_tags)
    trace_sources: List[Dict[str, Any]] = []

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

        idx_list = list(idxs)
        base_marker = {k: _to_serialisable(v) for k, v in marker.items() if k != "color"}
        trace_sources.append(
            {
                "traceIndex": len(traces) - 1,
                "label": label,
                "x": [float(x[idx]) for idx in idx_list],
                "y": [float(y[idx]) for idx in idx_list],
                "text": [_to_serialisable(summary_texts[idx]) for idx in idx_list],
                "customdata": [_to_serialisable(customdata_all[idx]) for idx in idx_list],
                "colors": _to_serialisable(color_values[idx_list].tolist()),
                "baseMarker": base_marker,
                "hovertemplate": trace.get("hovertemplate", "%{text}<extra></extra>"),
                "filterValues": {
                    key: [_to_serialisable(filter_fields[key][idx]) for idx in idx_list]
                    for key in filter_fields
                },
            }
        )

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
        "filters": filter_definitions,
    }
    if meta:
        for key, value in meta.items():
            meta_payload[str(key)] = _to_serialisable(value)

    filter_dataset = {
        "traceSources": trace_sources,
        "fields": filter_definitions,
    }
    meta_payload["filterDataset"] = filter_dataset

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


def _normalise_filter_values(values: Optional[Sequence[Any]]) -> Optional[set[str]]:
    if values is None:
        return None
    normalised: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            normalised.add(str(int(value)))
        else:
            normalised.add(str(value))
    return normalised


def apply_scatter_filters(
    payload: Dict[str, Any],
    *,
    cardinality: Optional[Sequence[int]] = None,
    interval_pattern: Optional[Sequence[str]] = None,
    pitch_class: Optional[Sequence[int]] = None,
    max_internal_interval: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a filtered copy of a scatter payload.

    Parameters
    ----------
    payload:
        Scatter payload generated by :func:`build_scatter_payload`.
    cardinality:
        Allowed numbers of notes.
    interval_pattern:
        Allowed interval patterns encoded as ``"-".join(intervals)`` strings.
    pitch_class:
        Allowed pitch classes. If provided, a chord must include at least one of the
        selected pitch classes to remain visible.
    max_internal_interval:
        Upper bound (inclusive) for the largest adjacent interval inside a chord.
    """

    dataset = payload.get("meta", {}).get("filterDataset")
    if not dataset:
        raise ValueError("Payload does not contain filter metadata")

    cardinality_set = _normalise_filter_values(cardinality)
    interval_set = _normalise_filter_values(interval_pattern)
    pitch_class_set = _normalise_filter_values(pitch_class)
    max_internal_value = None
    if max_internal_interval is not None:
        try:
            max_internal_value = int(max_internal_interval)
        except (TypeError, ValueError):
            max_internal_value = None

    filtered_traces: List[Dict[str, Any]] = []
    original_traces = payload.get("data", [])

    for source in dataset.get("traceSources", []):
        trace_index = int(source.get("traceIndex", 0))
        if trace_index >= len(original_traces):
            continue
        base_trace = copy.deepcopy(original_traces[trace_index])

        x_source = list(source.get("x", []))
        y_source = list(source.get("y", []))
        text_source = list(source.get("text", []))
        custom_source = list(source.get("customdata", []))
        color_source = list(source.get("colors", []))
        filter_values = source.get("filterValues", {})

        cardinality_values = [str(v) for v in filter_values.get("cardinality", [])]
        interval_values = [str(v) for v in filter_values.get("intervalPattern", [])]
        pitch_values_raw = filter_values.get("pitchClass", [])
        max_internal_raw = filter_values.get("maxInternalInterval", [])

        selected_indices: List[int] = []
        for idx in range(len(x_source)):
            if cardinality_set and cardinality_values[idx] not in cardinality_set:
                continue
            if interval_set and interval_values[idx] not in interval_set:
                continue
            if pitch_class_set:
                pitch_values = pitch_values_raw[idx]
                if isinstance(pitch_values, (list, tuple, set)):
                    pitch_as_set = {str(v) for v in pitch_values if v is not None}
                else:
                    pitch_as_set = {str(pitch_values)} if pitch_values is not None else set()
                if not (pitch_as_set & pitch_class_set):
                    continue
            if max_internal_value is not None:
                try:
                    current_max = max_internal_raw[idx]
                    if isinstance(current_max, (list, tuple)):
                        current_max = current_max[0] if current_max else None
                    current_val = int(current_max)
                except (ValueError, TypeError, IndexError):
                    current_val = None
                if current_val is None or current_val > max_internal_value:
                    continue
            selected_indices.append(idx)

        def _select(values: List[Any]) -> List[Any]:
            return [values[idx] for idx in selected_indices]

        base_trace["x"] = _select(x_source)
        base_trace["y"] = _select(y_source)
        base_trace["text"] = _select(text_source)
        base_trace["customdata"] = _select(custom_source)

        marker_base = dict(source.get("baseMarker", {}))
        marker_base["color"] = _select(color_source)
        if source.get("hovertemplate"):
            base_trace["hovertemplate"] = source["hovertemplate"]
        base_trace["marker"] = marker_base
        filtered_traces.append(base_trace)

    new_payload = copy.deepcopy(payload)
    new_payload["data"] = filtered_traces
    meta = new_payload.get("meta", {})
    meta["activeFilters"] = {
        "cardinality": sorted(cardinality_set) if cardinality_set else None,
        "intervalPattern": sorted(interval_set) if interval_set else None,
        "pitchClass": sorted(pitch_class_set) if pitch_class_set else None,
        "maxInternalInterval": max_internal_value,
    }
    new_payload["meta"] = meta
    if "layout" in new_payload:
        layout_meta = new_payload["layout"].get("meta", {})
        layout_meta.update(meta)
        new_payload["layout"]["meta"] = layout_meta
    return new_payload


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
