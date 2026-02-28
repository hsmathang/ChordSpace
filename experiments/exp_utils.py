"""
exp_utils.py
============
Utilidades compartidas para experimentos de ChordSpace.

Define la función canónica `run_experiment()` que:
  1. Toma una lista de ChordEntry ya construidos y sus vectores 12-D
  2. Calcula distancias (Euclidiana, Coseno, o cualquier otra soportada)
  3. Corre MDS
  4. Etiquetas de nombre encima de cada punto (show_labels=True)
  5. Layout de calidad paper (paper_quality=True)
  4. Calcula el conjunto COMPLETO de métricas del proyecto
     (trustworthiness, continuity, knn_recall, stress, shepard_r2,
      silhouette, davies_bouldin, relative_rank_error, var_ratio,
      cardinality_logreg_acc, knn_hit_card_N, ...)
  5. Genera y guarda las figuras exactas del report.html
     (build_scatter_figure, build_heatmap_figure, build_shepard_figure)
  6. Guarda metricas.txt en la carpeta de salida

Uso mínimo:
    from exp_utils import build_entries, run_experiment

    entries, X = build_entries(chords_raw)           # chords_raw = lista de (name, intervals, freqs, notes_abs)
    run_experiment(entries, X, "euclidean", output_dir)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import umap

from pre_process import Acorde, ModeloSetharesVec, get_chord_type_from_intervals
from tools.proposals_pipeline.population import (
    ChordEntry,
    compute_interval_counts,
    determine_dyad_bin,
)
from tools.proposals_pipeline.figures import (
    build_scatter_figure,
    build_heatmap_figure,
    build_shepard_figure,
    HighlightSettings,
)
from tools.proposals_pipeline.metrics import (
    summarise_embedding_metrics,
)

# ── Constantes ──────────────────────────────────────────────────────────────
LABEL_FONT_SIZE   = 8    # pt — tamaño fuente etiquetas sobre puntos
LABEL_FONT_FAMILY = "Arial"
PAPER_FONT_SIZE   = 13   # pt — fuente base para calidad paper
PAPER_WIDTH       = 900  # px
PAPER_HEIGHT      = 700  # px
NOTE_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")

DEFAULT_HIGHLIGHT = HighlightSettings(
    threshold=500,
    size_scale=1.4,
    size_delta=4.0,
    selected_opacity=0.95,
    fade_factor=0.25,
)


# ── Helpers visualización ────────────────────────────────────────────────────
def _add_labels_to_scatter(
    fig: go.Figure,
    embedding: np.ndarray,
    entries: List[ChordEntry],
    font_size: int = LABEL_FONT_SIZE,
) -> go.Figure:
    """
    Añade etiquetas de nombre encima de cada punto del scatter MDS.
    Usa un trace Scatter separado en modo 'text' para no interferir
    con el trace principal.
    """
    labels = [e.identity_name for e in entries]
    fig.add_trace(go.Scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        mode="text",
        text=labels,
        textposition="top center",
        textfont=dict(
            size=font_size,
            family=LABEL_FONT_FAMILY,
            color="#111111",
        ),
        showlegend=False,
        hoverinfo="skip",
        name="",
    ))
    return fig


def _paper_quality_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """
    Aplica estilo de calidad paper al scatter MDS:
    - Fondo blanco puro
    - Fuente más grande
    - Sin cuadrícula
    - Márgenes compactos
    - Ejes ocultos (sólo frame)
    """
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=PAPER_FONT_SIZE + 2, family="Arial", color="#222"),
            x=0.0, xanchor="left",
        ),
        font=dict(family="Arial", size=PAPER_FONT_SIZE, color="#222"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=PAPER_WIDTH,
        height=PAPER_HEIGHT,
        margin=dict(l=30, r=140, t=50, b=30),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            showline=False, visible=False,
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            showline=False, visible=False,
        ),
        legend=dict(
            font=dict(size=PAPER_FONT_SIZE - 1),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc",
            borderwidth=1,
        ),
    )
    return fig


def _build_inset_zoom_scatter(
    fig_main: go.Figure,
    embedding: np.ndarray,
    entries: List[ChordEntry],
    *,
    zoom_center_pct: float = 50,   # percentil central para detectar región densa
    zoom_half_pct: float = 22,     # radio en percentiles (±22 → cubre ~44% central)
    zoom_window: Optional[List[float]] = None, # [x0, x1, y0, y1] exactos
    title: str = "",
) -> go.Figure:
    """
    Vista main+zoom de dos paneles:
      - Izquierda: scatter completo con caja roja marcando la región ampliada
      - Derecha:   misma figura pero con ejes limitados a esa región

    Para volver al scatter normal: scatter_mode="normal" en run_experiment().
    """
    import copy
    from plotly.subplots import make_subplots

    x = embedding[:, 0]
    y = embedding[:, 1]

    if zoom_window and len(zoom_window) == 4:
        x0, x1, y0, y1 = zoom_window
    else:
        # Región de zoom: percentiles centrados
        lo_pct = zoom_center_pct - zoom_half_pct
        hi_pct = zoom_center_pct + zoom_half_pct
        x0, x1 = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
        y0, y1 = np.percentile(y, lo_pct), np.percentile(y, hi_pct)
        # Añadir un margen del 10% a la región de zoom
        dx, dy = (x1 - x0) * 0.12, (y1 - y0) * 0.12
        x0 -= dx; x1 += dx; y0 -= dy; y1 += dy

    # Crear figura de 2 columnas
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.60, 0.40],
        subplot_titles=["Vista completa", "Zoom región central"],
        horizontal_spacing=0.04,
    )

    # Copiar traces del scatter original a ambos paneles
    for trace in fig_main.data:
        t1 = copy.deepcopy(trace)
        t2 = copy.deepcopy(trace)
        # Solo el panel izquierdo muestra leyenda
        if hasattr(t2, 'showlegend'):
            t2.showlegend = False
        fig.add_trace(t1, row=1, col=1)
        fig.add_trace(t2, row=1, col=2)

    # ── Preservar coloraxis del scatter original (Turbo, rangos, etc.) ────
    orig_layout = fig_main.layout.to_plotly_json()
    coloraxis_updates = {k: v for k, v in orig_layout.items() if k.startswith("coloraxis")}
    if coloraxis_updates:
        fig.update_layout(**coloraxis_updates)

    # Caja roja en el panel de vista completa
    fig.add_shape(
        type="rect",
        x0=x0, y0=y0, x1=x1, y1=y1,
        line=dict(color="rgba(200,0,0,0.8)", width=2, dash="dash"),
        fillcolor="rgba(220,0,0,0.04)",
        xref="x", yref="y",
    )

    # Limitar ejes del panel derecho (zoom)
    fig.update_xaxes(range=[x0, x1], showgrid=False, zeroline=False,
                     showticklabels=False, showline=True,
                     linecolor="rgba(200,0,0,0.6)", linewidth=2, row=1, col=2)
    fig.update_yaxes(range=[y0, y1], showgrid=False, zeroline=False,
                     showticklabels=False, showline=True,
                     linecolor="rgba(200,0,0,0.6)", linewidth=2, row=1, col=2)

    # Ocultar ejes del panel izquierdo
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False,
                     showline=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                     showline=False, row=1, col=1)

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=PAPER_FONT_SIZE + 2, family="Arial", color="#222"),
            x=0.0, xanchor="left",
        ),
        font=dict(family="Arial", size=PAPER_FONT_SIZE, color="#222"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=int(PAPER_WIDTH * 1.55),   # más ancho para 2 paneles
        height=PAPER_HEIGHT,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            font=dict(size=PAPER_FONT_SIZE - 1),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc",
            borderwidth=1,
        ),
    )
    return fig


def _build_faceted_scatter(
    fig_main: go.Figure,
    embedding: np.ndarray,
    entries: List[ChordEntry],
    facets: List[str],
    facet_symbols: Optional[List[str]] = None,
    facet_opacities: Optional[List[float]] = None,
    font_size: int = LABEL_FONT_SIZE,
    title: str = "",
    layout_style: str = "grid", # "grid", "overview_top", "single_overview"
    show_labels: bool = True,
) -> go.Figure:
    """
    Crea una figura "small multiples" separando los acordes según su categoría.
    - layout="grid": matriz normal dinámica.
    - layout="overview_top": matriz con panel ancho arriba (Vista general) y 
      los desgloses debajo.
    - layout="single_overview": un solo panel grande con todos los puntos, 
      conservando los símbolos por categoría y la leyenda.
    """
    import math
    from plotly.subplots import make_subplots

    unique_facets = []
    for f in facets:
        if f not in unique_facets:
            unique_facets.append(f)

    n_facets = len(unique_facets)
    
    if layout_style == "overview_top":
        cols = 3
        rows = math.ceil((n_facets + 1) / cols)
        specs = None
        subplot_titles = ["Vista General"] + unique_facets
    elif layout_style == "single_overview":
        cols = 1
        rows = 1
        specs = None
        subplot_titles = [""] # Sin subtítulo, el título principal basta
    else:
        cols = min(4, max(1, n_facets))
        rows = math.ceil(n_facets / cols)
        specs = None
        subplot_titles = unique_facets

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        specs=specs,
        horizontal_spacing=0.04,
        vertical_spacing=0.10,
    )

    x = embedding[:, 0]
    y = embedding[:, 1]
    
    dx = (x.max() - x.min()) * 0.05
    dy = (y.max() - y.min()) * 0.05
    x_range = [x.min() - dx, x.max() + dx]
    y_range = [y.min() - dy, y.max() + dy]

    all_bruta = []
    all_p05 = []
    all_p10 = []
    all_t05 = []
    all_t10 = []
    
    num_traces = 0

    if layout_style != "single_overview":
        for i, facet in enumerate(unique_facets):
            if layout_style == "overview_top":
                idx_panel = i + 1  # panel 0 es el overview
            else:
                idx_panel = i
            
            # Cálculo directo (1-indexed en Plotly)
            r = (idx_panel // cols) + 1
            c = (idx_panel % cols) + 1

            # Fondo gris (todos)
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='markers',
                marker=dict(color='#e0e0e0', size=10, symbol='circle'),
                showlegend=False, hoverinfo='skip'
            ), row=r, col=c)
            num_traces += 1
            
            all_bruta.append('#e0e0e0')
            all_p05.append('#e0e0e0')
            all_p10.append('#e0e0e0')
            all_t05.append('#e0e0e0')
            all_t10.append('#e0e0e0')

            # Puntos activos del facet
            idx = [j for j, f in enumerate(facets) if f == facet]
            
            c_bruta, c_p05, c_p10, c_t05, c_t10 = [], [], [], [], []
            for j in idx:
                total = entries[j].total
                n_notas = entries[j].n_notes
                pares = max(1.0, float(n_notas * (n_notas - 1) / 2))
                tipos = max(1.0, float(np.count_nonzero(entries[j].hist)))
                
                c_bruta.append(total)
                c_p05.append(total / (pares ** 0.5))
                c_p10.append(total / pares)
                c_t05.append(total / (tipos ** 0.5))
                c_t10.append(total / tipos)
                
            all_bruta.append(c_bruta)
            all_p05.append(c_p05)
            all_p10.append(c_p10)
            all_t05.append(c_t05)
            all_t10.append(c_t10)
            
            totals = c_bruta # defecto al graficar
            labels = [entries[j].identity_name for j in idx]
            hovertexts = [
                f"<b>{entries[j].identity_name}</b><br>"
                f"Int: {entries[j].acorde.intervals}<br>"
                f"Total R: {entries[j].total:.2f}<br>"
                f"Vec: [{', '.join(f'{v:.2f}' for v in entries[j].hist)}]"
                for j in idx
            ]
            syms = [facet_symbols[j] for j in idx] if facet_symbols else ['circle']*len(idx)
            opacs = [facet_opacities[j] for j in idx] if facet_opacities else [1.0]*len(idx)
            
            mode = 'markers+text' if show_labels else 'markers'
            
            sizes = [24 if s != "circle" else 10 for s in syms]
            lines = [3.5 if s != "circle" else 0 for s in syms]
            
            fig.add_trace(go.Scatter(
                x=x[idx], y=y[idx], mode=mode,
                text=labels if show_labels else None, textposition="top center",
                hovertext=hovertexts, hoverinfo="text+name",
                textfont=dict(size=font_size, family=LABEL_FONT_FAMILY, color="#555"),
                marker=dict(
                    color=totals, coloraxis="coloraxis", size=sizes,
                    symbol=syms, line=dict(width=lines), opacity=opacs
                ),
                showlegend=False, name=facet,
            ), row=r, col=c)
            num_traces += 1

    # ── Panel ancho al principio (Overview con Leyenda) ──
    if layout_style in ["overview_top", "single_overview"]:
        r_all = 1
        c_all = 1
        for idx_facet, facet in enumerate(unique_facets):
            idx = [j for j, f in enumerate(facets) if f == facet]
            if not idx: continue
            
            c_bruta, c_p05, c_p10, c_t05, c_t10 = [], [], [], [], []
            for j in idx:
                total = entries[j].total
                n_notas = entries[j].n_notes
                pares = max(1.0, float(n_notas * (n_notas - 1) / 2))
                tipos = max(1.0, float(np.count_nonzero(entries[j].hist)))
                c_bruta.append(total)
                c_p05.append(total / (pares ** 0.5))
                c_p10.append(total / pares)
                c_t05.append(total / (tipos ** 0.5))
                c_t10.append(total / tipos)
                
            all_bruta.append(c_bruta)
            all_p05.append(c_p05)
            all_p10.append(c_p10)
            all_t05.append(c_t05)
            all_t10.append(c_t10)
            
            totals = c_bruta
            labels = [entries[j].identity_name for j in idx]
            hovertexts = [
                f"<b>{entries[j].identity_name}</b><br>"
                f"Int: {entries[j].acorde.intervals}<br>"
                f"Total R: {entries[j].total:.2f}<br>"
                f"Vec: [{', '.join(f'{v:.2f}' for v in entries[j].hist)}]"
                for j in idx
            ]
            single_sym = facet_symbols[idx[0]] if facet_symbols else 'circle'
            single_opac = facet_opacities[idx[0]] if facet_opacities else 1.0
            
            mode = 'markers+text' if show_labels else 'markers'
            
            single_size = 24 if single_sym != "circle" else 10
            single_line = 3.5 if single_sym != "circle" else 0
            
            fig.add_trace(go.Scatter(
                x=x[idx], y=y[idx], mode=mode,
                text=labels if show_labels else None, textposition="top center",
                hovertext=hovertexts, hoverinfo="text+name",
                textfont=dict(size=font_size, family=LABEL_FONT_FAMILY, color="#555"),
                marker=dict(
                    color=totals, coloraxis="coloraxis", size=single_size,
                    symbol=single_sym, line=dict(width=single_line), opacity=single_opac
                ),
                showlegend=True, name=facet,
            ), row=r_all, col=c_all)
            num_traces += 1

    orig_layout = fig_main.layout.to_plotly_json()
    coloraxis_updates = {k: v for k, v in orig_layout.items() if k.startswith("coloraxis")}
    
    if layout_style == "single_overview":
        w = int(PAPER_WIDTH * 0.8) + 120
        h = int(PAPER_HEIGHT * 0.8)
    else:
        w = int(PAPER_WIDTH * max(1, cols * 0.35)) + 120
        h = int(PAPER_HEIGHT * max(1, rows * 0.40))
        
    def get_max(arrays):
        m = 0
        for arr in arrays:
            if isinstance(arr, list) and len(arr) > 0:
                m = max(m, max(arr))
        return m if m > 0 else 1

    cmax_bruta = get_max(all_bruta)
    cmax_p05 = get_max(all_p05)
    cmax_p10 = get_max(all_p10)
    cmax_t05 = get_max(all_t05)
    cmax_t10 = get_max(all_t10)
    
    updatemenus = [
        dict(
            buttons=list([
                dict(
                    args=[{"marker.color": all_bruta}, {"coloraxis.cmax": cmax_bruta}],
                    label="Rugosidad Bruta",
                    method="update"
                ),
                dict(
                    args=[{"marker.color": all_p05}, {"coloraxis.cmax": cmax_p05}],
                    label="Por Pares (Exp: 0.50)",
                    method="update"
                ),
                dict(
                    args=[{"marker.color": all_p10}, {"coloraxis.cmax": cmax_p10}],
                    label="Por Pares (Exp: 1.00)",
                    method="update"
                ),
                dict(
                    args=[{"marker.color": all_t05}, {"coloraxis.cmax": cmax_t05}],
                    label="Por Tipos (Exp: 0.50)",
                    method="update"
                ),
                dict(
                    args=[{"marker.color": all_t10}, {"coloraxis.cmax": cmax_t10}],
                    label="Por Tipos (Exp: 1.00)",
                    method="update"
                )
            ]),
            direction="down", pad={"r": 10, "t": 10}, showactive=True,
            x=0.01, xanchor="left", y=1.08, yanchor="top",
            font=dict(family="Arial", size=11, color="#333"),
            bgcolor="white", bordercolor="#ccc"
        ),
    ]

    fig.update_layout(
        title=dict(
            text=title, font=dict(size=PAPER_FONT_SIZE + 2, family="Arial", color="#222"),
            x=0.0, xanchor="left",
        ),
        updatemenus=updatemenus,
        font=dict(family="Arial", size=PAPER_FONT_SIZE, color="#222"),
        paper_bgcolor="white", plot_bgcolor="#FBFBFB" if layout_style != "single_overview" else "white",
        width=w, 
        height=h,
        margin=dict(l=30, r=120, t=150, b=30),  # Margen superior expandido para la nueva altura de la leyenda
        legend=dict(
            title="Grupos",
            font=dict(size=PAPER_FONT_SIZE),
            itemsizing='constant',
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1,
            x=1.03, y=1.45,  # Movido ~100px izquierda y ~320px arriba (proyectado en escala)
            xanchor="left",
            yanchor="top"
        ),
        **coloraxis_updates
    )
    
    # Bordes sutiles a todas las celdas
    for i in range(1, (rows * cols) + 1 if specs is None else len(unique_facets) + 2):
        try:
            fig.update_xaxes(
                range=x_range, showgrid=False, zeroline=False, showticklabels=False,
                showline=True, linecolor="#DDDDDD", linewidth=1, mirror=True,
                row=(i-1)//cols+1, col=(i-1)%cols+1 if specs is None else None
            )
            fig.update_yaxes(
                range=y_range, showgrid=False, zeroline=False, showticklabels=False,
                showline=True, linecolor="#DDDDDD", linewidth=1, mirror=True,
                row=(i-1)//cols+1, col=(i-1)%cols+1 if specs is None else None
            )
        except:
            pass
            
    # Forzamos los ejes para toda la figura en caso de subplots irregulares
    fig.update_xaxes(range=x_range, showgrid=False, zeroline=False, showticklabels=False, showline=True, linecolor="#DDDDDD", linewidth=1, mirror=True)
    fig.update_yaxes(range=y_range, showgrid=False, zeroline=False, showticklabels=False, showline=True, linecolor="#DDDDDD", linewidth=1, mirror=True)
    
    return fig


# ── Helpers generación de acordes ─────────────────────────────────────────────
def midi_nombre(m: int) -> str:
    return f"{NOTE_NAMES[m % 12]}{(m // 12) - 1}"


def chords_from_midi(combos: List[Tuple]) -> List[Tuple[str, List[int], List[float], List[int]]]:
    """
    Convierte una lista de tuplas MIDI en (name, intervals, freqs, notes_abs).
    Cada elemento de combos puede ser una tupla de enteros MIDI.
    """
    result = []
    for notas in combos:
        notas = sorted(notas)
        name = "-".join(midi_nombre(n) for n in notas)
        intervals = [notas[i+1] - notas[i] for i in range(len(notas)-1)]
        freqs = [440.0 * 2**((m - 69) / 12.0) for m in notas]
        result.append((name, intervals, freqs, list(notas)))
    return result


def build_entries(
    chords_raw: List[Tuple[str, List[int], List[float], List[int]]],
    *,
    modelo: Optional[ModeloSetharesVec] = None,
    proposal: str = "perclass_alpha0_75",
) -> Tuple[List[ChordEntry], np.ndarray]:
    """
    Construye ChordEntry objects y la matriz X (N×12) de vectores 12-D.

    Args:
        chords_raw: lista de (name, intervals, freqs, notes_abs)
        modelo: ModeloSetharesVec a usar (por defecto crea uno con config canónica)

    Returns:
        entries: lista de ChordEntry
        X: np.ndarray (N, 12)
    """
    if modelo is None:
        modelo = ModeloSetharesVec(config={})

    entries: List[ChordEntry] = []
    vectores: List[np.ndarray] = []

    for name, intervals, freqs, notes_abs in chords_raw:
        acorde = Acorde(name=name, intervals=intervals, frequencies=freqs)
        acorde.notes_abs = list(notes_abs)

        identity_obj = get_chord_type_from_intervals(intervals, with_alias=True)
        identity_name = getattr(identity_obj, "name", str(identity_obj)) or name
        identity_aliases = tuple(getattr(identity_obj, "aliases", ()))
        is_named = bool(identity_name and identity_name not in ("Unknown", ""))

        hist, total = modelo.calcular(acorde)
        hist = np.asarray(hist, dtype=float)

        counts = compute_interval_counts(intervals)
        total_pairs = max(float(np.sum(counts)), 1.0)
        n_notes = len(intervals) + 1

        entry = ChordEntry(
            acorde=acorde,
            hist=hist,
            total=float(total),
            counts=counts,
            total_pairs=total_pairs,
            n_notes=n_notes,
            dyad_bin=determine_dyad_bin(intervals) if n_notes == 2 else None,
            identity_name=name,
            identity_aliases=identity_aliases,
            is_named=is_named,
        )
        entries.append(entry)
        vectores.append(hist)

    X = np.stack(vectores)
    
    if proposal == "perclass_alpha0_75":
        counts_matrix = np.stack([e.counts for e in entries])
        adjusted = X.copy()
        for i in range(adjusted.shape[0]):
            divisor = np.power(np.clip(counts_matrix[i], 1.0, None), 0.75)
            adjusted[i] = adjusted[i] / divisor
        X = np.clip(adjusted, 0.0, None)
        
    return entries, X


def run_experiment(
    entries: List[ChordEntry],
    X: np.ndarray,
    metric: str,
    output_dir: Path,
    *,
    experiment_name: str = "experimento",
    scatter_title: Optional[str] = None,
    n_init: int = 8,
    random_state: int = 42,
    highlight: Optional[HighlightSettings] = None,
    heatmap_max_n: int = 80,
    show_labels: bool = True,
    label_font_size: int = LABEL_FONT_SIZE,
    paper_quality: bool = True,
    scatter_mode: str = "normal",  # "normal" | "inset" | "faceted"
    facet_labels: Optional[List[str]] = None,
    facet_symbols: Optional[List[str]] = None, # iconos de los puntos (ej 'triangle-up')
    facet_opacities: Optional[List[float]] = None, # opacidad por punto
    facet_layout: str = "grid",    # "grid" | "overview_top" | "single_overview"
    zoom_center_pct: float = 50,   # solo en scatter_mode="inset"
    zoom_half_pct: float = 22,     # solo en scatter_mode="inset"
    zoom_window: Optional[List[float]] = None, # solo en "inset": [x0, x1, y0, y1]
    reducer: str = "mds",          # "mds" | "umap"
    verbose: bool = True,
) -> dict:
    """
    Pipeline completo de experimento MDS para un conjunto de acordes.

    Args:
        entries        : lista de ChordEntry
        X              : matriz (N, 12) de vectores rugosidad 12-D
        metric         : "euclidean" | "cosine" | cualquier métrica de scipy.pdist
        output_dir     : carpeta de salida (se crea si no existe)
        experiment_name: prefijo para los archivos de salida
        n_init         : número de inicializaciones MDS
        random_state   : semilla aleatoria
        highlight      : configuración de highlight para scatter
        heatmap_max_n  : máximo de acordes a mostrar en el heatmap
        show_labels    : muestra el nombre de cada acorde encima del punto
        label_font_size: tamaño de fuente de las etiquetas (pt)
        paper_quality  : aplica estilo publicación (sin ejes, fondo blanco, fuente grande)
        scatter_mode   : "normal" | "inset" (main+zoom dual) | "faceted" (paneles por categoría)
        facet_labels   : lista alineada con entries para agrupar por categoría (requerido si mode="faceted")
        facet_symbols  : (opcional) iconos (square, circle, cross, etc) para las facetas
        facet_layout   : "grid" matrix o "overview_top" con resumen grande arriba
        zoom_center_pct: percentil central de la región de zoom (default 50 = centro)
        zoom_half_pct  : radio en percentiles de la región de zoom (default ±22)
        zoom_window    : [x0, x1, y0, y1] coordenadas exactas para la región del zoom
        verbose        : imprimir progreso
        dict con todas las métricas calculadas
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if highlight is None:
        highlight = DEFAULT_HIGHLIGHT

    N = len(entries)
    if verbose:
        print(f"[EXP] '{experiment_name}'  N={N}  métrica={metric}")

    # ── Distancias ────────────────────────────────────────────────────────
    dist_condensed = pdist(X, metric=metric)
    dist_matrix    = squareform(dist_condensed)

    # ── Reducción de Dimensionalidad ──────────────────────────────────────
    if reducer.lower() == "umap":
        if verbose:
            print(f"[EXP] UMAP (n_neighbors=15, min_dist=0.1)...")
        reducer_obj = umap.UMAP(
            n_neighbors=min(15, max(2, N - 1)),
            min_dist=0.1,
            metric=metric,
            random_state=random_state,
            init="spectral"
        )
        embedding = reducer_obj.fit_transform(X)
    else:
        if verbose:
            print(f"[EXP] MDS (n_init={n_init})...")
        reducer_obj = MDS(
            n_components=2,
            metric=True,
            dissimilarity="precomputed",
            random_state=random_state,
            max_iter=1000,
            n_init=n_init,
        )
        embedding = reducer_obj.fit_transform(dist_matrix)

    # ── Métricas completas (función canónica del proyecto) ─────────────
    labels_card = np.array([e.n_notes for e in entries])
    metrics = summarise_embedding_metrics(
        X_original=X,
        embedding=embedding,
        dist_matrix=dist_matrix,
        dist_condensed=dist_condensed,
        labels=labels_card,
        seed=random_state,
    )

    if verbose:
        print(f"\n[MÉTRICAS] {experiment_name} — {metric}")
        for k, v in metrics.items():
            if v is not None and not k.endswith("_kb"):
                print(f"  {k:35s} = {v:.4f}")

    # ── Exportar métricas y coordenadas ──────────────────────────────────
    lines = [
        f"=== {experiment_name} | métrica={metric} | N={N} ===\n",
        f"ModeloSetharesVec (canónico) → vector 12-D\n\n",
    ]
    for k, v in metrics.items():
        val_str = f"{v:.6f}" if isinstance(v, float) else str(v)
        lines.append(f"{k:35s}: {val_str}\n")

    metricas_path = output_dir / f"metricas_{experiment_name}_{metric}.txt"
    metricas_path.write_text("".join(lines), encoding="utf-8")

    csv_lines = ["index,name,x,y,total_roughness,n_notes\n"]
    for i, e in enumerate(entries):
        csv_lines.append(f"{i},{e.identity_name},{embedding[i,0]:.6f},{embedding[i,1]:.6f},{e.total:.6f},{e.n_notes}\n")
    coords_path = output_dir / f"coords_{experiment_name}_{metric}.csv"
    coords_path.write_text("".join(csv_lines), encoding="utf-8")
    
    if verbose:
        print(f"\n  → {metricas_path}")
        print(f"  → {coords_path}")

    # ── Figuras ──────────────────────────────────────────────────────────
    totals_arr = np.array([e.total for e in entries], dtype=float)
    pairs_arr  = np.array([e.total_pairs for e in entries], dtype=float)

    def _save(fig, fname):
        p = output_dir / f"{fname}.html"
        fig.write_html(str(p), include_plotlyjs="cdn", full_html=True)
        if verbose:
            print(f"  → {p}")
        try:
            fig.write_image(str(output_dir / f"{fname}.png"), width=1200, height=900, scale=2)
        except Exception:
            pass

    # Scatter MDS
    if not scatter_title:
        scatter_title = f"{experiment_name} — MDS {metric}"
    
    fig = build_scatter_figure(
        embedding=embedding,
        entries=entries,
        color_values=totals_arr,
        pair_counts=pairs_arr,
        type_counts=pairs_arr,
        vectors=X,
        adjusted_vectors=X,
        title=scatter_title,
        is_proposal=False,
        color_title="Rugosidad total",
        highlight_settings=highlight,
    )
    if show_labels:
        fig = _add_labels_to_scatter(fig, embedding, entries, font_size=label_font_size)

    if scatter_mode == "inset":
        fig = _build_inset_zoom_scatter(
            fig, embedding, entries,
            zoom_center_pct=zoom_center_pct,
            zoom_half_pct=zoom_half_pct,
            zoom_window=zoom_window,
            title=scatter_title,
        )
        suffix = "_inset"
    elif scatter_mode == "faceted":
        if not facet_labels or len(facet_labels) != N:
            raise ValueError("scatter_mode='faceted' requiere passet_labels del mismo tamaño que entries.")
        fig = _build_faceted_scatter(
            fig, embedding, entries,
            facets=facet_labels,
            facet_symbols=facet_symbols,
            facet_opacities=facet_opacities,
            font_size=label_font_size,
            title=scatter_title,
            layout_style=facet_layout,
            show_labels=show_labels,
        )
        suffix = "_faceted"
    elif paper_quality:
        fig = _paper_quality_layout(fig, title=scatter_title)
        suffix = ""
    else:
        suffix = ""

    _save(fig, f"scatter_{experiment_name}_{metric}{suffix}")

    # Heatmap
    n_heat = min(N, heatmap_max_n)
    fig = build_heatmap_figure(
        dist_matrix=dist_matrix[:n_heat, :n_heat],
        entries=entries[:n_heat],
        title=f"Heatmap {metric} — {experiment_name} (N={n_heat})",
    )
    _save(fig, f"heatmap_{experiment_name}_{metric}")

    # Shepard
    fig = build_shepard_figure(
        dist_original_condensed=dist_condensed,
        embedding=embedding,
        entries=entries,
        title=f"Shepard {metric} — {experiment_name}",
    )
    _save(fig, f"shepard_{experiment_name}_{metric}")

    return metrics
