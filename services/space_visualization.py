"""Renderers centralizados para figuras del espacio (scatter, heatmap, Shepard).

Todas las funciones devuelven plotly.Figure sin escribir archivos.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_scatter(embeddings: np.ndarray, df_pop: pd.DataFrame, *, title: str = "") -> go.Figure:
    """Scatter básico (o embellecido) usando plotly express."""
    if embeddings.shape[1] < 2:
        raise ValueError("Embeddings deben tener al menos 2 columnas para scatter.")
    df = df_pop.copy()
    df["x"] = embeddings[:, 0]
    df["y"] = embeddings[:, 1]
    color_col = "span_semitones" if "span_semitones" in df.columns else None
    symbol_col = None
    if "n" in df.columns:
        symbol_col = df["n"].astype(str)
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_col,
        symbol=symbol_col,
        hover_data=[c for c in ("id", "code", "interval", "n") if c in df.columns],
        title=title or "Scatter",
    )
    fig.update_layout(legend_title_text="n" if symbol_col is not None else None)
    return fig


def build_heatmap(distance_matrix: np.ndarray, *, title: str = "") -> go.Figure:
    """Heatmap de distancias."""
    fig = px.imshow(distance_matrix, color_continuous_scale="Plasma", origin="lower")
    fig.update_layout(title=title or "Matriz de distancias")
    return fig


def build_shepard(embeddings: np.ndarray, distance_matrix: np.ndarray, *, title: str = "") -> go.Figure:
    """Gráfico de Shepard: distancias originales vs distancias embebidas."""
    if embeddings.shape[0] != distance_matrix.shape[0]:
        raise ValueError("Embeddings y matriz de distancias no concuerdan en filas.")
    from sklearn.metrics import pairwise_distances

    emb_dist = pairwise_distances(embeddings)
    # Solo triángulo superior sin diagonal
    iu = np.triu_indices_from(distance_matrix, k=1)
    x = distance_matrix[iu].ravel()
    y = emb_dist[iu].ravel()
    fig = px.scatter(
        x=x,
        y=y,
        labels={"x": "Distancia original", "y": "Distancia embebida"},
        title=title or "Gráfico de Shepard",
        opacity=0.6,
    )
    # Ajuste lineal simple
    if len(x) > 1:
        coeffs = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ys = coeffs[0] * xs + coeffs[1]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="OLS"))
    return fig


__all__ = ["build_scatter", "build_heatmap", "build_shepard"]
