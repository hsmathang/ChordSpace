"""
visualization.py
----------------
Este módulo se encarga de generar visualizaciones para los resultados del laboratorio de acordes.
Incluye funciones para:

  - Construir un DataFrame de visualización a partir de embeddings, objetos Acorde y vectores originales.
  - Generar gráficos combinados (scatter plots con contornos de densidad).
  - Visualizar heatmaps de matrices de distancia.
  - Generar gráficos de Shepard para comparar distancias originales y embebidas.
  - Explorar el grafo generado por UMAP.
  - Visualizar embeddings obtenidos por Laplacian Eigenmaps.
  - Visualizar la representación Sp–Sf de los acordes.

Se ha refactorizado el código para manejar la rugosidad de forma robusta: se decidió eliminar 'decidir_escala_rugosidad' y 'get_color_column',
y en su lugar se introdujo la función 'transform_rugosidad_for_color', que:
   1) calcula estadísticas de la distribución,
   2) decide la transformación adecuada (log, raíz o lineal) segun los valores,
   3) retorna la columna transformada y la configuración de la barra de color en la escala original (ticks, etc.).
"""

import numpy as np
np.bool8 = np.bool_
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from matplotlib.colors import to_rgb
from scipy.spatial.distance import pdist, squareform
from itertools import combinations


# Importar constantes desde config.py
from config import COMMON_RUG_COLORSCALE, COMMON_COLORBAR, EVAL_N_NEIGHBORS

# Importar funciones desde pre_process.py
from pre_process import get_chord_type_from_intervals, representacion_sp_sf


def sample_colors_for_values(values, cmin: float, cmax: float, colorscale: str = "Viridis") -> list:
    """Map numeric values to actual hex colors using the given colorscale."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    valid = np.isfinite(arr)
    if not valid.any():
        arr = np.zeros_like(arr, dtype=float)
    if np.isclose(cmax, cmin):
        norm = np.zeros_like(arr, dtype=float) + 0.5
    else:
        norm = np.clip((arr - cmin) / (cmax - cmin), 0.0, 1.0)
    return px.colors.sample_colorscale(colorscale, norm.tolist())

# =======================
# Helper Functions
# =======================

def build_chord_dataframe(embeddings: np.ndarray, acordes: list, X_original: np.ndarray) -> pd.DataFrame:
    """
    Construye un DataFrame a partir de los datos de visualizacion de acordes.

    Args:
        embeddings (np.ndarray): Coordenadas embebidas (Nx2).
        acordes (list): Lista de objetos Acorde.
        X_original (np.ndarray): Matriz de vectores originales.

    Returns:
        pd.DataFrame: DataFrame con columnas:
          - X, Y: Coordenadas del embedding.
          - Acorde: Nombre del acorde.
          - Intervalos: Cadena con los intervalos.
          - Tipo: Tipo principal con alias concatenados.
          - TotalRug: Valor de rugosidad total.
          - VectorOriginal: Representacion del vector original.
          - Conocido: "Con Nombre" o "Desconocido" (segun el tipo).
          - hover_text: Texto enriquecido para hover.
    """
    identities = [get_chord_type_from_intervals(ac.intervals, with_alias=True) for ac in acordes]

    tipo_base = [identity.name for identity in identities]
    tipo_display = []
    for identity in identities:
        if identity.name == "Unknown":
            tipo_display.append("Unknown")
            continue
        alias_list = [alias for alias in identity.aliases if alias]
        if alias_list:
            tipo_display.append(", ".join([identity.name] + list(alias_list)))
        else:
            tipo_display.append(identity.name)

    data = {
        "X": embeddings[:, 0],
        "Y": embeddings[:, 1],
        "Acorde": [ac.name for ac in acordes],
        "Intervalos": [str(ac.intervals) for ac in acordes],
        "Tipo": tipo_display,
        "TotalRug": [ac.total_roughness if ac.total_roughness is not None else 0 for ac in acordes],
        "VectorOriginal": [
            "[{head}\n {tail}]".format(
                head=", ".join(f"{float(v):.3f}" for v in vec[:6]),
                tail=", ".join(f"{float(v):.3f}" for v in vec[6:])
            )
            for vec in X_original
        ],
    }
    df = pd.DataFrame(data)
    df["__idx__"] = np.arange(len(df))

    df["Conocido"] = ["Con Nombre" if name != "Unknown" else "Desconocido" for name in tipo_base]
    df["hover_text"] = create_hover_text(df)
    return df



def create_hover_text(df: pd.DataFrame) -> pd.Series:
    """
    Crea la columna 'hover_text' a partir de las columnas relevantes del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame que contiene columnas: Acorde, Intervalos, Tipo, TotalRug, VectorOriginal.

    Returns:
        pd.Series: Serie con el texto de hover formateado.
    """
    hover = (
        "Acorde: " + df["Acorde"] + "<br>" +
        "Intervalos: " + df["Intervalos"] + "<br>" +
        "Tipo: " + df["Tipo"] + "<br>" +
        "TotalRug: " + df["TotalRug"].astype(str) + "<br>" +
        "VectorOriginal: " + df["VectorOriginal"]
    )
    return hover




def calcular_rango_rugosidad(df, columna="TotalRug", lower=1, upper=99):
    """
    Calcula un rango robusto de rugosidad basado en percentiles para ignorar outliers extremos.
    Por defecto, se toma del 1% al 99% para reducir el efecto de valores anómalos.
    """
    min_val = np.percentile(df[columna], lower)
    max_val = np.percentile(df[columna], upper)
    return min_val, max_val


# --------------------------------------------------------------------------
# NUEVA FUNCIÓN: transform_rugosidad_for_color
# --------------------------------------------------------------------------
import numpy as np

def transform_rugosidad_for_color(
    df,
    columna="TotalRug",
    modo_auto=True,
    modo_forzado=None,
    n_ticks=6,
    clip_percentiles=False,
):
    """Transforma la columna de rugosidad y calibra la barra de color.

    Args:
        df (pd.DataFrame): conjunto de datos con la columna de rugosidad.
        columna (str): nombre de la columna que contiene los valores originales.
        modo_auto (bool): si ``True`` decide log/sqrt/lineal segun la dispersión.
        modo_forzado (Optional[str]): forzar 'log', 'sqrt' o 'linear'.
        n_ticks (int): numero de marcas visibles en la barra de color.
        clip_percentiles (bool): si ``True`` usa p1/p99; de lo contrario min/max reales.

    Returns:
        Tuple[pd.Series, float, float, np.ndarray, List[str], str]:
            serie transformada, limites de color, posiciones y textos de ticks, y título.
    """
    valores = df[columna].to_numpy(dtype=float, copy=True)
    if valores.size == 0:
        valores = np.array([0.0])
    mascara_finite = np.isfinite(valores)
    if not mascara_finite.any():
        mascara_finite = np.ones_like(valores, dtype=bool)
    valores_finitos = valores[mascara_finite]

    if clip_percentiles:
        base_min = float(np.percentile(valores_finitos, 1))
        base_max = float(np.percentile(valores_finitos, 99))
    else:
        base_min = float(valores_finitos.min())
        base_max = float(valores_finitos.max())

    if not np.isfinite(base_min):
        base_min = 0.0
    if not np.isfinite(base_max):
        base_max = base_min

    if base_max <= base_min:
        base_max = base_min + 1e-6

    positivos = valores_finitos[valores_finitos > 0]
    if modo_forzado in ("log", "sqrt", "linear"):
        modo = modo_forzado
    elif modo_auto:
        if positivos.size >= 5:
            p5, p95 = np.percentile(positivos, [5, 95])
            ratio = p95 / p5 if p5 > 0 else float("inf")
        else:
            ratio = 1.0
        if ratio > 20:
            modo = "log"
        elif ratio > 5:
            modo = "sqrt"
        else:
            modo = "linear"
    else:
        modo = "linear"

    if modo == "log" and base_max <= 0:
        modo = "linear"

    epsilon = 0.0
    if modo == "log":
        if base_min > 0:
            epsilon = max(1e-12, 0.01 * base_min)
        else:
            minimo_positivo = positivos.min() if positivos.size else max(base_max, 1.0)
            epsilon = max(1e-12, 0.01 * minimo_positivo, 1e-6)
        transformados = np.log10(np.clip(valores, a_min=0, a_max=None) + epsilon)
        cmin = float(np.log10(base_min + epsilon))
        cmax = float(np.log10(base_max + epsilon))
        if cmax <= cmin:
            cmax = cmin + 1e-6
        tvals = np.linspace(cmin, cmax, max(2, n_ticks))
        ttext = [f"{max(0.0, 10**v - epsilon):.2f}" for v in tvals]
        label_bar = "Rugosidad (log10)"
    elif modo == "sqrt":
        transformados = np.sqrt(np.clip(valores, a_min=0, a_max=None))
        cmin = float(np.sqrt(max(base_min, 0.0)))
        cmax = float(np.sqrt(max(base_max, 0.0)))
        if cmax <= cmin:
            cmax = cmin + 1e-6
        tvals = np.linspace(cmin, cmax, max(2, n_ticks))
        ttext = [f"{(val**2):.2f}" for val in tvals]
        label_bar = "Rugosidad (sqrt)"
    else:
        transformados = np.clip(valores, base_min, base_max)
        cmin = base_min
        cmax = base_max
        if cmax <= cmin:
            cmax = cmin + 1e-6
        tvals = np.linspace(cmin, cmax, max(2, n_ticks))
        ttext = [f"{val:.2f}" for val in tvals]
        label_bar = "Rugosidad"

    serie_color = pd.Series(transformados, index=df.index, name="__color__")
    return serie_color, cmin, cmax, tvals, ttext, label_bar
def robust_to_rgb(c) -> tuple:
    """
    Convierte de forma robusta un color a formato RGB normalizado.
    """
    try:
        if isinstance(c, str) and c.startswith("rgb("):
            nums = c[4:-1].split(',')
            rgb = tuple(int(x.strip()) for x in nums)
            return tuple(x / 255 for x in rgb)
        else:
            return to_rgb(c)
    except Exception as e:
        print(f"Error en la conversión del color: {c}. {e}")
        return (0, 0, 0)


# =======================
# Visualización Específica
# =======================

import plotly.express as px
import plotly.graph_objects as go
# from . import transform_rugosidad_for_color  # depende de tu estructura

def visualizar_scatter_density(
    embeddings: np.ndarray,
    acordes: list,
    X_original: np.ndarray,
    title: str = "Percepción y Exploración de Acordes",
    modo_forzado: str = None,
    circle_size: int = 7,
    circle_opacity: float = 0.8
) -> go.Figure:
    """
    Genera un gráfico combinado (contorno de densidad + scatter)
    con fondo blanco, sin ejes y paleta "Turbo".
    """
    df = build_chord_dataframe(embeddings, acordes, X_original)

    # 1) Transformar rugosidad
    color_vals, cmin, cmax, tickvals, ticktext, label_bar = transform_rugosidad_for_color(
        df, "TotalRug", modo_forzado=modo_forzado, n_ticks=6
    )
    df["__color__"] = color_vals

    # 2) Densidad (se mantiene sin cambios visuales relevantes más allá del layout general)
    fig_density = px.density_contour(
        df, x="X", y="Y", nbinsx=30, nbinsy=30 # Titulo se pone en el layout final
    )
    fig_density.update_traces(
        contours_coloring='fill',
        colorscale='Blues', # Mantenemos Blues para la densidad, no afecta 'Turbo' del scatter
        showscale=False,
        opacity=0.3,
        line=dict(width=1)
    )

    # 3) Scatter
    fig_scatter = px.scatter(
        df, x="X", y="Y",
        color="__color__", # Color basado en rugosidad transformada
        # color_continuous_scale="Turbo", # <-- Se define en layout.coloraxis
        hover_name="hover_text",
        hover_data={"X": False, "Y": False, "TotalRug": False, "Conocido": False},
        symbol="Conocido",
        symbol_sequence=["star", "circle"],
        custom_data=["__idx__"]
        # title=title <-- Se define en layout final
    )
    # 4) Ajustes por grupo (sin cambios)
    for trace in fig_scatter.data:
        if trace.name == "Con Nombre":
            trace.marker.size = 12
            trace.marker.symbol = "star"
            trace.marker.line = dict(width=0)
            trace.marker.opacity = 1.0
        else: # Desconocido
            trace.marker.size = circle_size
            trace.marker.symbol = "circle"
            trace.marker.line = dict(width=0)
            trace.marker.opacity = circle_opacity

    # 5) Combinar densidad + scatter
    fig = go.Figure(data=list(fig_density.data) + list(fig_scatter.data))

    # 6) Asignar coloraxis a los scatter (para que usen la barra de color definida en layout)
    for tr in fig.data:
        if tr.type == "scatter": # Aplica a los puntos de fig_scatter
            tr.marker.coloraxis = "coloraxis"

    # 7) Layout general MODIFICADO
    fig.update_layout(
        title=title,
        width=700, # Puedes ajustar el tamaño
        height=600, # Puedes ajustar el tamaño
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='white', # <-- Cambio: Fondo blanco
        xaxis=dict(
            visible=False, # <-- Cambio: Ocultar eje X
            scaleanchor="y", 
            scaleratio=1
            ), 
        yaxis=dict(
            visible=False # <-- Cambio: Ocultar eje Y
            ), 
        legend=dict(x=1.02, y=1.14),
        coloraxis=dict( # Define la barra de color principal (para rugosidad)
            colorscale="Turbo",  # <-- Cambio: Paleta Turbo
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(
                title=label_bar,
                x=1.02,
                # Ajustes para los ticks personalizados
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext
            )
        )
    )
    
    # 8) (Los ticks ya se configuraron dentro de coloraxis.colorbar)

    return fig




def visualizar_heatmap(
    matriz: np.ndarray,
    acordes: list,
    title: str = "Matriz de Distancias"
) -> go.Figure:
    """
    Heatmap con hover que muestra los nombres de los acordes
    pero sin etiquetas visibles en ejes.
    """
    labels = [ac.name for ac in acordes]

    fig = px.imshow(
        matriz,
        x=labels,
        y=labels,
        color_continuous_scale="plasma",
        title=title,
        labels={"color": "Dissimilarity"}
    )
    fig.update_traces(
        hovertemplate=(
            "Acorde X: %{x}<br>"
            "Acorde Y: %{y}<br>"
            "Dissimilarity: %{z:.4f}"
            "<extra></extra>"
        )
    )

    # Oculta solo los nombres de los ticks (los labels), no el hover
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # También quita títulos de ejes por si quedaron
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig



def graficar_shepard(embedded: np.ndarray, matriz_distancias: np.ndarray, title: str = "Gráfico de Shepard") -> go.Figure:
    """
    Genera un gráfico de Shepard comparando las disimilitudes originales con las distancias en el embedding.
    """
    d_orig = squareform(matriz_distancias)
    d_emb = pdist(embedded)
    df = pd.DataFrame({
        "Distancia_Original": d_orig,
        "Distancia_Embebida": d_emb
    })
    try:
        import statsmodels.api as sm  # noqa: F401
        fig = px.scatter(df, x="Distancia_Original", y="Distancia_Embebida", trendline="ols", title=title)
    except Exception:
        fig = px.scatter(df, x="Distancia_Original", y="Distancia_Embebida", title=title)
        try:
            x = df["Distancia_Original"].values
            y = df["Distancia_Embebida"].values
            if len(x) >= 2:
                m, b = np.polyfit(x, y, 1)
                xs = np.linspace(np.min(x), np.max(x), 100)
                ys = m * xs + b
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="OLS (np.polyfit)", line=dict(color="red", width=2)))
        except Exception:
            pass
    return fig


def explore_umap_graph(reducer_obj, X_original: np.ndarray, acordes: list, n_neighbors_metrics: int = 5,
                       modo_forzado: str = None):
    """
    Genera un diagrama del grafo UMAP con fondo blanco, sin ejes y paleta "Turbo".
    """
    # Verificar si el grafo está disponible
    if not hasattr(reducer_obj, "graph_") or reducer_obj.graph_ is None:
        # Crear figura vacía con mensaje de error
        fig = go.Figure()
        fig.update_layout(
            title="Error: Grafo UMAP no disponible",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white', # Fondo blanco incluso para el error
            annotations=[dict(text="El objeto UMAP no fue entrenado con 'return_graph=True' o el grafo no se generó.", 
                              showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)]
        )
        return "El objeto UMAP no tiene atributo 'graph_' o es None.", fig

    graph_sparse = reducer_obj.graph_

    # Compatibilidad NetworkX (versiones nuevas vs viejas)
    try:
        # Intentar con la función más nueva
         G = nx.from_scipy_sparse_array(graph_sparse)
    except AttributeError:
         # Usar la función más antigua si la nueva no existe
        G = nx.from_scipy_sparse_matrix(graph_sparse)


    # Obtener posiciones de los nodos (desde el embedding si existe, sino layout)
    if hasattr(reducer_obj, "embedding_") and reducer_obj.embedding_ is not None:
        pos = {i: (reducer_obj.embedding_[i, 0], reducer_obj.embedding_[i, 1])
               for i in range(reducer_obj.embedding_.shape[0])}
    else:
        # Generar layout si no hay embedding (menos común para visualización final)
        print("Advertencia: Usando layout de grafo (spring_layout) porque no se encontró embedding en el objeto reducer.")
        pos = nx.spring_layout(G, seed=42) # Seed para reproducibilidad

    # Crear traza para las aristas (edges)
    edge_x, edge_y = [], []
    edge_weights = [] # Para posible color/grosor por peso
    for edge in G.edges(data=True): # Obtener datos (peso) de la arista
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]) # None para separar líneas
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2].get('weight', 1.0)) # Obtener peso, default 1

    # Normalizar pesos para grosor (opcional)
    min_w, max_w = min(edge_weights), max(edge_weights)
    if max_w > min_w:
        edge_widths = [0.1 + 1.5 * (w - min_w) / (max_w - min_w) for w in edge_weights]
    else:
        edge_widths = [0.5] * len(edge_weights) # Grosor fijo si todos pesos iguales

    # Crear traza de aristas (una sola línea con segmentos)
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'), # Color fijo gris para aristas
        hoverinfo="none",
        mode="lines",
        showlegend=False
    )
    # Alternativa: Crear múltiples líneas para variar grosor (más complejo)
    # edge_traces = []
    # current_idx = 0
    # for i in range(0, len(edge_x), 3): # Cada 3 puntos es una arista (x0, x1, None)
    #     width = edge_widths[current_idx]
    #     edge_traces.append(go.Scatter(x=edge_x[i:i+2], y=edge_y[i:i+2], mode='lines',
    #                                   line=dict(width=width, color='#888'), hoverinfo='none', showlegend=False))
    #     current_idx += 1


    # Crear DataFrame para los nodos
    pos_array = np.array([pos[i] for i in sorted(pos.keys())]) # Asegurar orden correcto
    df_nodes = build_chord_dataframe(pos_array, acordes, X_original)

    # Transformación de color para los nodos (rugosidad)
    color_vals, cmin, cmax, tvals, ttext, label_bar = transform_rugosidad_for_color(
        df_nodes, "TotalRug", modo_forzado=modo_forzado
    )
    df_nodes["__color__"] = color_vals

    # Crear figura de Plotly Express para los nodos
    fig_nodes = px.scatter(
        df_nodes, x="X", y="Y",
        color="__color__", # Color por rugosidad
        # color_continuous_scale="Turbo", # <-- Se define en layout.coloraxis
        symbol="Conocido", symbol_sequence=["star", "circle"],
        hover_name="hover_text",
        hover_data={"X": False, "Y": False, "TotalRug": False, "Conocido": False},
        # title="Grafo UMAP" # <-- Se pone en layout final
    )

    # Ajustar tamaño/borde de los nodos (igual que en scatter_density)
    for trace in fig_nodes.data:
        if trace.name == "Con Nombre":
            trace.marker.size = 12
            trace.marker.line = dict(color="black", width=2)
        else: # Desconocido
            trace.marker.size = 8
            trace.marker.line = dict(width=0)
        # Asegurar que usen el coloraxis global
        trace.marker.coloraxis = "coloraxis"

    # Combinar aristas y nodos en una sola figura
    # fig = go.Figure(data=edge_traces + list(fig_nodes.data)) # Si usamos múltiples edge traces
    fig = go.Figure(data=[edge_trace] + list(fig_nodes.data)) # Si usamos una sola edge trace

    # Actualizar layout MODIFICADO
    fig.update_layout(
        title="Grafo UMAP",
        width=700, # Ajustable
        height=600, # Ajustable
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='white', # <-- Cambio: Fondo blanco
        xaxis=dict(
            visible=False, # <-- Cambio: Ocultar eje X
            scaleanchor="y", 
            scaleratio=1
            ),
        yaxis=dict(
            visible=False # <-- Cambio: Ocultar eje Y
            ),
        legend=dict(x=1.02, y=1.14), # Ajustar posición leyenda
        coloraxis=dict( # Barra de color para los nodos (rugosidad)
            cmin=cmin, cmax=cmax,
            colorscale="Turbo", # <-- Cambio: Paleta Turbo
            colorbar=dict(
                title=label_bar, 
                x=1.02,
                # Ticks personalizados
                tickmode="array",
                tickvals=tvals,
                ticktext=ttext
                )
        )
    )

    # Calcular metrica simple del grafo (promedio de grado)
    avg_degree = np.mean([d for _, d in G.degree()])
    metrics_str = f"Promedio de grado del grafo: {avg_degree:.2f}"

    return metrics_str, fig

def visualizar_laplacian_scatter(embedding: np.ndarray, acordes: list, X_original: np.ndarray, 
                                 title: str = "Embebido Laplacian",
                                 modo_forzado: str = None) -> go.Figure:
    """
    Genera un gráfico combinando un contorno de densidad y un scatter plot para un embedding obtenido 
    mediante Laplacian Eigenmaps, con transformación robusta de rugosidad.
    """
    df = build_chord_dataframe(embedding, acordes, X_original)

    # Transform
    color_vals, cmin, cmax, tvals, ttext, label_bar = transform_rugosidad_for_color(
        df, "TotalRug", modo_forzado=modo_forzado
    )
    df["__color__"] = color_vals

    fig_density = px.density_contour(df, x="X", y="Y", nbinsx=30, nbinsy=30, title=title)
    fig_density.update_traces(contours_coloring='fill', colorscale='Blues', showscale=False,
                              opacity=0.3, line=dict(width=1))

    fig_scatter = px.scatter(
        df, x="X", y="Y",
        color="__color__",
        color_continuous_scale="Viridis",
        symbol="Conocido", symbol_sequence=["star", "circle"],
        hover_name="hover_text",
        title=title,
        hover_data={"X": False, "Y": False, "TotalRug": False, "Conocido": False}
    )
    outline_traces = []
    for trace in fig_scatter.data:
        if trace.name == "Con Nombre":
            trace.marker.size = 12
            trace.marker.symbol = "star"
            trace.marker.line = dict(width=0)

            outline_marker = {
                "symbol": "star-open",
                "size": trace.marker.size + 2,
                "line": {"width": 2},
                "coloraxis": getattr(trace.marker, "coloraxis", None) or "coloraxis",
                "showscale": False,
            }
            if getattr(trace.marker, "color", None) is not None:
                outline_marker["color"] = trace.marker.color
            if getattr(trace.marker, "colorscale", None) is not None:
                outline_marker["colorscale"] = trace.marker.colorscale
            if getattr(trace.marker, "cmin", None) is not None:
                outline_marker["cmin"] = trace.marker.cmin
            if getattr(trace.marker, "cmax", None) is not None:
                outline_marker["cmax"] = trace.marker.cmax

            
            trace.marker.line = dict(width=0)

    fig = go.Figure(data=list(fig_density.data) + list(fig_scatter.data) + outline_traces)
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        legend=dict(x=1.02, y=1.14),
        coloraxis=dict(
            cmin=cmin, cmax=cmax,
            colorscale="Viridis",
            colorbar=dict(title=label_bar, x=1.02)
        )
    )

    # Asignar coloraxis a todos los scatter
    for trace in fig.data:
        if trace.type == "scatter":
            trace.marker.coloraxis = "coloraxis"

    if tvals is not None and ttext is not None:
        fig.update_layout(
            coloraxis_colorbar=dict(
                tickmode="array",
                tickvals=tvals,
                ticktext=ttext
            )
        )

    return fig


def visualizar_sp_sf(acordes: list, title: str = "Representaci�n Sp�Sf de Acordes",
                     modo_forzado: str = None) -> go.Figure:
    """
    Calcula y muestra la representaci�n Sp�Sf a partir de una lista de acordes,
    con la l�gica de transformaci�n de color para la rugosidad.
    """
    embeddings = representacion_sp_sf(acordes)
    identities = [get_chord_type_from_intervals(ac.intervals, with_alias=True) for ac in acordes]

    tipo_base = [identity.name for identity in identities]
    tipo_display = []
    total_roughness = []
    for ac, identity in zip(acordes, identities):
        if identity.name == "Unknown":
            tipo_display.append("Unknown")
        else:
            alias_list = [alias for alias in identity.aliases if alias]
            if alias_list:
                tipo_display.append(", ".join([identity.name] + list(alias_list)))
            else:
                tipo_display.append(identity.name)
        total_roughness.append(ac.total_roughness if ac.total_roughness is not None else 0)

    df = pd.DataFrame({
        "Sp": embeddings[:, 0],
        "Sf": embeddings[:, 1],
        "Acorde": [ac.name for ac in acordes],
        "Intervalos": [str(ac.intervals) for ac in acordes],
        "Tipo": tipo_display,
        "TotalRug": total_roughness,
    })
    df["Conocido"] = ["Con Nombre" if name != "Unknown" else "Desconocido" for name in tipo_base]
    df["hover_text"] = (
        "Acorde: " + df["Acorde"] + "<br>" +
        "Intervalos: " + df["Intervalos"] + "<br>" +
        "Tipo: " + df["Tipo"] + "<br>" +
        "TotalRug: " + df["TotalRug"].astype(str)
    )

    # Transformaci�n color
    color_vals, cmin, cmax, tvals, ttext, label_bar = transform_rugosidad_for_color(
        df, "TotalRug", modo_forzado=modo_forzado
    )
    df["__color__"] = color_vals

    trace_conocidos = df[df["Conocido"] == "Con Nombre"]
    trace_desconocidos = df[df["Conocido"] == "Desconocido"]

    scatter_known = go.Scatter(
        x=trace_conocidos["Sp"],
        y=trace_conocidos["Sf"],
        mode="markers",
        marker=dict(
            symbol="star",
            size=12,
            color=outline_colors_sp,
            showscale=False,
            line=dict(color=outline_colors_sp, width=2),
            colorbar=dict(title=label_bar, x=1.02),
            coloraxis="coloraxis"
        ),
        hoverinfo="text",
        text=trace_conocidos["hover_text"],
        name="Con Nombre"
    )

    outline_colors_sp = sample_colors_for_values(trace_conocidos["__color__"], cmin, cmax)
    scatter_unknown = go.Scatter(
        x=trace_desconocidos["Sp"],
        y=trace_desconocidos["Sf"],
        mode="markers",
        marker=dict(
            symbol="circle",
            size=10,
            color=trace_desconocidos["__color__"],
            colorscale="Viridis",
            cmin=cmin, cmax=cmax,
            line=dict(color="gray", width=1)
        ),
        hoverinfo="text",
        text=trace_desconocidos["hover_text"],
        name="Desconocido"
    )

    fig = go.Figure(data=[scatter_known, scatter_unknown])
    fig.update_layout(
        title=title,
        xaxis_title="Sp",
        yaxis_title="Sf",
        coloraxis_colorscale="Viridis",
        coloraxis_colorbar=dict(title=label_bar, x=1.02)
    )
    fig.update_layout(coloraxis_colorbar=COMMON_COLORBAR)
    return fig



# =======================
# Clase Integradora de Visualización
# =======================

class VisualizadorAcordesModular:
    """
    Clase integradora que agrupa todas las funciones de visualización para el laboratorio de acordes.
    """
    def reporte(self, resultado: 'ResultadoExperimento', X_original: np.ndarray):
        from metrics import compute_additional_metrics  # Importación local para evitar dependencias circulares
        metrics = compute_additional_metrics(X_original, resultado.embeddings, n_neighbors=EVAL_N_NEIGHBORS)
        report_text = f"Modelo de Reducción: {resultado.metricas.get('reduccion_method', 'N/A')}\n"
        if resultado.metricas.get('stress') is not None:
            report_text += f"Stress (MDS): {resultado.metricas.get('stress'):.4f}\n"
        if resultado.metricas.get('stress_normalizado') is not None:
            report_text += f"Stress Normalizado: {resultado.metricas.get('stress_normalizado'):.4f}\n"
        if resultado.metricas.get('loss') is not None:
            report_text += f"UMAP Loss: {resultado.metricas.get('loss'):.4f}\n"
        if resultado.metricas.get('varianza_explicada') is not None:
            report_text += f"Varianza Explicada (dist vs embedding): {resultado.metricas.get('varianza_explicada'):.4f}\n"
    
        report_text += "\n--- Métricas Compartidas ---\n"
        for key, value in metrics.items():
            report_text += f"{key}: {value:.4f}\n"
        print(report_text)

    def scatter(self, embeddings: np.ndarray, acordes: list, X_original: np.ndarray, modo_forzado: str = None):
        fig = visualizar_scatter_density(embeddings, acordes, X_original,
                                         title="Embeddings del Espacio Acústico",
                                         modo_forzado=modo_forzado)
        fig.show()

    def heatmap(self, matriz_distancias: np.ndarray, acordes: list):
        fig = visualizar_heatmap(matriz_distancias, acordes, title="Matriz de Distancias")
        fig.show()

    def shepard(self, embeddings: np.ndarray, matriz_distancias: np.ndarray):
        fig = graficar_shepard(embeddings, matriz_distancias, title="Gráfico de Shepard")
        fig.show()

    def grafo_umap(self, reducer_obj, X_original: np.ndarray, acordes: list, n_neighbors_metrics: int,
                   modo_forzado: str = None):
        metrics_str, fig = explore_umap_graph(reducer_obj, X_original, acordes,
                                              n_neighbors_metrics=n_neighbors_metrics,
                                              modo_forzado=modo_forzado)
        print(metrics_str)
        fig.show()

    def laplacian(self, embedding: np.ndarray, acordes: list, X_original: np.ndarray, modo_forzado: str = None):
        fig = visualizar_laplacian_scatter(embedding, acordes, X_original,
                                           title="Embebido Laplacian",
                                           modo_forzado=modo_forzado)
        fig.show()

    def sp_sf(self, acordes: list, modo_forzado: str = None):
        fig = visualizar_sp_sf(acordes, title="Representación Sp–Sf de Acordes",
                               modo_forzado=modo_forzado)
        fig.show()


print("[ok] Módulo visualization.py cargado con la nueva lógica de color transformado.")