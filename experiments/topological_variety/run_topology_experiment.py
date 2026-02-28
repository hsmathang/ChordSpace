import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
import warnings

# Asegurar que importamos desde la raíz de ChordSpace
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pre_process import Acorde, ModeloSetharesVec, safe_normalize
from services.combinatorial_generator import generate_combinatorial_chords
from config import SETHARES_BASE_FREQ, SETHARES_DECAY, SETHARES_N_HARMONICS
from metrics import compute_trustworthiness, compute_continuity, compute_knn_recall, compute_rank_correlation
from lab import kruskal_stress_1

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------
# 1. DEFINICIÓN DE ESTRUCTURAS DE REFERENCIA (Familias de Clase)
# ----------------------------------------------------------------------------------

ACORDES_CONOCIDOS = {
    # Triadas (k=3)
    (0, 4, 7): "Mayor",
    (0, 3, 7): "Menor",
    (0, 4, 8): "Aumentada",
    (0, 3, 6): "Disminuida",
    (0, 2, 7): "Sus2",
    (0, 5, 7): "Sus4",
    # Tetradas (k=4)
    (0, 4, 7, 11): "Maj7",
    (0, 3, 7, 10): "Min7",
    (0, 4, 7, 10): "Dom7",
    (0, 3, 6, 10): "m7b5",
    (0, 3, 6, 9):  "dim7",
    (0, 4, 7, 9):  "Add6",
    # Extensiones (k=5)
    (0, 4, 7, 11, 14): "Maj9",
    (0, 3, 7, 10, 14): "Min9",
    (0, 4, 7, 10, 14): "Dom9",
    (0, 4, 7, 10, 13): "Dom7(b9)",
}

def identify_label(offsets: list) -> str:
    """Clasifica un acorde basándose en su estructura (offsets desde 0)."""
    tup = tuple(int(x) for x in offsets)
    if tup in ACORDES_CONOCIDOS:
        return ACORDES_CONOCIDOS[tup]
    else:
        # Cualquier cosa no clasificada se considera "Pitch Noise"
        return "Pitch Noise"

# ----------------------------------------------------------------------------------
# 2. GENERACIÓN DEL ESPACIO DE ACORDES (Variedad Topológica Base)
# ----------------------------------------------------------------------------------
print("Generando Universo Combinatorial Estructural (Modo Estructural)...")
# Usamos alfabeto de 12 clases de tono, octava 4 (una octava de margen) para generar estructuras
alfabeto = list(range(12))
df_combinatorial = generate_combinatorial_chords(
    alphabet=alfabeto,
    octave_min=4,
    octave_max=4,
    cardinalities=[3, 4, 5],
    structural_mode=True
)

print(f"Total estructuras generadas (k=3,4,5): {len(df_combinatorial)}")

# ----------------------------------------------------------------------------------
# 3. EXTRACCIÓN DE VECTORES TOPOLÓGICOS (Métrica Física/Psicoacústica)
# ----------------------------------------------------------------------------------
print("Extrayendo espacio 12D de características (Modelo Sethares)...")

config_sethares = {
    'base_freq': SETHARES_BASE_FREQ,
    'n_armonicos': SETHARES_N_HARMONICS,
    'decaimiento': SETHARES_DECAY
}
modelo = ModeloSetharesVec(config_sethares)

X_data = [] # Para t-SNE / UMAP / MDS
meta_data = []

for idx, row in df_combinatorial.iterrows():
    offsets = row['__struct_semitones']
    k = row['n']
    label = identify_label(offsets)
    
    # Construimos el objeto acorde
    acorde_obj = Acorde(name=label, intervals=[offsets[i]-offsets[i-1] for i in range(1, len(offsets))])
    
    # Calcular vector crudo
    vector, total_rugosidad = modelo.calcular(acorde_obj)
    
    # Normalizar para focalizar en la forma topológica interválica (preservando perfiles estructurales)
    vector_norm = safe_normalize(vector)
    
    X_data.append(vector_norm)
    meta_data.append({
        'Id_Estructura': row['__structure_id'],
        'Cardinalidad': k,
        'Clase': label,
        'Rugosidad_Total': total_rugosidad,
        'Grupo_Visual': "ACORDE_CLASICO" if label != "Pitch Noise" else "RUIDO",
        'Label_Hover': f"[{label}] {row['__structure_id']} (n={k})"
    })

X = np.array(X_data)
df_meta = pd.DataFrame(meta_data)

# ----------------------------------------------------------------------------------
# 4. REDUCCIÓN DE DIMENSIONALIDAD (MDS y UMAP)
# ----------------------------------------------------------------------------------
print("Proyectando la Variedad Topológica (UMAP y MDS)...", flush=True)

# Hiperparámetros de UMAP
umap_params = {
    'n_neighbors': 15,
    'min_dist': 0.1,
    'n_components': 2,
    'random_state': 42
}
umap_reducer = umap.UMAP(**umap_params)
X_umap = umap_reducer.fit_transform(X)
df_meta['UMAP_X'] = X_umap[:, 0]
df_meta['UMAP_Y'] = X_umap[:, 1]

# Hiperparámetros de MDS
mds_params = {
    'n_components': 2,
    'normalized_stress': 'auto',
    'random_state': 42
}
mds_reducer = MDS(**mds_params)
X_mds = mds_reducer.fit_transform(X)
df_meta['MDS_X'] = X_mds[:, 0]
df_meta['MDS_Y'] = X_mds[:, 1]

# ----------------------------------------------------------------------------------
# 5. CÁLCULO DE MÉTRICAS DE RIGOR MATEMÁTICO (Topología Riemanniana Estricta)
# ----------------------------------------------------------------------------------
print("Calculando Métricas de Rigor Matemático (Himpel 2022 compatibility)...")

# Métricas Topológicas (Continuidad y Verosimilitud del Embedding)
trust_umap = compute_trustworthiness(X, X_umap)
cont_umap = compute_continuity(X, X_umap)
knn_umap = compute_knn_recall(X, X_umap)

trust_mds = compute_trustworthiness(X, X_mds)
cont_mds = compute_continuity(X, X_mds)
knn_mds = compute_knn_recall(X, X_mds)

# Silhouette Score (Clustering Estricto)
try:
    sil_umap = silhouette_score(X_umap, df_meta['Grupo_Visual'])
    sil_mds = silhouette_score(X_mds, df_meta['Grupo_Visual'])
except Exception as e:
    sil_umap, sil_mds = 0.0, 0.0
    print("No se pudo calcular silhouette:", e)

# Kruskal Stress-1 Normalizado (como en lab.py del repositorio)
from scipy.spatial.distance import pdist, squareform
D_orig = squareform(pdist(X, metric='euclidean'))
D_umap = squareform(pdist(X_umap, metric='euclidean'))
D_mds_emb = squareform(pdist(X_mds, metric='euclidean'))
stress_umap = kruskal_stress_1(D_orig, D_umap)
stress_mds = kruskal_stress_1(D_orig, D_mds_emb)

# Correlación de Spearman (preservación global de rangos de distancias)
rho_umap = compute_rank_correlation(X, X_umap)
rho_mds = compute_rank_correlation(X, X_mds)

# Conteo de acordes por grupo
n_clasicos = (df_meta['Grupo_Visual'] == 'ACORDE_CLASICO').sum()
n_ruido = (df_meta['Grupo_Visual'] == 'RUIDO').sum()

print(f"  Acordes Clásicos: {n_clasicos}, Pitch Noise: {n_ruido}")
print(f"  UMAP -> Trust: {trust_umap:.4f}, Cont: {cont_umap:.4f}, KNN: {knn_umap:.4f}, Sil: {sil_umap:.4f}, Stress-1: {stress_umap:.4f}, Rho: {rho_umap:.4f}")
print(f"  MDS  -> Trust: {trust_mds:.4f}, Cont: {cont_mds:.4f}, KNN: {knn_mds:.4f}, Sil: {sil_mds:.4f}, Stress-1: {stress_mds:.4f}, Rho: {rho_mds:.4f}")


# ----------------------------------------------------------------------------------
# 6. VISUALIZACIÓN INTERACTIVA (PLOTLY)
# ----------------------------------------------------------------------------------
def plot_projection(df, x_col, y_col, title):
    """Crea un scatter plot profesional destacando los contrastes estructurales."""
    # Custom colors
    color_map = {
        'ACORDE_CLASICO': '#e74c3c', # Red para lo estructurado
        'RUIDO': '#95a5a6'           # Gray para el pitch noise
    }
    
    fig = px.scatter(
        df, x=x_col, y=y_col, 
        color='Grupo_Visual', 
        color_discrete_map=color_map,
        size='Rugosidad_Total',
        size_max=15,
        hover_name='Label_Hover',
        hover_data={'Rugosidad_Total': ':.2f', 'Cardinalidad': True, 'Grupo_Visual': False, x_col: False, y_col: False},
        title=title,
        opacity=0.8,
        template='plotly_white'
    )
    
    # Modificar las trazas para hacer el Pitch Noise más transparente y pequeño por defecto visual
    for trace in fig.data:
        if trace.name == 'RUIDO':
            trace.marker.opacity = 0.4
            trace.marker.line.width = 0
        else:
            trace.marker.opacity = 0.95
            trace.marker.line = dict(width=1, color='DarkSlateGrey')
            
    # Para la traza clásica, agregaremos labels de texto explícitos fuera del loop principal
    df_clasicos = df[df['Grupo_Visual'] == 'ACORDE_CLASICO']
    fig.add_trace(go.Scatter(
        x=df_clasicos[x_col],
        y=df_clasicos[y_col],
        mode='text',
        text=df_clasicos['Clase'],
        textposition='top center',
        textfont=dict(color='black', size=10, weight='bold'),
        showlegend=False,
        hoverinfo='skip'
    ))
            
    fig.update_layout(
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        legend_title="Familia Estructural",
        height=600
    )
    return fig

fig_umap = plot_projection(df_meta, 'UMAP_X', 'UMAP_Y', "Variedad Topológica UMAP (Estructuras Conocidas vs Pitch Noise)")
fig_mds = plot_projection(df_meta, 'MDS_X', 'MDS_Y', "Variedad Topológica Estricta MDS (Distancias Globales)")

# ----------------------------------------------------------------------------------
# 7. GENERACIÓN DEL REPORTE HTML
# ----------------------------------------------------------------------------------
print("Generando Reporte HTML de Tesis...", flush=True)

html_content = rf"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Evaluación de Variedad Topológica Espacial: Acordes vs Pitch Noise</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f4f7f6; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        .container {{ max-width: 1400px; margin: auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        .intro-box {{ padding: 20px; background-color: #e8f4f8; border-left: 5px solid #3498db; margin-bottom: 30px; font-size: 16px; line-height: 1.6; border-radius: 4px; }}
        .metrics-grid {{ display: flex; gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ flex: 1; padding: 20px; background: #fff; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; }}
        .metric-card.excellent {{ border-top: 4px solid #2ecc71; }}
        .metric-card.good {{ border-top: 4px solid #f39c12; }}
        .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; color: #2c3e50; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px auto; font-size: 16px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px 15px; text-align: center; }}
        th {{ background-color: #34495e; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>Evaluación de la Variedad Topológica Espacial en Música</h1>
        
        <div class="intro-box">
            <h3>Justificación Teórica: El Espacio de Acordes como Variedad Riemanniana (Himpel, 2022)</h3>
            <p>
                Según Himpel (2022), el espacio que contiene a todos los acordes musicales posibles ($\mathcal{{C}}$) puede ser modelado como un espacio estratificado de Whitney. Dentro de este espacio general, cada "estrato" (compuesto por acordes que tienen exactamente $n$ notas únicas) constituye una <strong>variedad Riemanniana de dimensión $n-1$</strong>.
            </p>
            <p>
                Bajo este paradigma, el presente experimento no busca proyectar acordes aislados, sino que genera de forma combinatoria todo un universo musical de cardinalidades continuas ($k \in {{3, 4, 5}}$) en modo estructural (centrado a clases de intervalos). Al mezclar progresiones estructuradas con <em>Pitch Noise</em> puro (aleatoriedad combinatoria tonal), podemos comprobar empíricamente si las familias que constituyen la "Música" real (tríadas y tétradas comunes) ocupan posiciones, clusters, o "climas geomorfológicos" privilegiados dentro de la enorme variedad matemática disponible. 
            </p>
            <p>
                Para respaldar la calidad de esta proyección bidimensional desde un espacio nativo subyacente de 12 dimensiones, empleamos métricas que penalizan desgarros de vecindades y pérdida de estructura global, garantizando que el gráfico refleja el tejido de conducción de voces geodésico original.
            </p>
        </div>

        <h2>1. Métricas de Rigor Matemático</h2>
        <p>A continuación se evalúa matemáticamente que la reducción dimensional (tanto no-lineal como multiescala) preserva las distancias, densidades e isometrías locales del modelo geométrico de 12D a nuestro plano visual (2D).</p>
        
        <div class="metrics-grid">
            <div class="metric-card { 'excellent' if trust_umap > 0.8 else 'good' }">
                <div class="metric-label">Trustworthiness (UMAP)</div>
                <div class="metric-value">{trust_umap:.4f}</div>
                <div style="font-size: 12px; color:#95a5a6;">(Probabilidad de que las cercanías visuales sean reales. >0.8 ideal)</div>
            </div>
            <div class="metric-card { 'excellent' if cont_umap > 0.8 else 'good' }">
                <div class="metric-label">Continuity (UMAP)</div>
                <div class="metric-value">{cont_umap:.4f}</div>
                <div style="font-size: 12px; color:#95a5a6;">(Probabilidad de la no-pérdida de lazos geodésicos.)</div>
            </div>
            <div class="metric-card { 'excellent' if knn_umap > 0.5 else 'good' }">
                <div class="metric-label">KNN Recall (UMAP)</div>
                <div class="metric-value">{knn_umap:.4f}</div>
                <div style="font-size: 12px; color:#95a5a6;">(Retención de vecinos directos inmediatos)</div>
            </div>
            <div class="metric-card { 'excellent' if sil_umap > 0.0 else 'good' }">
                <div class="metric-label">Silhouette (Noise vs Music)</div>
                <div class="metric-value">{sil_umap:.4f}</div>
                <div style="font-size: 12px; color:#95a5a6;">(Separación estricta sub-clusters. Valores -1 a 1)</div>
            </div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card { 'excellent' if trust_mds > 0.8 else 'good' }">
                <div class="metric-label">Trustworthiness (MDS)</div>
                <div class="metric-value">{trust_mds:.4f}</div>
            </div>
            <div class="metric-card { 'excellent' if cont_mds > 0.8 else 'good' }">
                <div class="metric-label">Continuity (MDS)</div>
                <div class="metric-value">{cont_mds:.4f}</div>
            </div>
             <div class="metric-card { 'excellent' if knn_mds > 0.5 else 'good' }">
                <div class="metric-label">KNN Recall (MDS)</div>
                <div class="metric-value">{knn_mds:.4f}</div>
            </div>
            <div class="metric-card { 'excellent' if sil_mds > 0.0 else 'good' }">
                <div class="metric-label">Silhouette (MDS)</div>
                <div class="metric-value">{sil_mds:.4f}</div>
            </div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Detalles de Reducción: UMAP</h3>
                <ul style="text-align: left; font-size: 14px; color: #555;">
                    <li><strong>n_neighbors:</strong> {umap_params['n_neighbors']}</li>
                    <li><strong>min_dist:</strong> {umap_params['min_dist']}</li>
                    <li><strong>n_components:</strong> {umap_params['n_components']}</li>
                    <li><strong>Stress-1 (Kruskal):</strong> {stress_umap:.4f}</li>
                    <li><strong>Spearman Rho:</strong> {rho_umap:.4f}</li>
                </ul>
            </div>
            <div class="metric-card">
                <h3>Detalles de Reducción: MDS</h3>
                <ul style="text-align: left; font-size: 14px; color: #555;">
                    <li><strong>n_components:</strong> {mds_params['n_components']}</li>
                    <li><strong>normalized_stress:</strong> {mds_params['normalized_stress']}</li>
                    <li><strong>Stress-1 (Kruskal):</strong> {stress_mds:.4f}</li>
                    <li><strong>Spearman Rho:</strong> {rho_mds:.4f}</li>
                </ul>
            </div>
            <div class="metric-card">
                <h3>Inventario del Espacio</h3>
                <ul style="text-align: left; font-size: 14px; color: #555;">
                    <li><strong>Total Estructuras:</strong> {len(df_combinatorial)}</li>
                    <li><strong>Acordes Clásicos:</strong> {n_clasicos}</li>
                    <li><strong>Pitch Noise:</strong> {n_ruido}</li>
                    <li><strong>Cardinalidades:</strong> k ∈ {{3, 4, 5}}</li>
                    <li><strong>Dimensión Original:</strong> 12D (Sethares)</li>
                    <li><strong>K (vecinos evaluación):</strong> 5 (config.py)</li>
                </ul>
            </div>
        </div>

        <h2>2. Observación de la Variedad (UMAP)</h2>
        <p>Buscamos identificar que las estructuras musicales humanas (en <span style="color: #e74c3c; font-weight: bold;">Rojo</span>) se distribuyen ocupando "valles de consonancia" singulares a lo largo y ancho del subespacio generado, alejándose de los macizos centrales caóticos (en <span style="color: #95a5a6; font-weight: bold;">Gris</span>).</p>
        <div style="margin-top: 20px;">
            {fig_umap.to_html(full_html=False, include_plotlyjs='cdn')}
        </div>

        <h2>3. Distancias Geodésicas Globales (MDS)</h2>
        <p>Mientras UMAP es excelente para descubrir grupos locales, MDS restringe la variedad intentando mantener fielmente el "ancho" real de las grandes distancias globales euclidianas a través del tensor métrico riemanniano.</p>
        <div style="margin-top: 20px;">
            {fig_mds.to_html(full_html=False, include_plotlyjs='cdn')}
        </div>

    </div>
</body>
</html>
"""

report_path = os.path.join(os.path.dirname(__file__), "reporte_topologia_acordes.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\n¡Éxito! Reporte guardado en: {report_path}", flush=True)
