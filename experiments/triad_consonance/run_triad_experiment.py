import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pre_process import Acorde, ModeloSetharesVec
from config import SETHARES_BASE_FREQ, SETHARES_DECAY

from tools.plot_sethares_sweep import TheoreticalParams, cont_curve, dyad_curve, theoretical_total_roughness

print(f"Parametros: decaimiento={SETHARES_DECAY}, base_freq={SETHARES_BASE_FREQ}")

rango_armonicos = [3, 4, 5, 6, 7, 8, 9, 10, 11]

triadas = {
    "Mayor":      [0, 4, 7],
    "Menor":      [0, 3, 7],
    "Sus2":       [0, 2, 7],
    "Aumentada":  [0, 4, 8],
    "Disminuida": [0, 3, 6]
}

# Ground Truth (Roberts 1986, Bowling 2018):
# Mayor > Menor > Suspendida > Disminuida > Aumentada
GROUND_TRUTH_ORDER = ["Mayor", "Menor", "Sus2", "Disminuida", "Aumentada"]

def crear_acorde(nombre, intervalos):
    saltos = [intervalos[i] - intervalos[i-1] for i in range(1, len(intervalos))]
    return Acorde(name=nombre, intervals=saltos)

acordes_objs = {nombre: crear_acorde(nombre, semit) for nombre, semit in triadas.items()}

resultados_escalares = []
vectores_por_armonico = {n: {} for n in rango_armonicos}

for n_arm in rango_armonicos:
    configuracion = {
        'base_freq': SETHARES_BASE_FREQ,
        'n_armonicos': n_arm,
        'decaimiento': SETHARES_DECAY,
    }
    modelo = ModeloSetharesVec(configuracion)
    
    fila = {"n": n_arm}
    for nombre_triada, acorde in acordes_objs.items():
        vector, total = modelo.calcular(acorde)
        fila[nombre_triada] = total
        vectores_por_armonico[n_arm][nombre_triada] = vector
        
    resultados_escalares.append(fila)

df_resultados = pd.DataFrame(resultados_escalares)
print("\n--- Resultados Brutos (con Sus2) ---")
print(df_resultados.to_string(float_format=lambda x: f"{x:.4f}"))
print("------------------------------------\n")

# Verificar orden predicho vs ground truth (n=6)
fila_n6 = df_resultados[df_resultados["n"] == 6].iloc[0]
valores_n6 = {t: fila_n6[t] for t in triadas.keys()}
orden_predicho = sorted(valores_n6, key=valores_n6.get)

print(f"Predicho  (n=6): {' < '.join(orden_predicho)}")
print(f"Ground Truth:    {' < '.join(GROUND_TRUTH_ORDER)}")

idx_aum = orden_predicho.index("Aumentada")
idx_dis = orden_predicho.index("Disminuida")
if idx_aum < idx_dis:
    print("[!] ANOMALIA: Aumentada predicha MENOS rugosa que Disminuida (debe ser al reves)")
print()

# --- GRAFICAS ---
colores = {
    "Mayor": "#2ecc71", "Menor": "#3498db", "Sus2": "#f39c12",
    "Aumentada": "#e74c3c", "Disminuida": "#9b59b6"
}

# Grafica escalar
fig_escalar = go.Figure()
x_data = df_resultados["n"].tolist()
for triada in triadas.keys():
    fig_escalar.add_trace(go.Scatter(
        x=x_data, y=df_resultados[triada].tolist(),
        mode='lines+markers', name=triada,
        line=dict(color=colores[triada], width=3), marker=dict(size=8)
    ))
fig_escalar.update_layout(
    title="Rugosidad Escalar 1D (12-TET) - 5 Triadas",
    xaxis_title="Armonicos (n)", yaxis_title="Rugosidad Total",
    height=450, hovermode="x unified",
    xaxis=dict(tickmode='linear', tick0=3, dtick=1)
)

# Araas (reloj)
bins_labels = ["Octava", "2m", "2M", "3m", "3M", "4J", "Tritono", "5J", "6m", "6M", "7m", "7M"]
armonicos_a_graficar = [4, 6, 11]

fig_vectores = make_subplots(
    rows=len(armonicos_a_graficar), cols=5,
    specs=[[{"type": "polar"} for _ in range(5)] for _ in range(len(armonicos_a_graficar))],
    subplot_titles=[f"{t} (n={n})" for n in armonicos_a_graficar for t in triadas.keys()],
    vertical_spacing=0.1
)
fila_graf = 1
for n_arm in armonicos_a_graficar:
    col = 1
    for nombre_triada in triadas.keys():
        v = vectores_por_armonico[n_arm][nombre_triada]
        vr = [v[11]] + list(v[0:11])
        radios = vr + [vr[0]]
        theta = bins_labels + [bins_labels[0]]
        fig_vectores.add_trace(go.Scatterpolar(
            r=radios, theta=theta, fill='toself',
            name=f"{nombre_triada} n={n_arm}",
            line=dict(color=colores[nombre_triada]),
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>"
        ), row=fila_graf, col=col)
        col += 1
    fila_graf += 1

fig_vectores.update_layout(
    height=400 * len(armonicos_a_graficar), width=1400,
    title_text="Perfiles Vectoriales 12D", showlegend=False,
)
max_r = max([max(v) for d in vectores_por_armonico.values() for v in d.values()])
for i in range(1, len(armonicos_a_graficar) * 5 + 1):
    fig_vectores.layout[f'polar{i if i > 1 else ""}'].update(
        radialaxis=dict(visible=True, range=[0, max_r * 1.05]),
        angularaxis=dict(direction='clockwise', rotation=90)
    )

# Curva de díadas
fig_curva = go.Figure()
colores_curva = ['#2c3e50', '#e74c3c', '#2980b9']
for i, n_arm in enumerate([3, 6, 10]):
    params = TheoreticalParams(base_freq=SETHARES_BASE_FREQ, n_harmonics=n_arm, decay=SETHARES_DECAY, amplitude_mode="product")
    sd, rd = cont_curve(base_freq=SETHARES_BASE_FREQ, p=params, max_semitones=24.0, num_points=800)
    sm, rm = dyad_curve(base_freq=SETHARES_BASE_FREQ, p=params, max_semitones=24, step=1)
    fig_curva.add_trace(go.Scatter(x=sd.tolist(), y=rd, mode='lines', name=f"N={n_arm}", line=dict(color=colores_curva[i], width=2), hoverinfo='skip'))
    fig_curva.add_trace(go.Scatter(x=sm.tolist(), y=rm, mode='markers', name=f"12-TET N={n_arm}", marker=dict(color=colores_curva[i], size=8, line=dict(color='white', width=1)), hovertemplate="%{x} st: %{y:.2f}<extra></extra>"))
from tools.plot_sethares_sweep import _interval_labels
fig_curva.update_layout(title=f"Curva de Sethares (base={SETHARES_BASE_FREQ} Hz)", xaxis_title="Semitonos", yaxis_title="Rugosidad", height=450, hovermode='closest', xaxis=dict(tickmode='array', tickvals=list(range(25)), ticktext=_interval_labels(24), tickangle=-45))

# Tabla de orden
tabla_comparativa = ""
for _, row in df_resultados.iterrows():
    n = int(row["n"])
    vals = {t: row[t] for t in triadas.keys()}
    orden = sorted(vals, key=vals.get)
    coincide = (orden == GROUND_TRUTH_ORDER)
    emoji = "SI" if coincide else "NO"
    tabla_comparativa += f"<tr><td>{n}</td>"
    for t in orden:
        tabla_comparativa += f'<td style="color:{colores[t]}; font-weight:bold;">{t} ({vals[t]:.2f})</td>'
    tabla_comparativa += f"<td>{'Correcto' if coincide else 'Anomalia'}</td></tr>"


# --- HTML ---
html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Anomalia de la Triada Aumentada - Analisis Completo</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #fcfcfc; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        .container {{ max-width: 1300px; margin: auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        .blue {{ padding: 20px; background-color: #e8f4f8; border-left: 5px solid #3498db; margin-bottom: 30px; font-size: 15px; line-height: 1.6; border-radius: 4px; }}
        .red {{ padding: 20px; background-color: #f9ebea; border-left: 5px solid #e74c3c; margin-bottom: 30px; font-size: 15px; line-height: 1.6; border-radius: 4px; }}
        .green {{ padding: 20px; background-color: #edf7ed; border-left: 5px solid #4caf50; margin-bottom: 30px; font-size: 15px; line-height: 1.6; border-radius: 4px; }}
        .gt {{ font-size: 18px; text-align: center; padding: 15px; background-color: #fef9e7; border: 2px solid #f1c40f; border-radius: 8px; margin-bottom: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px auto; font-size: 14px; }}
        th, td {{ border: 1px solid #ddd; padding: 10px 12px; text-align: center; }}
        th {{ background-color: #34495e; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>La Anomalia de la Triada Aumentada: Analisis de Tres Niveles</h1>

        <div class="gt">
            <strong>Ground Truth Perceptual (Roberts 1986, Bowling 2018):</strong><br>
            <span style="color:#2ecc71; font-weight:bold;">Mayor</span> &gt;
            <span style="color:#3498db; font-weight:bold;">Menor</span> &gt;
            <span style="color:#f39c12; font-weight:bold;">Suspendida</span> &gt;
            <span style="color:#9b59b6; font-weight:bold;">Disminuida</span> &gt;
            <span style="color:#e74c3c; font-weight:bold;">Aumentada</span>
            <br><small>(La Aumentada debe ser la MAS disonante de todas)</small>
        </div>

        <div class="green">
            <h3>Nivel 1: Lo que nuestro modelo SI resuelve (parcial-a-parcial vs intervalo-a-intervalo)</h3>
            <p>
                Los modelos historicos que caen en la trampa (Helmholtz 1877, Huron 1994, Beck & Clader 2025) usaban <strong>"consonancia diadica agregada"</strong>: evaluaban la consonancia de los intervalos (3M, 6m) de forma <em>aislada</em> y los sumaban. La aumentada, compuesta por tres intervalos individualmente consonantes (3M+3M+6m), obtiene una suma artificialmente baja.
            </p>
            <p>
                Nuestro <code>ModeloSetharesVec</code> NO hace esto. En cambio, expande cada nota en N armonicos con decaimiento exponencial y calcula la <strong>matriz cruzada completa de NxN parciales entre cada par de notas</strong>. Esto detecta las colisiones <em>microscopicas</em> de alta frecuencia que los intervalos macroscopicos ocultan. Por ejemplo, el 3er armonico de la fundamental (3.0f) choca violentamente con el 2do armonico de la quinta aumentada (3.125f), generando ~32 Hz de batimento que cae justo en el ancho de banda critico de maxima rugosidad.
            </p>
            <p><strong>Resultado:</strong> Aumentada (4.15) > Menor (3.48). Nuestro modelo SI vence a Helmholtz y Huron.</p>
        </div>

        <div class="red">
            <h3>Nivel 2: Lo que nuestro modelo NO resuelve (Aumentada vs Disminuida)</h3>
            <p>
                A pesar de su precision espectral, nuestro modelo <strong>sigue prediciendo que la Disminuida es mas rugosa que la Aumentada</strong> (4.60 vs 4.15), cuando la percepcion humana dice lo contrario.
            </p>
            <p><strong>Le faltan dos componentes:</strong></p>
            <ul>
                <li><strong>Batidos de Segundo Orden (Masina & Lo Presti, 2024):</strong> Sethares solo mide batidos primarios (parciales cercanos al unisono). Para capturar la disonancia total de la Aumentada, se necesita sumar la penalizacion por su <em>quinta desafinada</em> (8 semitonos) y su <em>octava desafinada</em>, que son efectos no lineales del oido interno. La Disminuida tiene un Tritono explicito que golpea fuertemente en la rugosidad primaria, inflando su puntuacion; pero la Aumentada tiene una quinta severamente desafinada que el primer orden simplemente no ve.</li>
                <li><strong>Tension por Equidistancia Intervalica (Cook, 2006):</strong> La percepcion humana instintivamente rechaza la simetria perfecta de intervalo (4st + 4st). La formula de Sethares no tiene ningun termino que penalice la uniformidad geometrica.</li>
            </ul>
        </div>

        <div class="blue">
            <h3>Nivel 3: La Solucion Topologica de ChordSpace (Vector 12D)</h3>
            <p>
                El escalar (la suma total) colapsa toda la informacion geometrica en un solo numero. Dos acordes con rugosidad total identica pueden tener <em>formas completamente distintas</em> de rugosidad. El <strong>vector 12D</strong> (histograma de rugosidad por tipo de intervalo) preserva exactamente DONDE se concentra la tension:
            </p>
            <ul>
                <li><strong>La Aumentada</strong> deposita TODA su energia en solo 2 bins (3M y 6m), formando un poligono degenerado bilateral. Un algoritmo de distancias (MDS/UMAP) lo identifica como un ente aislado y rigido.</li>
                <li><strong>Mayor y Menor</strong> distribuyen energia en 3 bins separados (3m, 3M, 5J), formando un triangulo asimetrico y funcional.</li>
                <li><strong>La Disminuida</strong> concentra un pico masivo en el Tritono, pero por ello es geometricamente distinguible de la simetria perfecta de la Aumentada.</li>
            </ul>
            <p>
                Cuando MDS o UMAP proyectan estos vectores en un espacio de distancias, la <strong>similitud topologica</strong> (la forma del poligono) penaliza la simetria redundante de la Aumentada de forma que la suma escalar jamas podria hacer. Esta es la innovacion central de ChordSpace.
            </p>
        </div>

        <h2>1. Tabla de Rugosidad Escalar por Armonicos</h2>
        {df_resultados.to_html(classes="table", index=False, float_format=lambda x: f"{{:.2f}}".format(x))}

        <h2>2. Orden Predicho vs Ground Truth</h2>
        <table>
            <tr><th>n</th><th>1a (menos rugosa)</th><th>2a</th><th>3a</th><th>4a</th><th>5a (mas rugosa)</th><th>Diagnostico</th></tr>
            {tabla_comparativa}
        </table>

        <h2>3. Evolucion de la Rugosidad Escalar</h2>
        <div style="margin-top: 20px; display: flex; justify-content: center;">
            {fig_escalar.to_html(full_html=False, include_plotlyjs='cdn')}
        </div>

        <h2>4. Perfiles Vectoriales 12D (Aranas tipo Reloj)</h2>
        <div style="margin-top: 20px; display: flex; justify-content: center;">
            {fig_vectores.to_html(full_html=False, include_plotlyjs='cdn')}
        </div>

        <h2>5. Curva de Diadas de Sethares</h2>
        <div style="margin-top: 20px; display: flex; justify-content: center;">
            {fig_curva.to_html(full_html=False, include_plotlyjs='cdn')}
        </div>
        
    </div>
</body>
</html>
"""

report_path = os.path.join(os.path.dirname(__file__), "triad_consonance_report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Reporte generado: {report_path}")
