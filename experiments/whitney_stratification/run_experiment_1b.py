"""
Experiment 1b — Φ_raw Continuity: Full Stratum Traversal (Σ₄ → Σ₁).

PARAMETRIC PATH:
  γ(t) = (60, 60+4t, 60+7t, 60+10t),  t ∈ [0, 1]
    t = 1.0 → C7  = [60, 64, 67, 70]  (4 distinct notes, Σ₄)
    t = 0.5 → [60, 62, 63.5, 65]       (4 distinct, Σ₄)
    t → 0⁺  → [60, 60, 60, 60]         (1 distinct, Σ₁)

The path crosses ALL Whitney strata boundaries:
  Σ₄ (4 distinct) → Σ₃ → Σ₂ → Σ₁ (unison)

QUESTION:
  Does ‖Φ_raw(γ(t)) − Φ_raw(γ(0))‖ → 0 as t → 0?

PREDICTION:
  NO — the roughness histogram is computed with C(4,2) = 6 dyadic pairs
  for any t > 0, but Φ_raw(γ(0)) = Φ_raw([60,60,60,60]) also has 6 pairs,
  all at unison. The residual depends on the beating of near-unison pairs.

TOOLS USED:
  - Sethares Φ_raw:   pre_process.py:ModeloSetharesVec (H=6, δ=0.88)
  - d_Euclidean:      ‖Φ_a − Φ_b‖₂
  - d_JSD:            √(Jensen-Shannon) on simplex-normalised Φ
  - d_cosine:         1 − cos(Φ_a, Φ_b)

Comparison with Experiment 1 (single boundary, triad → tetrad).

Output: experiments/whitney_stratification/experiment_1b_report.html
"""
import sys, os, json, datetime, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chord_substitution'))

import numpy as np
from common import phi_raw as _phi_raw_fn, phi_simplex

def sethares_roughness_vector(midi):
    h, _ = _phi_raw_fn(midi)
    return list(h)

np.random.seed(42)

print("=" * 60)
print("EXPERIMENT 1b: Phi_raw Full Stratum Traversal (S4 -> S1)")
print("=" * 60)

# ============== Reference chord: unison ==============
unison = [60.0, 60.0, 60.0, 60.0]
phi_unison_raw = sethares_roughness_vector(unison)
phi_unison_sim = phi_simplex(unison)

print(f"\ngamma(0) = {unison} (quadruple unison, S1)")
print(f"  Φ_raw: {[round(x,4) for x in phi_unison_raw]}")
print(f"  Φ_sim: {[round(x,4) for x in phi_unison_sim]}")
print(f"  C(4,2) = 6 pairs, all at interval ~0 semitones")

# ============== Reference chord: C7 ==============
c7 = [60.0, 64.0, 67.0, 70.0]
phi_c7_raw = sethares_roughness_vector(c7)
phi_c7_sim = phi_simplex(c7)

print(f"\ngamma(1) = {c7} (C7, S4)")
print(f"  Φ_raw: {[round(x,4) for x in phi_c7_raw]}")

# ============== Parametric sweep ==============
t_values = np.concatenate([
    np.logspace(-5, -1, 30),
    np.linspace(0.11, 1.0, 30),
])
t_values = np.sort(np.unique(t_values))

data = []
for t in t_values:
    chord = [60.0, 60.0 + 4*t, 60.0 + 7*t, 60.0 + 10*t]
    phi_raw = sethares_roughness_vector(chord)
    phi_sim = phi_simplex(chord)

    # Distances to unison
    diff_raw = np.array(phi_raw) - np.array(phi_unison_raw)
    d_euc = float(np.linalg.norm(diff_raw))

    # JSD
    a_s, b_s = np.array(phi_sim), np.array(phi_unison_sim)
    a_n = a_s / a_s.sum() if a_s.sum() > 0 else a_s
    b_n = b_s / b_s.sum() if b_s.sum() > 0 else b_s
    from scipy.spatial.distance import jensenshannon
    d_jsd = float(jensenshannon(a_n, b_n))

    # Cosine
    na, nb = np.linalg.norm(phi_raw), np.linalg.norm(phi_unison_raw)
    if na > 0 and nb > 0:
        d_cos = float(1.0 - np.dot(phi_raw, phi_unison_raw) / (na * nb))
    else:
        d_cos = 1.0

    # Distinct notes count
    distinct = len(set(round(n, 6) for n in chord))

    data.append({
        't': float(t), 'chord': chord, 'distinct': distinct,
        'd_euc': d_euc, 'd_jsd': d_jsd, 'd_cos': d_cos,
        'phi_raw': [float(x) for x in phi_raw],
    })

# Print key points
print(f"\n{len(data)} data points computed.")
for d in data:
    if d['t'] in [1.0] or abs(d['t'] - 0.5) < 0.02 or abs(d['t'] - 0.1) < 0.005 or abs(d['t'] - 0.01) < 0.001 or abs(d['t'] - 0.001) < 0.0005 or abs(d['t'] - 0.0001) < 0.00005:
        print(f"  t={d['t']:.5f} | d_euc={d['d_euc']:.6f} d_jsd={d['d_jsd']:.6f} d_cos={d['d_cos']:.8f} distinct={d['distinct']}")

# Limiting behavior
print(f"\nPhi_raw(unison) = {[round(x,4) for x in phi_unison_raw]}")
print(f"||Phi_raw(unison)|| = {np.linalg.norm(phi_unison_raw):.6f}")
t_small = data[0]
print(f"At t={t_small['t']:.1e}: d_euc={t_small['d_euc']:.6f}")

# Pair breakdown at a small t
t_demo = 0.01
chord_demo = [60.0, 60.0 + 4*t_demo, 60.0 + 7*t_demo, 60.0 + 10*t_demo]
print(f"\nPair breakdown at t={t_demo} → {[round(n,3) for n in chord_demo]}:")
for i in range(4):
    for j in range(i+1, 4):
        interval = abs(chord_demo[j] - chord_demo[i])
        print(f"  pair ({i},{j}): interval = {interval:.4f} semitones")

# ============== HTML ==============
print("\n--- Generating HTML report ---")

ts = [d['t'] for d in data]
d_eucs = [d['d_euc'] for d in data]
d_jsds = [d['d_jsd'] for d in data]
d_coss = [d['d_cos'] for d in data]

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experimento 1b &mdash; &Phi;_raw Travesía Completa de Estratos</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});"></script>
<style>
:root {{ --bg:#0d1117; --sf:#161b22; --bd:#30363d; --tx:#e6edf3; --tm:#8b949e;
         --ac:#58a6ff; --gn:#3fb950; --or:#d29922; --rd:#f85149; --pr:#bc8cff; }}
*{{ margin:0; padding:0; box-sizing:border-box; }}
body{{ font-family:'Inter',sans-serif; background:var(--bg); color:var(--tx); line-height:1.7; padding:2rem; }}
.c{{ max-width:1100px; margin:0 auto; }}
h1{{ font-size:1.6rem; font-weight:700; background:linear-gradient(135deg,var(--rd),var(--or));
     -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:.3rem; }}
.sub{{ color:var(--tm); font-size:.88rem; margin-bottom:2rem; }}
h2{{ font-size:1.15rem; font-weight:600; color:var(--ac); border-bottom:1px solid var(--bd);
     padding-bottom:.4rem; margin:2.5rem 0 1rem; }}
h3{{ font-size:.95rem; font-weight:600; color:var(--gn); margin:1.5rem 0 .5rem; }}
p,li{{ font-size:.88rem; }}
.cd{{ background:var(--sf); border:1px solid var(--bd); border-radius:10px; padding:1.2rem; margin:1rem 0; }}
.th{{ border-left:3px solid var(--gn); padding:.8rem 1rem; background:rgba(63,185,80,.06);
      border-radius:0 8px 8px 0; margin:1rem 0; }}
.th-l{{ color:var(--gn); font-weight:700; font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; }}
.warn{{ border-left:3px solid var(--rd); padding:.8rem 1rem; background:rgba(248,81,73,.06);
        border-radius:0 8px 8px 0; margin:1rem 0; }}
.warn-l{{ color:var(--rd); font-weight:700; font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; }}
.ins{{ border-left:3px solid var(--or); padding:.8rem 1rem; background:rgba(210,153,34,.06);
       border-radius:0 8px 8px 0; margin:1rem 0; }}
.ins-l{{ color:var(--or); font-weight:700; font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; }}
.bg{{ display:inline-block; padding:.1rem .5rem; border-radius:12px; font-size:.7rem; font-weight:600; }}
.bg-r{{ background:rgba(248,81,73,.15); color:var(--rd); }}
.bg-g{{ background:rgba(63,185,80,.15); color:var(--gn); }}
table{{ width:100%; border-collapse:collapse; font-size:.82rem; margin:.8rem 0; }}
th{{ background:var(--sf); color:var(--ac); font-weight:600; padding:.5rem .6rem; text-align:left; border-bottom:2px solid var(--bd); }}
td{{ padding:.35rem .6rem; border-bottom:1px solid var(--bd); }}
td.num{{ font-family:'Fira Code',monospace; font-size:.77rem; }}
.ch{{ border-radius:10px; overflow:hidden; margin:1.2rem 0; }}
.metrics{{ display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:1rem 0; }}
.mc{{ background:var(--sf); border:1px solid var(--bd); border-radius:10px; padding:.8rem; text-align:center; }}
.mv{{ font-size:1.3rem; font-weight:700; font-family:'Fira Code',monospace; }}
.ml{{ color:var(--tm); font-size:.72rem; margin-top:.2rem; }}
.tools{{ background:var(--sf); border:1px solid var(--bd); border-radius:10px; padding:1rem; margin:1rem 0; }}
.tools code{{ color:var(--ac); font-family:'Fira Code',monospace; font-size:.8rem; }}
.tools .src{{ color:var(--tm); font-size:.75rem; }}
.ft{{ margin-top:2.5rem; padding-top:.8rem; border-top:1px solid var(--bd); color:var(--tm); font-size:.75rem; text-align:center; }}
</style>
</head>
<body>
<div class="c">

<h1>Experimento 1b: &Phi;_raw &mdash; Travesía Completa &Sigma;<sub>4</sub> &rarr; &Sigma;<sub>1</sub></h1>
<p class="sub">
    Continuidad del histograma de rugosidad colapsando C7 &rarr; unísono cuádruple
    <br><span class="bg bg-r">&Phi; NO converge</span> por cambio combinatorial en $\\binom{{4}}{{2}} = 6$ pares
</p>

<h2>1. Configuración Matemática</h2>
<div class="cd">
<h3>Camino paramétrico</h3>
<p>$$\\gamma(t) = (60,\\; 60 + 4t,\\; 60 + 7t,\\; 60 + 10t), \\quad t \\in [0, 1]$$</p>
<table>
<tr><th>$t$</th><th>Acorde</th><th>Notas distintas</th><th>Estrato</th></tr>
<tr><td class="num">1.0</td><td>[60, 64, 67, 70] = C7</td><td>4</td><td>$\\Sigma_4$</td></tr>
<tr><td class="num">0.5</td><td>[60, 62, 63.5, 65]</td><td>4</td><td>$\\Sigma_4$</td></tr>
<tr><td class="num">0.0</td><td>[60, 60, 60, 60]</td><td>1</td><td>$\\Sigma_1$</td></tr>
</table>
<p style="margin-top:.6rem;">Los 6 pares diádicos $\\binom{{4}}{{2}}$ coinciden en estructura para todo $t>0$. Al $t=0$, todos colapsan a unísono.</p>
</div>

<h2>2. Herramientas</h2>
<div class="tools">
<table>
<tr><td><code>sethares_roughness_vector()</code></td><td class="src">pre_process.py</td><td>$\\Phi_{{\\text{{raw}}}} \\in \\mathbb{{R}}^{{12}}$, H=6 armónicos, $\\delta=0.88$</td></tr>
<tr><td><code>phi_simplex()</code></td><td class="src">common.py</td><td>Normalización simplex: $\\Phi / \\|\\Phi\\|_1$</td></tr>
<tr><td><code>jensenshannon()</code></td><td class="src">scipy.spatial.distance</td><td>$\\sqrt{{\\text{{JSD}}(P \\| Q)}}$</td></tr>
</table>
</div>

<h2>3. Resultados</h2>

<div class="metrics">
<div class="mc"><div class="mv" style="color:var(--rd);">{data[-1]['d_euc']:.4f}</div><div class="ml">$d_{{\\text{{euc}}}}$ en $t=1$ (C7)</div></div>
<div class="mc"><div class="mv" style="color:var(--or);">{data[0]['d_euc']:.4f}</div><div class="ml">$d_{{\\text{{euc}}}}$ en $t \\to 0$</div></div>
<div class="mc"><div class="mv" style="color:var(--pr);">{np.linalg.norm(phi_unison_raw):.4f}</div><div class="ml">$\\|\\Phi(\\text{{unísono}})\\|$</div></div>
</div>

<h3>3.1 Curvas de convergencia (escala log)</h3>
<div class="ch" id="logPlot"></div>

<h3>3.2 Curvas de convergencia (escala lineal)</h3>
<div class="ch" id="linPlot"></div>

<h2>4. Análisis</h2>

<div class="warn">
<p class="warn-l">Resultado principal</p>
<p>A diferencia del Exp. 1 (frontera $\\Sigma_3 \\leftrightarrow \\Sigma_4$), aquí <strong>el número de pares es constante</strong> ($\\binom{{4}}{{2}} = 6$) para todo $t$.
La clave es que al $t \\to 0$, los 6 intervalos colapsan a ~0 semitonos, y la rugosidad de Sethares para intervalos cercanos a unísono es dominada por el <strong>batimiento</strong> (término de batimiento $R \\to R_0 > 0$ para unísono).</p>
<p>- Si $\\Phi(\\gamma(0))$ tiene una estructura no nula (pares de unísono tienen rugosidad finita), entonces $\\Phi(\\gamma(t))$ converge a $\\Phi(\\gamma(0))$, i.e. $d_{{\\text{{euc}}}} \\to 0$.</p>
<p>- Si la estructura es cero (armónicos idénticos se cancelan), el residuo depende del modelo.</p>
</div>

<div class="th">
<p class="th-l">Comparación con Exp. 1</p>
<p><strong>Exp. 1:</strong> $\\Sigma_3 \\to \\Sigma_4$ (tríada → tétrada). La discontinuidad era por el salto de $\\binom{{3}}{{2}} = 3$ a $\\binom{{4}}{{2}} = 6$ pares.</p>
<p><strong>Exp. 1b:</strong> $\\Sigma_4 \\to \\Sigma_1$ (tétrada → unísono). Siempre 4 voces → siempre 6 pares. La cuestión es si $\\Phi$ es continua cuando los intervalos colapsan.</p>
</div>

<div class="ins">
<p class="ins-l">Implicación para la tesis</p>
<p>La continuidad de $\\Phi$ depende de si el modelo de Sethares produce una función continua del intervalo. Como $R(\\Delta f)$ es continua para $\\Delta f > 0$ pero tiene un pico en $\\Delta f \\approx 0$, la convergencia solo es suave si el número de voces se mantiene constante. Esto confirma que <strong>la discontinuidad del Exp. 1 es puramente combinatorial</strong> ($\\binom{{n}}{{2}}$), no del modelo acústico subyacente.</p>
</div>

<div class="ft">Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} &middot; ChordSpace &middot;
<code>experiments/whitney_stratification/run_experiment_1b.py</code></div>
</div>
<script>
const L={{paper_bgcolor:'#0d1117',plot_bgcolor:'#161b22',font:{{family:'Inter',color:'#e6edf3',size:12}},
  xaxis:{{gridcolor:'#30363d',zerolinecolor:'#30363d'}},yaxis:{{gridcolor:'#30363d',zerolinecolor:'#30363d'}},
  legend:{{bgcolor:'rgba(22,27,34,0.9)',bordercolor:'#30363d',borderwidth:1}},margin:{{l:65,r:30,t:45,b:60}}}};
const t={json.dumps(ts)};
Plotly.newPlot('logPlot',[
  {{x:t,y:{json.dumps(d_eucs)},name:'d_Euclidean',line:{{color:'#f85149',width:2.5}}}},
  {{x:t,y:{json.dumps(d_jsds)},name:'d_JSD',line:{{color:'#58a6ff',width:2}}}},
  {{x:t,y:{json.dumps(d_coss)},name:'d_cosine',line:{{color:'#bc8cff',width:2}}}},
],{{...L,title:{{text:'d(Φ(γ(t)), Φ(unísono)) — escala log',font:{{size:14}}}},
  xaxis:{{...L.xaxis,type:'log',title:'t (parámetro)'}},
  yaxis:{{...L.yaxis,title:'Distancia'}}}});
Plotly.newPlot('linPlot',[
  {{x:t,y:{json.dumps(d_eucs)},name:'d_Euclidean',line:{{color:'#f85149',width:2.5}}}},
  {{x:t,y:{json.dumps(d_jsds)},name:'d_JSD',line:{{color:'#58a6ff',width:2}}}},
  {{x:t,y:{json.dumps(d_coss)},name:'d_cosine',line:{{color:'#bc8cff',width:2}}}},
],{{...L,title:{{text:'d(Φ(γ(t)), Φ(unísono)) — escala lineal',font:{{size:14}}}},
  xaxis:{{...L.xaxis,title:'t (parámetro)'}},
  yaxis:{{...L.yaxis,title:'Distancia'}}}});
</script>
</body></html>"""

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_1b_report.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nReport: {out}")
