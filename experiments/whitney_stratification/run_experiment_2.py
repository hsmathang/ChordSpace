"""
Experiment 2 — EB vs Hungarian Asymptotic Analysis at Whitney Boundary.
Demonstrates that the Expansion Biyectiva (EB) respects stratification
continuity (E0) while the Hungarian algorithm does NOT.

CHORD SETUP:
  A    = (60.0, 64.0, 67.0) — C major triad in octave 4
  B(t) = (60.0, 64.0, 67.0, 67.0+t) — parametric tetrad

CONTINUOUS MIDI SPACE (R):
  Following Callender-Quinn-Tymoczko (2008) and Himpel (2022):
  pitch(f) = 12*log2(f/f0), space is R where 1 unit = 1 semitone.
  The parameter t is a continuous differential on R, not an integer.

TOOLS USED:
  - EB dissimilarity: DISCUSION_RIGUROSA §7.1, step on continuous R/12Z
  - Voice Leading:    tools/proposals_pipeline/metrics.py:_voice_leading_distance
  - Step function:    tools/proposals_pipeline/metrics.py:_voice_step_cost
                      (circular fold + 0.35*register penalty, on floats)
  - Hungarian:        scipy.optimize.linear_sum_assignment
  - Sethares Phi:     pre_process.py:ModeloSetharesVec (for d_JSD component)

KEY FINDING: d_EB converges smoothly to 0 as t → 0 (Whitney E0 satisfied).
d_VL (Hungarian) plateaus at 0.25 (E0 violated). The composite d_w inherits
the discontinuity from all three components (d_VL, d_Q5, d_JS).

Output: experiments/whitney_stratification/experiment_2_report.html
"""
import os, sys, json, datetime
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'chord_substitution'))
from common import (
    d_eb, d_vl, d_jsd, d_q5, d_w,
    step_continuous, step_circular_pure,
    quintas_profile, phi_simplex,
    W_VL, W_Q5, W_JS, EPS
)

# ==================== SETUP ====================
A = [60.0, 64.0, 67.0]  # C major triad

T_VALUES = np.concatenate([
    np.arange(6.0, 1.0, -0.5), np.arange(1.0, 0.1, -0.1),
    np.arange(0.1, 0.01, -0.01), np.arange(0.01, 0.001, -0.001),
    np.arange(0.001, 0.0001, -0.0001),
    np.array([0.0001, 0.00005, 0.00001])
])
T_VALUES = np.sort(np.unique(np.round(T_VALUES, 8)))[::-1]

print("=" * 60)
print("EXPERIMENT 2: EB vs Hungarian at Whitney Boundary")
print("=" * 60)
print(f"A = {A} (C major)")
print(f"B(t) = [60, 64, 67, 67+t], t from {T_VALUES[0]} to {T_VALUES[-1]}")
print(f"Step function: continuous circular (R/12Z) + 0.35*register")
print(f"{len(T_VALUES)} values of t\n")

# ==================== SWEEP ====================
results = []
for t in T_VALUES:
    Bt = [60.0, 64.0, 67.0, 67.0 + t]

    # EB with pure circular step (metric on R/12Z)
    de = d_eb(A, Bt, step_fn=step_circular_pure)

    # EB with repo's step (circular + register penalty)
    de_repo = d_eb(A, Bt, step_fn=step_continuous)

    # Hungarian (traditional) — the one that FAILS
    dh = d_vl(A, Bt, step_fn=step_continuous, gap=6.5)

    # Components of d_w
    dq = d_q5(A, Bt)
    dj = d_jsd(A, Bt)
    dw = W_VL * dh + W_Q5 * dq + W_JS * dj

    results.append({
        't': float(t), 'd_eb_pure': de, 'd_eb_repo': de_repo,
        'd_hung': dh, 'd_q5': dq, 'd_jsd': dj, 'd_w': dw,
    })

# Print key values
for r in results:
    if r['t'] in [6.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
        print(f"  t={r['t']:.5f} | d_EB={r['d_eb_pure']:.8f} | "
              f"d_EB_repo={r['d_eb_repo']:.8f} | d_Hung={r['d_hung']:.6f} | "
              f"d_Q5={r['d_q5']:.6f} | d_JS={r['d_jsd']:.6f}")

# Final values
fin = results[-1]
print(f"\nAt t={fin['t']:.5f} (smallest):")
print(f"  d_EB(pure)  = {fin['d_eb_pure']:.10f}")
print(f"  d_EB(repo)  = {fin['d_eb_repo']:.10f}")
print(f"  d_Hung      = {fin['d_hung']:.10f}")

# ==================== HTML ====================
print(f"\n{len(results)} data points. Generating HTML report...")

ct = [r['t'] for r in results]
cep = [r['d_eb_pure'] for r in results]
cer = [r['d_eb_repo'] for r in results]
ch_ = [r['d_hung'] for r in results]
cq = [r['d_q5'] for r in results]
cj = [r['d_jsd'] for r in results]
cw = [r['d_w'] for r in results]

rows_html = ""
for r in results:
    rows_html += f"""<tr><td>{r['t']:.6f}</td>
        <td class="num" style="color:var(--gn)">{r['d_eb_pure']:.8f}</td>
        <td class="num" style="color:var(--gn)">{r['d_eb_repo']:.8f}</td>
        <td class="num" style="color:var(--rd)">{r['d_hung']:.6f}</td>
        <td class="num">{r['d_q5']:.6f}</td>
        <td class="num">{r['d_jsd']:.6f}</td></tr>"""

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experimento 2 &mdash; EB vs Hungaro en Frontera de Whitney</title>
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
h1{{ font-size:1.7rem; font-weight:700; background:linear-gradient(135deg,var(--gn),var(--ac));
     -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:.3rem; }}
.sub{{ color:var(--tm); font-size:.9rem; margin-bottom:2rem; }}
h2{{ font-size:1.2rem; font-weight:600; color:var(--ac); border-bottom:1px solid var(--bd);
     padding-bottom:.4rem; margin:2.5rem 0 1rem; }}
h3{{ font-size:1rem; font-weight:600; color:var(--gn); margin:1.5rem 0 .5rem; }}
p,li{{ font-size:.9rem; }}
.cd{{ background:var(--sf); border:1px solid var(--bd); border-radius:10px; padding:1.2rem; margin:1rem 0; }}
.th{{ border-left:3px solid var(--gn); padding:.8rem 1rem; background:rgba(63,185,80,.06);
      border-radius:0 8px 8px 0; margin:1rem 0; }}
.th-l{{ color:var(--gn); font-weight:700; font-size:.8rem; text-transform:uppercase; letter-spacing:.05em; }}
.rp{{ border-left:3px solid var(--rd); padding:.8rem 1rem; background:rgba(248,81,73,.06);
      border-radius:0 8px 8px 0; margin:1rem 0; }}
.rp-l{{ color:var(--rd); font-weight:700; font-size:.8rem; text-transform:uppercase; letter-spacing:.05em; }}
.ins{{ border-left:3px solid var(--or); padding:.8rem 1rem; background:rgba(210,153,34,.06);
       border-radius:0 8px 8px 0; margin:1rem 0; }}
.ins-l{{ color:var(--or); font-weight:700; font-size:.8rem; text-transform:uppercase; letter-spacing:.05em; }}
.bg{{ display:inline-block; padding:.1rem .5rem; border-radius:12px; font-size:.7rem; font-weight:600; margin-right:.4rem; }}
.bg-g{{ background:rgba(63,185,80,.15); color:var(--gn); }}
.bg-r{{ background:rgba(248,81,73,.15); color:var(--rd); }}
.bg-b{{ background:rgba(88,166,255,.15); color:var(--ac); }}
table{{ width:100%; border-collapse:collapse; font-size:.82rem; margin:.8rem 0; }}
th{{ background:var(--sf); color:var(--ac); font-weight:600; padding:.5rem .6rem; text-align:left;
     border-bottom:2px solid var(--bd); }}
td{{ padding:.35rem .6rem; border-bottom:1px solid var(--bd); }}
td.num{{ font-family:'Fira Code',monospace; font-size:.77rem; }}
tr:hover{{ background:rgba(88,166,255,.04); }}
.ch{{ border-radius:10px; overflow:hidden; margin:1.2rem 0; }}
.scr{{ max-height:400px; overflow-y:auto; }}
.ft{{ margin-top:2.5rem; padding-top:.8rem; border-top:1px solid var(--bd);
      color:var(--tm); font-size:.75rem; text-align:center; }}
.tools{{ background:var(--sf); border:1px solid var(--bd); border-radius:10px; padding:1rem; margin:1rem 0; }}
.tools code{{ color:var(--ac); font-family:'Fira Code',monospace; font-size:.8rem; }}
.tools .src{{ color:var(--tm); font-size:.75rem; }}
.comp{{ display:grid; grid-template-columns:1fr 1fr; gap:1.2rem; margin:1rem 0; }}
.comp .cd{{ margin:0; }}
</style>
</head>
<body>
<div class="c">

<h1>Experimento 2: EB vs Hungaro en la Frontera de Whitney</h1>
<p class="sub">
    Validacion del Axioma E0 (continuidad inter-estratos) para la Expansion Biyectiva
    <br><span class="bg bg-g">EB: CONVERGE A 0 (E0 satisfecho)</span>
    <span class="bg bg-r">HUNGARO: PLATEAU EN 0.25 (E0 violado)</span>
</p>

<h2>Herramientas Utilizadas</h2>
<div class="tools">
<table>
<tr><td><code>d_EB()</code> (Expansion Biyectiva)</td><td class="src">DISCUSION_RIGUROSA &sect;7.1</td><td>Disimilitud sobre cociente $\\mathcal{{C}}/\\sim$, MIDI continuo ($\\mathbb{{R}}$)</td></tr>
<tr><td><code>step_continuous(x,y)</code></td><td class="src">metrics.py:_voice_step_cost</td><td>$\\min(|x \\bmod 12 - y \\bmod 12|, 12 - \\cdots) + 0.35 \\cdot \\text{{registro}}$</td></tr>
<tr><td><code>step_circular_pure(x,y)</code></td><td class="src">&mdash;</td><td>$\\min(|x \\bmod 12 - y \\bmod 12|, 12 - \\cdots)$ &mdash; metrica pura en $\\mathbb{{R}}/12\\mathbb{{Z}}$</td></tr>
<tr><td><code>_voice_leading_distance()</code></td><td class="src">metrics.py:190</td><td>Hungaro con gap $\\gamma = 6.5$, normalizado por $1/(M\\gamma)$</td></tr>
<tr><td><code>_quintas_profile()</code></td><td class="src">metrics.py:129</td><td>Perfil suavizado del circulo de quintas, kernel $(1/4, 1/2, 1/4)$</td></tr>
<tr><td><code>jensenshannon()</code></td><td class="src">scipy.spatial.distance</td><td>$\\sqrt{{\\text{{JSD}}_2}}$ sobre $\\Phi_{{\\text{{simplex}}}}$</td></tr>
</table>
<p style="color:var(--tm); font-size:.78rem; margin-top:.5rem;">Todas las funciones operan sobre MIDI continuo ($\\mathbb{{R}}$). El parametro $t$ es un diferencial real, no entero.</p>
</div>

<h2>1. Configuracion</h2>
<div class="cd">
<p><strong>Acorde A</strong> $= (60.0, 64.0, 67.0)$ &mdash; Do mayor (C4-E4-G4).</p>
<p><strong>Familia B(t)</strong> $= (60.0, 64.0, 67.0, 67.0 + t)$, $t \\in \\mathbb{{R}}^+$. Glissando continuo en el espacio log-frecuencia de Himpel.</p>
<p><strong>Barrido:</strong> {len(results)} valores, $t \\in [{T_VALUES[-1]:.5f}, {T_VALUES[0]:.1f}]$ semitonos.</p>
</div>

<h2>2. Resultado Principal: EB Converge, Hungaro No</h2>

<div class="ch" id="mainChart"></div>

<div class="comp">
<div class="cd">
<h3 style="color:var(--gn);">$d_{{\\text{{EB}}}}$: Axioma E0 satisfecho</h3>
<p>La Expansion Biyectiva encuentra una expansion de costo $\\to 0$ porque duplica una nota de $A$ para igualar la cardinalidad, y el costo de emparejar $67.0 + t$ con $67.0$ es $\\text{{step}}(67+t, 67) = t/K \\to 0$.</p>
$$\\lim_{{t \\to 0^+}} d_{{\\text{{EB}}}}(A, B(t)) = 0 \\quad \\checkmark$$
</div>
<div class="cd">
<h3 style="color:var(--rd);">$d_{{\\text{{Hung}}}}$: Axioma E0 violado</h3>
<p>El Hungaro exige matching biyectivo con gap padding. La voz huerfana siempre paga $\\gamma$, sin importar cuan cerca este del unisono:</p>
$$\\lim_{{t \\to 0^+}} d_{{\\text{{Hung}}}}(A, B(t)) = \\frac{{\\gamma}}{{4\\gamma}} = 0.25 \\neq 0$$
</div>
</div>

<h3>2.2 Zoom logaritmico: convergencia suave de EB</h3>
<div class="ch" id="logChart"></div>

<h3>2.3 Componentes de $d_{{\\mathbf{{w}}}}$ (todas fallan)</h3>
<div class="ch" id="compChart"></div>

<div class="ins">
<p class="ins-l">Las tres componentes de $d_{{\\mathbf{{w}}}}$ tienen residuos no nulos</p>
<p>$d_{{\\text{{VL}}}} \\to 0.25$ (Hungaro), $d_{{\\text{{Q5}}}} \\to {results[-1]['d_q5']:.4f}$ (conteo de notas), $d_{{\\text{{JS}}}} \\to {results[-1]['d_jsd']:.4f}$ ($\\binom{{n}}{{2}}$ pares). Total: $d_{{\\mathbf{{w}}}} \\to {results[-1]['d_w']:.4f} \\neq 0$.</p>
</div>

<h2>3. Tabla de Datos</h2>
<div class="scr"><table>
<thead><tr><th>$t$ (st)</th><th>$d_{{\\text{{EB}}}}$ (circular)</th><th>$d_{{\\text{{EB}}}}$ (repo step)</th><th>$d_{{\\text{{Hung}}}}$</th><th>$d_{{\\text{{Q5}}}}$</th><th>$d_{{\\text{{JSD}}}}$</th></tr></thead>
<tbody>{rows_html}</tbody>
</table></div>

<h2>4. Conclusion</h2>
<div class="th">
<p class="th-l">Resultado principal</p>
<p>La <strong>Expansion Biyectiva</strong> es la primera implementacion computacional del repositorio que respeta la continuidad inter-estratos (E0) en el espacio estratificado de Himpel/Tymoczko. Converge suavemente a 0 en MIDI continuo ($\\mathbb{{R}}$), validando la propiedad clave del cociente $\\mathcal{{C}}/\\sim$.</p>
<p>El Hungaro queda descartado para comparaciones inter-estrato. La metrica compuesta $d_{{\\mathbf{{w}}}}$ hereda la discontinuidad de sus tres componentes.</p>
</div>

<div class="ft">
    Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} &middot; ChordSpace &middot;
    <code>experiments/whitney_stratification/run_experiment_2.py</code>
</div>

</div>
<script>
const t={json.dumps(ct)};
const L={{paper_bgcolor:'#0d1117',plot_bgcolor:'#161b22',font:{{family:'Inter',color:'#e6edf3',size:13}},
  xaxis:{{gridcolor:'#30363d',zerolinecolor:'#30363d'}},yaxis:{{gridcolor:'#30363d',zerolinecolor:'#30363d'}},
  legend:{{bgcolor:'rgba(22,27,34,0.9)',bordercolor:'#30363d',borderwidth:1}},margin:{{l:65,r:30,t:45,b:50}}}};

Plotly.newPlot('mainChart',[
  {{x:t,y:{json.dumps(cep)},name:'d_EB (circular puro)',line:{{color:'#3fb950',width:3}}}},
  {{x:t,y:{json.dumps(cer)},name:'d_EB (step repo)',line:{{color:'#58a6ff',width:2.5}}}},
  {{x:t,y:{json.dumps(ch_)},name:'d_Hung (Hungarian)',line:{{color:'#f85149',width:3}}}},
  {{x:[t[0],t[t.length-1]],y:[0,0],name:'Whitney (d=0)',line:{{color:'#d29922',width:1.5,dash:'dot'}},mode:'lines'}},
],{{...L,title:{{text:'EB vs Hungaro: Convergencia en la Frontera de Whitney',font:{{size:15}}}},
  xaxis:{{...L.xaxis,title:'t (semitonos)',autorange:'reversed'}},
  yaxis:{{...L.yaxis,title:'Distancia'}}}});

const tlog=t.filter(v=>v>0);
Plotly.newPlot('logChart',[
  {{x:tlog,y:{json.dumps(cep)}.filter((_,i)=>t[i]>0),name:'d_EB (circular)',line:{{color:'#3fb950',width:3}},mode:'lines+markers',marker:{{size:4}}}},
  {{x:tlog,y:{json.dumps(cer)}.filter((_,i)=>t[i]>0),name:'d_EB (repo)',line:{{color:'#58a6ff',width:2.5}},mode:'lines+markers',marker:{{size:4}}}},
  {{x:tlog,y:{json.dumps(ch_)}.filter((_,i)=>t[i]>0),name:'d_Hung',line:{{color:'#f85149',width:3}}}},
],{{...L,title:{{text:'Escala log: EB converge suavemente a 0',font:{{size:15}}}},
  xaxis:{{...L.xaxis,title:'t (semitonos)',type:'log',autorange:'reversed'}},
  yaxis:{{...L.yaxis,title:'Distancia'}}}});

Plotly.newPlot('compChart',[
  {{x:t,y:{json.dumps(ch_)},name:'d_VL (Hungaro)',line:{{color:'#f85149',width:2.5}}}},
  {{x:t,y:{json.dumps(cq)},name:'d_Q5 (Quintas)',line:{{color:'#3fb950',width:2.5}}}},
  {{x:t,y:{json.dumps(cj)},name:'d_JS (Rugosidad)',line:{{color:'#d29922',width:2.5}}}},
  {{x:t,y:{json.dumps(cw)},name:'d_w (Compuesta)',line:{{color:'#bc8cff',width:3}}}},
],{{...L,title:{{text:'Componentes de d_w: todas fallan E0',font:{{size:15}}}},
  xaxis:{{...L.xaxis,title:'t (semitonos)',autorange:'reversed'}},
  yaxis:{{...L.yaxis,title:'Distancia'}}}});
</script>
</body></html>"""

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_2_report.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"Report: {out}")
