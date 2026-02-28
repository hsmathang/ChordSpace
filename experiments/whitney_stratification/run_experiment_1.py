"""
Experiment 1 — Asymptotic Continuity of the Roughness Feature Space Phi in R^12
at the Whitney Stratum Boundary C_3 → C_4.

CHORD SETUP:
  A  = (60, 64, 67) — C major triad (C4-E4-G4), generated as valid chord
       per services/combinatorial_generator.py (alphabet C,E,G, octave 4).
  B(t) = (60, 64, 67, 67+t) — parametric tetrad, 4th voice glissandos from
       G4+t down to G4 (unison with 3rd voice) as t → 0.

TOOLS USED (from ChordSpace repo):
  - Sethares roughness model: pre_process.py:ModeloSetharesVec (H=6, delta=0.88)
  - Roughness histogram:      pre_process.py:interval_to_ui_bin (12-bin mapping)
  - Metrics:                   Euclidean, JSD (scipy), Hellinger, Cosine on Phi
  - Continuous MIDI space:     Himpel (2022) / Callender-Quinn-Tymoczko (2008)
                               pitch(f) = 12*log2(f/f0), 1 unit = 1 semitone

KEY FINDING: Phi_raw is NOT stratum-continuous. The residual d > 0 as t → 0
arises because Phi aggregates over C(n,2) dyadic pairs, and C(4,2)=6 > C(3,2)=3.
The pair-level roughness R(f_i, f_j) IS continuous (Sethares).

Output: experiments/whitney_stratification/experiment_1_report.html
"""
import os, sys, json, datetime
import numpy as np

# Use shared module from chord_substitution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'chord_substitution'))
from common import phi_raw, phi_simplex, midi_to_freq, _pair_roughness, _bin, EPS
from scipy.spatial.distance import jensenshannon

# ==================== EXPERIMENT SETUP ====================
# Chord A: C major triad in octave 4
# Valid per combinatorial_generator.py: alphabet={C,E,G}, octave=4
A_MIDI = [60.0, 64.0, 67.0]  # C4=60, E4=64, G4=67
A_FREQS = [midi_to_freq(n) for n in A_MIDI]

# Parametric family B(t) with t in continuous MIDI semitones
T_VALUES = np.concatenate([
    np.arange(6.0, 1.0, -0.5), np.arange(1.0, 0.1, -0.1),
    np.arange(0.1, 0.01, -0.01), np.arange(0.01, 0.001, -0.001),
    np.array([0.001, 0.0005, 0.0001])
])
T_VALUES = np.sort(np.unique(np.round(T_VALUES, 6)))[::-1]

print("=" * 60)
print("EXPERIMENT 1: Phi_raw Continuity at Whitney Boundary")
print("=" * 60)
print(f"Chord A = {A_MIDI} (C major, octave 4)")
print(f"B(t) = [60, 64, 67, 67+t], t in [{T_VALUES[-1]}, {T_VALUES[0]}] semitones")

# Compute Phi(A) — 3 pairs
phi_A, total_A = phi_raw(A_MIDI)
p_A = phi_simplex(A_MIDI)
print(f"\nPhi(A) [{3} pairs, C(3,2)=3]:")
print(f"  raw:     {np.round(phi_A, 4).tolist()}")
print(f"  simplex: {np.round(p_A, 4).tolist()}")

# Compute limit Phi(B(0)) — exact unison, 6 pairs
B0_MIDI = [60.0, 64.0, 67.0, 67.0]
phi_B0, total_B0 = phi_raw(B0_MIDI)
p_B0 = phi_simplex(B0_MIDI)
print(f"\nPhi(B(0)) [{6} pairs, C(4,2)=6, 4th=unison of 3rd]:")
print(f"  raw:     {np.round(phi_B0, 4).tolist()}")
print(f"  simplex: {np.round(p_B0, 4).tolist()}")

# Theoretical limits
d_euc_lim = float(np.linalg.norm(phi_A - phi_B0))
d_jsd_lim = float(jensenshannon(p_A, p_B0, base=2.0))
d_hel_lim = float(np.linalg.norm(np.sqrt(p_A) - np.sqrt(p_B0)) / np.sqrt(2.0))
d_cos_lim = float(1.0 - np.dot(phi_A, phi_B0) /
                   (np.linalg.norm(phi_A) * np.linalg.norm(phi_B0) + EPS))
print(f"\nTheoretical limits (t=0 exact):")
print(f"  d_euc={d_euc_lim:.6f}  d_jsd={d_jsd_lim:.6f}  "
      f"d_hel={d_hel_lim:.6f}  d_cos={d_cos_lim:.8f}")

# ==================== SWEEP ====================
results = []
for t in T_VALUES:
    Bt = [60.0, 64.0, 67.0, 67.0 + t]
    phi_B, total_B = phi_raw(Bt)
    p_B = phi_simplex(Bt)

    d_euclidean = float(np.linalg.norm(phi_A - phi_B))
    d_jsd_v = float(jensenshannon(p_A, p_B, base=2.0))
    d_hellinger = float(np.linalg.norm(np.sqrt(p_A) - np.sqrt(p_B)) / np.sqrt(2.0))
    d_cosine = float(1.0 - np.dot(phi_A, phi_B) /
                     (np.linalg.norm(phi_A) * np.linalg.norm(phi_B) + EPS))

    results.append({
        't': float(t), 'd_euc': d_euclidean, 'd_jsd': d_jsd_v,
        'd_hel': d_hellinger, 'd_cos': d_cosine, 'total_B': float(total_B),
    })

# Print key values
for r in results:
    if r['t'] in [6.0, 1.0, 0.1, 0.01, 0.001, 0.0001]:
        print(f"  t={r['t']:.4f} | d_euc={r['d_euc']:.6f} d_jsd={r['d_jsd']:.6f} "
              f"d_hel={r['d_hel']:.6f} d_cos={r['d_cos']:.8f}")

# Pair breakdown for B(0) — explain why Phi(B(0)) != Phi(A)
print(f"\nPair breakdown B(0) = [60, 64, 67, 67]:")
B0_freqs = sorted(midi_to_freq(n) for n in B0_MIDI)
B0_st = [0.0] + [12.0 * np.log2(B0_freqs[i] / B0_freqs[0]) for i in range(1, 4)]
pair_info = []
for i in range(3):
    for j in range(i + 1, 4):
        iv = int(round(B0_st[j] - B0_st[i])) % 12
        r = _pair_roughness(B0_freqs[i], B0_freqs[j])
        in_A = "Yes" if (i < 3 and j < 3) else "No (extra)"
        pair_info.append((i, j, iv, r, in_A))
        print(f"  pair ({i},{j}): interval={iv} st, R={r:.6f}, in A? {in_A}")

# ==================== HTML REPORT ====================
print(f"\n{len(results)} data points. Generating HTML report...")

rows_html = ""
for r in results:
    rows_html += f"""<tr><td>{r['t']:.4f}</td>
        <td class="num">{r['d_euc']:.6f}</td><td class="num">{r['d_jsd']:.6f}</td>
        <td class="num">{r['d_hel']:.6f}</td><td class="num">{r['d_cos']:.8f}</td></tr>"""

pair_html = ""
for i, j, iv, r, inA in pair_info:
    pair_html += f"<tr><td>({i},{j})</td><td>{iv} st</td><td class='num'>{r:.6f}</td><td>{inA}</td></tr>"

ct = [r['t'] for r in results]
ce = [r['d_euc'] for r in results]
cj = [r['d_jsd'] for r in results]
ch = [r['d_hel'] for r in results]
cc = [r['d_cos'] for r in results]

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experimento 1 &mdash; Continuidad de Phi_raw en Frontera de Whitney</title>
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
h1{{ font-size:1.7rem; font-weight:700; background:linear-gradient(135deg,var(--or),var(--rd));
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
.lim{{ background:rgba(248,81,73,.08); border:1px dashed var(--rd); border-radius:10px;
       padding:1rem; margin:1.2rem 0; text-align:center; }}
.lim-v{{ font-size:1.5rem; font-weight:700; color:var(--rd); font-family:'Fira Code',monospace; }}
.ft{{ margin-top:2.5rem; padding-top:.8rem; border-top:1px solid var(--bd);
      color:var(--tm); font-size:.75rem; text-align:center; }}
.tools{{ background:var(--sf); border:1px solid var(--bd); border-radius:10px; padding:1rem; margin:1rem 0; }}
.tools code{{ color:var(--ac); font-family:'Fira Code',monospace; font-size:.8rem; }}
.tools .src{{ color:var(--tm); font-size:.75rem; }}
</style>
</head>
<body>
<div class="c">

<h1>Experimento 1: Continuidad de $\\Phi_{{\\text{{raw}}}} \\in \\mathbb{{R}}^{{12}}$ en la Frontera de Whitney</h1>
<p class="sub">
    Evaluacion en la frontera $\\mathcal{{C}}_3 \\to \\mathcal{{C}}_4$ del espacio estratificado de acordes
    <br><span class="bg bg-r">DISCONTINUIDAD COMBINATORIA DETECTADA</span>
    <span class="bg bg-b">MIDI continuo (R) &middot; Sethares H=6, &delta;=0.88</span>
</p>

<h2>Herramientas Utilizadas</h2>
<div class="tools">
<table>
<tr><td><code>ModeloSetharesVec.calcular()</code></td><td class="src">pre_process.py:508&ndash;586</td><td>Vector de rugosidad $\\Phi \\in \\mathbb{{R}}^{{12}}$ via $H=6$ armonicos, $\\delta=0.88$</td></tr>
<tr><td><code>interval_to_ui_bin()</code></td><td class="src">pre_process.py:88</td><td>Mapeo de intervalo a bin del histograma 12-D</td></tr>
<tr><td><code>midi_to_freq()</code></td><td class="src">pre_process.py</td><td>MIDI continuo (R) &rarr; Hz: $f = 440 \\cdot 2^{{(n-69)/12}}$</td></tr>
<tr><td><code>jensenshannon()</code></td><td class="src">scipy.spatial.distance</td><td>$\\sqrt{{\\text{{JSD}}_2}}$, base 2</td></tr>
</table>
<p style="color:var(--tm); font-size:.78rem; margin-top:.5rem;">Espacio: MIDI continuo ($\\mathbb{{R}}$), 1 unidad = 1 semitono. Compatible con Callender-Quinn-Tymoczko (2008) y Himpel (2022).</p>
</div>

<h2>1. Configuracion</h2>
<div class="cd">
<p><strong>Acorde A</strong> $= (60.0, 64.0, 67.0)$ &mdash; Do mayor (C4-E4-G4). Triada valida en el generador combinatorial (<code>services/combinatorial_generator.py</code>): alfabeto $\\{{C, E, G\\}}$, octava 4. Tiene $\\binom{{3}}{{2}} = 3$ pares diadicos.</p>
<p><strong>Familia B(t)</strong> $= (60.0, 64.0, 67.0, 67.0 + t)$, $t \\in (0, 6]$ semitonos. El parametro $t$ es un diferencial <strong>continuo</strong> sobre $\\mathbb{{R}}$ (glissando microtonal). Tiene $\\binom{{4}}{{2}} = 6$ pares.</p>
<p><strong>Barrido:</strong> {len(results)} valores de $t$ desde 6.0 hasta 0.0001 semitonos.</p>
</div>

<h2>2. Resultados</h2>

<lim>
<p style="color:var(--tm); margin-bottom:.3rem;">Residuos asintoticos (calculados con $t = 0$ exacto):</p>
<p class="lim-v">$d_{{\\text{{euc}}}} = {d_euc_lim:.4f}$ &emsp; $d_{{\\text{{JSD}}}} = {d_jsd_lim:.4f}$ &emsp; $d_H = {d_hel_lim:.4f}$ &emsp; $d_{{\\cos}} = {d_cos_lim:.4f}$</p>
</div>

<h3>2.1 Convergencia al residuo (4 metricas)</h3>
<div class="ch" id="mainChart"></div>

<h3>2.2 Escala logaritmica</h3>
<div class="ch" id="logChart"></div>

<h3>2.3 Tabla de datos</h3>
<div class="scr"><table>
<thead><tr><th>$t$ (st)</th><th>$d_{{\\text{{euc}}}}$</th><th>$d_{{\\text{{JSD}}}}$</th><th>$d_H$</th><th>$d_{{\\cos}}$</th></tr></thead>
<tbody>{rows_html}</tbody>
</table></div>

<h2>3. Diagnostico: Por que $\\Phi$ NO converge a cero</h2>

<div class="rp">
<p class="rp-l">Discontinuidad combinatoria del mapa de caracteristicas</p>
<p>$\\Phi_{{\\text{{raw}}}}$ agrega rugosidad sobre <strong>todos</strong> los $\\binom{{n}}{{2}}$ pares. La diferencia entre estratos: $\\binom{{4}}{{2}} - \\binom{{3}}{{2}} = 6 - 3 = 3$ pares adicionales.</p>
</div>

<h3>Desglose par-a-par de B(0) = [60, 64, 67, 67]</h3>
<div class="cd">
<table><thead><tr><th>Par</th><th>Intervalo</th><th>Rugosidad</th><th>&iquest;Existe en A?</th></tr></thead>
<tbody>{pair_html}</tbody></table>
<p style="margin-top:.6rem;">Los pares extra $(0,3)$ y $(1,3)$ duplican asimetricamente los bins de $\\Phi$, cambiando la <strong>forma</strong> del histograma incluso tras normalizacion simplex.</p>
</div>

<div class="ins">
<p class="ins-l">Implicacion</p>
<p>La rugosidad par-a-par $R(f_i, f_j)$ si es continua (Sethares). La discontinuidad viene de la <strong>agregacion combinatoria</strong> $\\binom{{n}}{{2}}$, que cambia discretamente con la cardinalidad. Las comparaciones intra-estrato ($|A| = |B|$) son topologicamente correctas.</p>
</div>

<div class="ft">
    Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} &middot; ChordSpace &middot;
    <code>experiments/whitney_stratification/run_experiment_1.py</code>
</div>

</div>
<script>
const t={json.dumps(ct)}, de={json.dumps(ce)}, dj={json.dumps(cj)}, dh={json.dumps(ch)}, dc={json.dumps(cc)};
const L={{paper_bgcolor:'#0d1117',plot_bgcolor:'#161b22',font:{{family:'Inter',color:'#e6edf3',size:13}},
  xaxis:{{gridcolor:'#30363d',zerolinecolor:'#30363d'}},yaxis:{{gridcolor:'#30363d',zerolinecolor:'#30363d'}},
  legend:{{bgcolor:'rgba(22,27,34,0.9)',bordercolor:'#30363d',borderwidth:1}},margin:{{l:65,r:30,t:45,b:50}}}};
const tr=[{{x:t,y:de,name:'Euclidiana',line:{{color:'#58a6ff',width:2.5}}}},
  {{x:t,y:dj,name:'JSD',line:{{color:'#3fb950',width:2.5}}}},
  {{x:t,y:dh,name:'Hellinger',line:{{color:'#d29922',width:2.5}}}},
  {{x:t,y:dc,name:'Coseno',line:{{color:'#bc8cff',width:2.5}}}}];
const lims=[{{x:[t[0],t[t.length-1]],y:[{d_euc_lim},{d_euc_lim}],name:'Lim euc',line:{{color:'#58a6ff',width:1.5,dash:'dash'}},mode:'lines'}},
  {{x:[t[0],t[t.length-1]],y:[{d_jsd_lim},{d_jsd_lim}],name:'Lim JSD',line:{{color:'#3fb950',width:1.5,dash:'dash'}},mode:'lines'}}];
Plotly.newPlot('mainChart',[...tr,...lims],{{...L,title:{{text:'Convergencia al residuo asintotico',font:{{size:15}}}},
  xaxis:{{...L.xaxis,title:'t (semitonos)',autorange:'reversed'}},yaxis:{{...L.yaxis,title:'Distancia'}}}});
Plotly.newPlot('logChart',tr.map(x=>({{...x,x:x.x.filter(v=>v>0),y:x.y.filter((_,i)=>x.x[i]>0)}})),
  {{...L,title:{{text:'Escala log: convergencia suave al residuo',font:{{size:15}}}},
  xaxis:{{...L.xaxis,title:'log(t)',type:'log',autorange:'reversed'}},yaxis:{{...L.yaxis,title:'Distancia'}}}});
</script>
</body></html>"""

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_1_report.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"Report: {out}")
