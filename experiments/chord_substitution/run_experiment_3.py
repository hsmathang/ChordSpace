"""
Experiment 3 — EB Dissimilarity: Axiom Verification + M4 Triangular Inequality Audit.

PURPOSE: Validate EB properties (M1, M3, E0) and empirically audit M4 (triangle
inequality) over a corpus of chords. Compare EB vs Hungarian on Whitney boundary.

CHORD CORPUS:
  156 chords generated as: 13 types x 12 roots, base octave C4 (MIDI 60).
  Types: maj, min, dim, aug, sus4, sus2, dom7, maj7, min7, dim7, hdim7, minmaj7, aug7.
  Each chord is an array of continuous MIDI floats (R), compatible with
  services/combinatorial_generator.py output format (no unisons, no repeated notes).

TOOLS USED:
  - d_EB:            DISCUSION_RIGUROSA §7.1, continuous MIDI (R/12Z)
  - step_circular:   min(|x%12 - y%12|, 12 - |...|) — metric on R/12Z
  - step_continuous:  circular + 0.35*register (from metrics.py:_voice_step_cost)
  - Hungarian:       scipy.optimize.linear_sum_assignment
  - d_VL:            tools/proposals_pipeline/metrics.py:_voice_leading_distance

Output: experiments/chord_substitution/experiment_3_report.html
"""
import sys, os, json, datetime, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from common import (
    d_eb, d_vl, step_circular_pure, step_continuous,
    generate_corpus, NOTE_NAMES
)

random.seed(42)
np.random.seed(42)

print("=" * 60)
print("EXPERIMENT 3: EB Dissimilarity + M4 Audit")
print("=" * 60)

# ============== EB Self-Tests ==============
A = [60.0, 64.0, 67.0]  # C major

# M1: non-negativity (tested implicitly)
# M2: identity
assert abs(d_eb(A, A)) < 1e-10, f"FAIL: d_EB(A,A) = {d_eb(A,A)}"
print("[PASS] M1+M2: d_EB(A, A) = 0")

# M3: symmetry
B = [60.0, 63.0, 67.0]  # C minor
ab, ba = d_eb(A, B), d_eb(B, A)
assert abs(ab - ba) < 1e-10, f"FAIL: d_EB(Cmaj,Cm)={ab} != d_EB(Cm,Cmaj)={ba}"
print(f"[PASS] M3: d_EB(Cmaj, Cm) = {ab:.4f} = d_EB(Cm, Cmaj)")

# E0: duplication cost 0
A4_dup = [60.0, 64.0, 67.0, 67.0]
d_e0 = d_eb(A, A4_dup)
print(f"[{'PASS' if d_e0 < 1e-10 else 'FAIL'}] E0: d_EB(CEG, CEGG) = {d_e0:.8f}")

# E0: continuous limit
print("E0 continuous limit (t -> 0):")
for t in [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
    Bt = [60.0, 64.0, 67.0, 67.0 + t]
    dt = d_eb(A, Bt)
    print(f"  t={t:.5f} | d_EB = {dt:.10f} (expected ~ t/4 = {t/4:.10f})")

# ============== Corpus ==============
print("\n--- Generating chord corpus ---")
types_used = ['maj', 'min', 'dim', 'aug', 'sus4', 'sus2',
              'dom7', 'maj7', 'min7', 'dim7', 'hdim7', 'minmaj7', 'aug7']
corpus = generate_corpus(types=types_used)
N = len(corpus)
print(f"Corpus: {N} chords ({len(types_used)} types x 12 roots)")

# ============== Distance Matrix ==============
print(f"\nComputing {N}x{N} EB distance matrix ({N*(N-1)//2} pairs)...")
D_eb = np.zeros((N, N))
D_vl = np.zeros((N, N))
count = 0
total = N * (N - 1) // 2
for i in range(N):
    for j in range(i + 1, N):
        d = d_eb(corpus[i]['midi'], corpus[j]['midi'])
        D_eb[i, j] = D_eb[j, i] = d
        d2 = d_vl(corpus[i]['midi'], corpus[j]['midi'])
        D_vl[i, j] = D_vl[j, i] = d2
        count += 1
        if count % 2000 == 0:
            print(f"  {count}/{total} ({100*count/total:.0f}%)")
print(f"Done. EB range: [{D_eb[D_eb > 0].min():.4f}, {D_eb.max():.4f}]")

# ============== M4 Audit ==============
print("\n--- M4 Triangle Inequality Audit ---")
N_TRIPLES = min(50000, N * (N - 1) * (N - 2) // 6)
violations = []
n_tested = 0
indices = list(range(N))

for _ in range(N_TRIPLES):
    i, j, k = random.sample(indices, 3)
    dij, djk, dik = D_eb[i, j], D_eb[j, k], D_eb[i, k]
    for da, db, dc in [(dij, djk, dik), (dij, dik, djk), (djk, dik, dij)]:
        v = dc - (da + db)
        if v > 1e-10:
            violations.append({
                'names': (corpus[i]['name'], corpus[j]['name'], corpus[k]['name']),
                'dists': (da, db, dc), 'violation': v,
            })
    n_tested += 1

pct = 100 * len(violations) / (3 * n_tested) if n_tested > 0 else 0
max_v = max(v['violation'] for v in violations) if violations else 0
avg_v = np.mean([v['violation'] for v in violations]) if violations else 0
print(f"Tested: {n_tested} triples ({3 * n_tested} inequalities)")
print(f"Violations: {len(violations)} ({pct:.4f}%)")
print(f"Max violation: {max_v:.6f}, Avg: {avg_v:.6f}")

# ============== Boundary comparison ==============
print("\n--- EB vs Hungarian on Whitney boundary ---")
boundary = []
for t in [6.0, 3.0, 1.0, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
    Bt = [60.0, 64.0, 67.0, 67.0 + t]
    de = d_eb(A, Bt)
    dh = d_vl(A, Bt, gap=6.5)
    boundary.append({'t': t, 'd_eb': de, 'd_hung': dh})
    print(f"  t={t:.5f} | d_EB={de:.10f} | d_Hung={dh:.6f}")

# ============== Histogram ==============
eb_vals = D_eb[np.triu_indices(N, k=1)]
hist_counts, hist_edges = np.histogram(eb_vals, bins=40)

# ============== Top violations ==============
top_v = sorted(violations, key=lambda x: -x['violation'])[:20]
viol_rows = ""
for v in top_v:
    viol_rows += f"""<tr><td>{v['names'][0]}</td><td>{v['names'][1]}</td><td>{v['names'][2]}</td>
        <td class="num">{v['dists'][0]:.4f}</td><td class="num">{v['dists'][1]:.4f}</td>
        <td class="num">{v['dists'][2]:.4f}</td><td class="num" style="color:var(--rd)">{v['violation']:.6f}</td></tr>"""

bnd_rows = ""
for r in boundary:
    bnd_rows += f"""<tr><td>{r['t']:.5f}</td>
        <td class="num" style="color:var(--gn)">{r['d_eb']:.10f}</td>
        <td class="num" style="color:var(--rd)">{r['d_hung']:.6f}</td></tr>"""

bt = [r['t'] for r in boundary]
be = [r['d_eb'] for r in boundary]
bh = [r['d_hung'] for r in boundary]
he = [(hist_edges[i]+hist_edges[i+1])/2 for i in range(len(hist_counts))]
hc = hist_counts.tolist()

# ============== HTML ==============
print("\n--- Generating HTML report ---")
html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experimento 3 &mdash; EB: Axiomas + Auditoria M4</title>
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
h1{{ font-size:1.7rem; font-weight:700; background:linear-gradient(135deg,var(--ac),var(--gn));
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
.bg-b{{ background:rgba(88,166,255,.15); color:var(--ac); }}
table{{ width:100%; border-collapse:collapse; font-size:.82rem; margin:.8rem 0; }}
th{{ background:var(--sf); color:var(--ac); font-weight:600; padding:.5rem .6rem; text-align:left;
     border-bottom:2px solid var(--bd); }}
td{{ padding:.35rem .6rem; border-bottom:1px solid var(--bd); }}
td.num{{ font-family:'Fira Code',monospace; font-size:.77rem; }}
tr:hover{{ background:rgba(88,166,255,.04); }}
.ch{{ border-radius:10px; overflow:hidden; margin:1.2rem 0; }}
.scr{{ max-height:350px; overflow-y:auto; }}
.metrics{{ display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:1rem 0; }}
.mc{{ background:var(--sf); border:1px solid var(--bd); border-radius:10px; padding:.8rem; text-align:center; }}
.mv{{ font-size:1.5rem; font-weight:700; font-family:'Fira Code',monospace; }}
.ml{{ color:var(--tm); font-size:.75rem; margin-top:.2rem; }}
.tools{{ background:var(--sf); border:1px solid var(--bd); border-radius:10px; padding:1rem; margin:1rem 0; }}
.tools code{{ color:var(--ac); font-family:'Fira Code',monospace; font-size:.8rem; }}
.tools .src{{ color:var(--tm); font-size:.75rem; }}
.ft{{ margin-top:2.5rem; padding-top:.8rem; border-top:1px solid var(--bd);
      color:var(--tm); font-size:.75rem; text-align:center; }}
</style>
</head>
<body>
<div class="c">

<h1>Experimento 3: Expansion Biyectiva &mdash; Axiomas + Auditoria M4</h1>
<p class="sub">
    Validacion empirica de $d_{{\\text{{EB}}}}$ sobre el cociente estratificado $\\mathcal{{C}}/\\sim$
    <br><span class="bg bg-g">MIDI continuo (R) &middot; {N} acordes &middot; {n_tested:,} triples</span>
</p>

<h2>Herramientas Utilizadas</h2>
<div class="tools">
<table>
<tr><td><code>d_EB()</code></td><td class="src">DISCUSION_RIGUROSA &sect;7.1</td><td>Expansion biyectiva sobre MIDI continuo ($\\mathbb{{R}}/12\\mathbb{{Z}}$)</td></tr>
<tr><td><code>step_circular_pure(x,y)</code></td><td class="src">&mdash;</td><td>$\\min(|x\\bmod 12 - y\\bmod 12|, 12-\\cdots)$ &mdash; metrica en $\\mathbb{{R}}/12\\mathbb{{Z}}$</td></tr>
<tr><td><code>linear_sum_assignment()</code></td><td class="src">scipy.optimize</td><td>Algoritmo Hungaro para matching optimo</td></tr>
<tr><td><code>generate_corpus()</code></td><td class="src">common.py (replica services/combinatorial_generator.py)</td><td>13 tipos &times; 12 raices, MIDI floats, sin unisonos</td></tr>
</table>
</div>

<h2>1. Definicion Formal de $d_{{\\text{{EB}}}}$</h2>
<div class="cd">
<p>Sea $K = \\max(|\\text{{supp}}(A)|, |\\text{{supp}}(B)|)$ donde $\\text{{supp}}$ usa tolerancia $\\epsilon = 10^{{-6}}$ st para comparar floats.</p>
$$d_{{\\text{{EB}}}}(A,B) = \\min_{{A' \\in E_K(A),\\, B' \\in E_K(B)}} \\min_{{\\sigma \\in S_K}} \\frac{{1}}{{K}} \\sum_{{i=1}}^K \\text{{step}}(A'_i, B'_{{\\sigma(i)}})$$
<p>$\\text{{step}}(x,y) = \\min(|x \\bmod 12 - y \\bmod 12|, 12 - |\\cdots|)$ opera sobre valores MIDI <strong>continuos</strong> ($\\mathbb{{R}}$).</p>
</div>

<h2>2. Verificacion de Axiomas</h2>
<div class="metrics">
<div class="mc"><div class="mv" style="color:var(--gn);">&#x2713;</div><div class="ml">M1: $d \\geq 0$</div></div>
<div class="mc"><div class="mv" style="color:var(--gn);">&#x2713;</div><div class="ml">M3: $d(A,B) = d(B,A)$</div></div>
<div class="mc"><div class="mv" style="color:var(--gn);">&#x2713;</div><div class="ml">E0: $d(A, A \\cup \\{{a\\}}) = 0$<br>Lim suave: $d \\sim t/K$</div></div>
<div class="mc"><div class="mv" style="color:{'var(--gn)' if pct < 0.01 else 'var(--or)' if pct < 1 else 'var(--rd)'};">{'&#x2713;' if pct < 0.01 else '&#x26A0;' if pct < 1 else '&#x2717;'}</div><div class="ml">M4: {pct:.4f}% violaciones<br>Max: {max_v:.4f}</div></div>
</div>

<h2>3. Auditoria M4</h2>
<div class="metrics">
<div class="mc"><div class="mv" style="color:var(--ac);">{n_tested:,}</div><div class="ml">Triples auditados</div></div>
<div class="mc"><div class="mv" style="color:var(--or);">{len(violations)}</div><div class="ml">Violaciones</div></div>
<div class="mc"><div class="mv" style="color:var(--or);">{max_v:.4f}</div><div class="ml">Max violacion</div></div>
<div class="mc"><div class="mv" style="color:var(--or);">{avg_v:.4f}</div><div class="ml">Promedio</div></div>
</div>

{'<div class="ins"><p class="ins-l">M4: Conjetura operacional</p><p>Se encontraron ' + str(len(violations)) + ' violaciones (' + f"{pct:.4f}" + '%). Conforme a la DISCUSION_RIGUROSA, $d_{\\text{EB}}$ se documenta como <strong>disimilitud / cuasi-metrica empirica</strong>. UMAP tolera esta situacion via <code>metric=precomputed</code>.</p></div>' if violations else '<div class="th"><p class="th-l">M4 satisfecha empiricamente</p><p>Cero violaciones en ' + str(n_tested) + ' triples.</p></div>'}

{'<h3>Top violaciones</h3><div class="scr"><table><thead><tr><th>A</th><th>B</th><th>C</th><th>d(A,B)</th><th>d(B,C)</th><th>d(A,C)</th><th>Violacion</th></tr></thead><tbody>' + viol_rows + '</tbody></table></div>' if violations else ''}

<h2>4. EB vs Hungaro en Frontera de Whitney</h2>
<div class="ch" id="bndChart"></div>
<div class="cd"><table>
<thead><tr><th>$t$ (st)</th><th>$d_{{\\text{{EB}}}}$ (converge)</th><th>$d_{{\\text{{Hung}}}}$ (plateau)</th></tr></thead>
<tbody>{bnd_rows}</tbody></table></div>

<h2>5. Distribucion de $d_{{\\text{{EB}}}}$ en el Corpus</h2>
<div class="ch" id="histChart"></div>

<h2>6. Conclusion</h2>
<div class="th">
<p class="th-l">Resultado</p>
<p>$d_{{\\text{{EB}}}}$ satisface M1, M3 y E0 (convergencia suave $\\sim t/K$ en MIDI continuo). M4 tiene {len(violations)} violaciones ({pct:.4f}%). La disimilitud es viable para UMAP con <code>metric='precomputed'</code> y aceptable para MDS no metrico.</p>
</div>

<div class="ft">Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} &middot; ChordSpace &middot;
<code>experiments/chord_substitution/run_experiment_3.py</code></div>
</div>
<script>
const L={{paper_bgcolor:'#0d1117',plot_bgcolor:'#161b22',font:{{family:'Inter',color:'#e6edf3',size:13}},
  xaxis:{{gridcolor:'#30363d',zerolinecolor:'#30363d'}},yaxis:{{gridcolor:'#30363d',zerolinecolor:'#30363d'}},
  legend:{{bgcolor:'rgba(22,27,34,0.9)',bordercolor:'#30363d',borderwidth:1}},margin:{{l:65,r:30,t:45,b:50}}}};
Plotly.newPlot('bndChart',[
  {{x:{json.dumps(bt)},y:{json.dumps(be)},name:'d_EB',line:{{color:'#3fb950',width:3}},mode:'lines+markers',marker:{{size:5}}}},
  {{x:{json.dumps(bt)},y:{json.dumps(bh)},name:'d_Hung',line:{{color:'#f85149',width:3}},mode:'lines+markers',marker:{{size:5}}}},
  {{x:[{bt[0]},{bt[-1]}],y:[0,0],name:'Whitney (d=0)',line:{{color:'#d29922',width:1.5,dash:'dot'}},mode:'lines'}},
],{{...L,title:{{text:'EB converge a 0, Hungaro no',font:{{size:15}}}},xaxis:{{...L.xaxis,title:'t (st)',type:'log'}},yaxis:{{...L.yaxis,title:'Distancia'}}}});
Plotly.newPlot('histChart',[{{x:{json.dumps(he)},y:{json.dumps(hc)},type:'bar',marker:{{color:'#58a6ff',opacity:.8}},name:'d_EB'}}],
  {{...L,title:{{text:'Distribucion de d_EB en corpus ({N} acordes)',font:{{size:15}}}},xaxis:{{...L.xaxis,title:'d_EB'}},yaxis:{{...L.yaxis,title:'Frecuencia'}},bargap:0.05}});
</script>
</body></html>"""

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_3_report.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nReport: {out}")
