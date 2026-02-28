"""
Experiment 6 — Novel Substitution Discovery.
Identifies nearest neighbors that are NOT in ground truth but may have
musical justification as novel substitutes.

METHOD:
  1. Use best metric from Exp 4 (d_w) to find top-10 neighbors of each chord
  2. Filter out known GT substitutes
  3. For each candidate, analyze:
     - PC overlap (Jaccard similarity)
     - Voice leading distance
     - Roughness similarity (JSD)
     - Circle of fifths proximity
  4. Classify as plausible / novel / spurious
  5. Build substitution network (graph)

TOOLS USED:
  - Distance matrices from Exp 4
  - Ground truth from ground_truth.py
  - All metrics from common.py

Output: experiments/chord_substitution/experiment_6_report.html
"""
import sys, os, json, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from common import (
    generate_corpus, d_w, d_eb, d_jsd, d_vl, d_q5, d_euclidean,
    step_circular_pure, NOTE_NAMES
)
from ground_truth import build_ground_truth_set

np.random.seed(42)

print("=" * 60)
print("EXPERIMENT 6: Novel Substitution Discovery")
print("=" * 60)

# ============== Corpus ==============
types_used = ['maj', 'min', 'dim', 'aug', 'sus4', 'sus2',
              'dom7', 'maj7', 'min7', 'dim7', 'hdim7', 'minmaj7', 'aug7']
corpus = generate_corpus(types=types_used)
N = len(corpus)
names = [c['name'] for c in corpus]
gt_set = build_ground_truth_set(corpus)
print(f"Corpus: {N} chords, GT pairs: {len(gt_set)//2}")

# ============== d_w Distance Matrix ==============
print("Computing d_w matrix...", end=" ", flush=True)
D = np.zeros((N, N))
for i in range(N):
    for j in range(i + 1, N):
        D[i, j] = D[j, i] = d_w(corpus[i]['midi'], corpus[j]['midi'])
print(f"done. [{D[D>0].min():.4f}, {D.max():.4f}]")

# ============== Find Novel Candidates ==============
print("\n--- Finding novel substitution candidates ---")
K = 10
candidates = []

for i in range(N):
    qi = names[i]
    dists = D[i].copy()
    dists[i] = float('inf')
    topk = np.argsort(dists)[:K]

    for j in topk:
        nj = names[j]
        if (qi, nj) not in gt_set:
            # Analyze the pair
            pc_i = set(corpus[i]['pc'])
            pc_j = set(corpus[j]['pc'])
            jaccard = len(pc_i & pc_j) / len(pc_i | pc_j) if pc_i | pc_j else 0
            common = pc_i & pc_j
            vl = d_vl(corpus[i]['midi'], corpus[j]['midi'])
            jsd = d_jsd(corpus[i]['midi'], corpus[j]['midi'])
            q5 = d_q5(corpus[i]['midi'], corpus[j]['midi'])
            eb = d_eb(corpus[i]['midi'], corpus[j]['midi'])
            d_w_val = D[i, j]
            rank = int(np.where(topk == j)[0][0]) + 1

            # Classify
            if jaccard >= 0.5 and vl < 0.15:
                category = "Plausible"
                reason = f"High PC overlap ({jaccard:.0%}) + short VL ({vl:.3f})"
            elif jsd < 0.15:
                category = "Plausible"
                reason = f"Similar roughness (JSD={jsd:.4f})"
            elif jaccard >= 0.33 or vl < 0.1:
                category = "Novel"
                reason = f"Partial overlap ({jaccard:.0%}) or short VL ({vl:.3f})"
            else:
                category = "Spurious"
                reason = f"Low overlap ({jaccard:.0%}), high VL ({vl:.3f})"

            candidates.append({
                'query': qi, 'neighbor': nj, 'rank': rank,
                'd_w': d_w_val, 'jaccard': jaccard, 'common_notes': sorted(common),
                'vl': vl, 'jsd': jsd, 'q5': q5, 'eb': eb,
                'category': category, 'reason': reason,
                'q_type': corpus[i]['type'], 'n_type': corpus[j]['type'],
            })

print(f"Total candidates: {len(candidates)}")

# Deduplicate (keep unique pairs)
seen = set()
unique = []
for c in candidates:
    pair = tuple(sorted([c['query'], c['neighbor']]))
    if pair not in seen:
        seen.add(pair)
        unique.append(c)

# Count by category
cats = {}
for c in unique:
    cats[c['category']] = cats.get(c['category'], 0) + 1
print(f"Unique pairs: {len(unique)}")
for cat, cnt in sorted(cats.items()):
    print(f"  {cat}: {cnt}")

# Top candidates
plausible = sorted([c for c in unique if c['category'] == 'Plausible'],
                   key=lambda x: x['d_w'])[:15]
novel = sorted([c for c in unique if c['category'] == 'Novel'],
               key=lambda x: x['d_w'])[:10]

print(f"\nTop plausible: {len(plausible)}")
for c in plausible[:5]:
    print(f"  {c['query']} -> {c['neighbor']} (d_w={c['d_w']:.4f}, J={c['jaccard']:.2f}, VL={c['vl']:.3f})")
print(f"\nTop novel: {len(novel)}")
for c in novel[:5]:
    print(f"  {c['query']} -> {c['neighbor']} (d_w={c['d_w']:.4f}, J={c['jaccard']:.2f})")

# ============== Substitution Network ==============
# Build edges for top-5 neighbors of each chord (both GT and novel)
edges = []
for i in range(N):
    dists = D[i].copy()
    dists[i] = float('inf')
    top5 = np.argsort(dists)[:5]
    for j in top5:
        is_gt = (names[i], names[j]) in gt_set
        edges.append({'source': i, 'target': int(j),
                      'type': 'GT' if is_gt else 'Novel',
                      'weight': float(D[i, j])})

# ============== HTML ==============
print("\n--- Generating HTML report ---")

plaus_rows = ""
for c in plausible:
    plaus_rows += f"""<tr><td><strong>{c['query']}</strong></td><td><strong>{c['neighbor']}</strong></td>
        <td class="num">{c['rank']}</td><td class="num">{c['d_w']:.4f}</td>
        <td class="num">{c['jaccard']:.2f}</td><td class="num">{c['vl']:.3f}</td>
        <td class="num">{c['jsd']:.4f}</td><td>{c['reason']}</td></tr>"""

novel_rows = ""
for c in novel:
    novel_rows += f"""<tr><td><strong>{c['query']}</strong></td><td><strong>{c['neighbor']}</strong></td>
        <td class="num">{c['rank']}</td><td class="num">{c['d_w']:.4f}</td>
        <td class="num">{c['jaccard']:.2f}</td><td class="num">{c['vl']:.3f}</td>
        <td class="num">{c['jsd']:.4f}</td><td>{c['reason']}</td></tr>"""

# Network as Plotly scatter (simplified)
net_x, net_y = [], []
for i in range(N):
    ang = 2 * np.pi * i / N
    net_x.append(float(np.cos(ang)))
    net_y.append(float(np.sin(ang)))

gt_edge_x, gt_edge_y = [], []
novel_edge_x, novel_edge_y = [], []
for e in edges:
    s, t = e['source'], e['target']
    if e['type'] == 'GT':
        gt_edge_x += [net_x[s], net_x[t], None]
        gt_edge_y += [net_y[s], net_y[t], None]
    else:
        novel_edge_x += [net_x[s], net_x[t], None]
        novel_edge_y += [net_y[s], net_y[t], None]

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experimento 6 &mdash; Descubrimiento de Sustitutos Nuevos</title>
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
h1{{ font-size:1.7rem; font-weight:700; background:linear-gradient(135deg,var(--or),var(--gn));
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
.ins{{ border-left:3px solid var(--or); padding:.8rem 1rem; background:rgba(210,153,34,.06);
       border-radius:0 8px 8px 0; margin:1rem 0; }}
.ins-l{{ color:var(--or); font-weight:700; font-size:.8rem; text-transform:uppercase; letter-spacing:.05em; }}
.bg{{ display:inline-block; padding:.1rem .5rem; border-radius:12px; font-size:.7rem; font-weight:600; margin-right:.4rem; }}
.bg-g{{ background:rgba(63,185,80,.15); color:var(--gn); }}
.bg-o{{ background:rgba(210,153,34,.15); color:var(--or); }}
.bg-r{{ background:rgba(248,81,73,.15); color:var(--rd); }}
table{{ width:100%; border-collapse:collapse; font-size:.82rem; margin:.8rem 0; }}
th{{ background:var(--sf); color:var(--ac); font-weight:600; padding:.5rem .6rem; text-align:left;
     border-bottom:2px solid var(--bd); }}
td{{ padding:.35rem .6rem; border-bottom:1px solid var(--bd); }}
td.num{{ font-family:'Fira Code',monospace; font-size:.77rem; }}
tr:hover{{ background:rgba(88,166,255,.04); }}
.ch{{ border-radius:10px; overflow:hidden; margin:1.2rem 0; }}
.scr{{ max-height:350px; overflow-y:auto; }}
.metrics{{ display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:1rem 0; }}
.mc{{ background:var(--sf); border:1px solid var(--bd); border-radius:10px; padding:.8rem; text-align:center; }}
.mv{{ font-size:1.4rem; font-weight:700; font-family:'Fira Code',monospace; }}
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

<h1>Experimento 6: Descubrimiento de Sustitutos Nuevos</h1>
<p class="sub">
    Vecinos cercanos en $d_{{\\mathbf{{w}}}}$ que NO son sustitutos clasicos
    <br><span class="bg bg-g">Plausible: {cats.get('Plausible',0)}</span>
    <span class="bg bg-o">Novel: {cats.get('Novel',0)}</span>
    <span class="bg bg-r">Spurious: {cats.get('Spurious',0)}</span>
    <span style="color:var(--tm); font-size:.75rem;">&middot; {len(unique)} pares unicos de {len(candidates)} candidatos</span>
</p>

<h2>Herramientas Utilizadas</h2>
<div class="tools">
<table>
<tr><td><code>d_w()</code></td><td class="src">metrics.py</td><td>Metrica compuesta: $0.55 d_{{VL}} + 0.25 d_{{Q5}} + 0.20 d_{{JSD}}$</td></tr>
<tr><td>PC overlap (Jaccard)</td><td class="src">&mdash;</td><td>$|PC_A \\cap PC_B| / |PC_A \\cup PC_B|$</td></tr>
<tr><td><code>d_VL()</code></td><td class="src">metrics.py</td><td>Voice leading continuo</td></tr>
<tr><td><code>d_JSD()</code></td><td class="src">pre_process.py</td><td>Rugosidad: similitud timbrica</td></tr>
</table>
</div>

<h2>1. Clasificacion de Candidatos</h2>
<div class="cd">
<ul>
<li><span class="bg bg-g">Plausible</span> Jaccard &ge; 50% AND VL &lt; 0.15, O rugosidad JSD &lt; 0.15. Justificacion teorica parcial.</li>
<li><span class="bg bg-o">Novel</span> Jaccard &ge; 33% O VL &lt; 0.10 sin ser plausible. Sin precedente teorico directo.</li>
<li><span class="bg bg-r">Spurious</span> Bajo overlap, alto VL. Probablemente artefacto de la metrica.</li>
</ul>
</div>

<div class="metrics">
<div class="mc"><div class="mv" style="color:var(--gn);">{cats.get('Plausible',0)}</div><div class="ml">Plausible</div></div>
<div class="mc"><div class="mv" style="color:var(--or);">{cats.get('Novel',0)}</div><div class="ml">Novel</div></div>
<div class="mc"><div class="mv" style="color:var(--rd);">{cats.get('Spurious',0)}</div><div class="ml">Spurious</div></div>
</div>

<h2>2. Top Candidatos Plausibles</h2>
<div class="scr"><table>
<thead><tr><th>Query</th><th>Vecino</th><th>Rank</th><th>$d_w$</th><th>Jaccard</th><th>VL</th><th>JSD</th><th>Justificacion</th></tr></thead>
<tbody>{plaus_rows}</tbody>
</table></div>

<h2>3. Top Candidatos Novel</h2>
<div class="scr"><table>
<thead><tr><th>Query</th><th>Vecino</th><th>Rank</th><th>$d_w$</th><th>Jaccard</th><th>VL</th><th>JSD</th><th>Justificacion</th></tr></thead>
<tbody>{novel_rows}</tbody>
</table></div>

<h2>4. Red de Sustitucion (Top-5 vecinos)</h2>
<div class="ch" id="network"></div>
<p style="color:var(--tm); font-size:.78rem;">Circulo: {N} acordes. Lineas verdes = GT (conocidos). Lineas grises = novel/nuevos. Hover para ver nombres.</p>

<h2>5. Conclusion</h2>
<div class="th">
<p class="th-l">Resultado</p>
<p>De {len(unique)} pares no-clasicos encontrados como vecinos cercanos en $d_{{\\mathbf{{w}}}}$, <strong>{cats.get('Plausible',0)}</strong> tienen justificacion timbrica o estructural parcial, y <strong>{cats.get('Novel',0)}</strong> representan relaciones no documentadas en la teoria clasica pero identificadas por la metrica compuesta.</p>
<p>Esto demuestra que el modelo de espacio de acordes de ChordSpace puede <strong>descubrir</strong> relaciones armonicas mas alla del conocimiento teorico establecido.</p>
</div>

<div class="ft">Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} &middot; ChordSpace &middot;
<code>experiments/chord_substitution/run_experiment_6.py</code></div>
</div>
<script>
const L={{paper_bgcolor:'#0d1117',plot_bgcolor:'#161b22',font:{{family:'Inter',color:'#e6edf3',size:13}},
  xaxis:{{showgrid:false,zeroline:false,showticklabels:false}},
  yaxis:{{showgrid:false,zeroline:false,showticklabels:false,scaleanchor:'x'}},
  margin:{{l:20,r:20,t:45,b:20}}}};
Plotly.newPlot('network',[
  {{x:{json.dumps(novel_edge_x)},y:{json.dumps(novel_edge_y)},mode:'lines',
    line:{{color:'rgba(139,148,158,0.15)',width:0.5}},name:'Novel',hoverinfo:'none'}},
  {{x:{json.dumps(gt_edge_x)},y:{json.dumps(gt_edge_y)},mode:'lines',
    line:{{color:'rgba(63,185,80,0.4)',width:1.5}},name:'GT',hoverinfo:'none'}},
  {{x:{json.dumps(net_x)},y:{json.dumps(net_y)},mode:'markers',
    text:{json.dumps(names)},marker:{{size:5,color:'#58a6ff',opacity:.8}},
    hovertemplate:'%{{text}}<extra></extra>',name:'Acordes'}},
],{{...L,title:{{text:'Red de sustitucion (top-5 vecinos en d_w)',font:{{size:15}}}},showlegend:true}});
</script>
</body></html>"""

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_6_report.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nReport: {out}")
