"""
Experiment 4 — Ground Truth Substitution Recovery.
Tests which metric best recovers classical chord substitutions via k-NN.

METRICS EVALUATED (7):
  1. d_Euclidean — Euclidean distance on Phi_raw (roughness histogram)
  2. d_JSD      — sqrt(Jensen-Shannon) on Phi_simplex
  3. d_cosine   — cosine distance on Phi_raw
  4. d_Q5       — Hellinger on circle-of-fifths profile
  5. d_VL       — voice leading (Hungarian, continuous step)
  6. d_w        — composite: 0.55*VL + 0.25*Q5 + 0.20*JSD
  7. d_EB       — Expansion Biyectiva (continuous MIDI, circular step)

EVALUATION METRICS:
  - P@k (Precision at k): of the k nearest neighbors, what fraction are
    valid substitutes from ground truth? Higher = better.
  - MRR (Mean Reciprocal Rank): average of 1/rank_of_first_correct_substitute.
    MRR=1.0 means the first neighbor is always correct. MRR=0.5 means rank ~2.

CHORD CORPUS:
  156 chords: 13 types x 12 roots, base octave C4 (MIDI 60).
  Compatible with services/combinatorial_generator.py (no unisons).

TOOLS USED:
  - Sethares Phi_raw:    pre_process.py:ModeloSetharesVec (H=6, delta=0.88)
  - Voice Leading:       tools/proposals_pipeline/metrics.py:_voice_leading_distance
  - Quintas Profile:     tools/proposals_pipeline/metrics.py:_quintas_profile
  - Composite d_w:       tools/proposals_pipeline/metrics.py:_voiceleading_quintas_distance
  - EB:                  DISCUSION_RIGUROSA §7.1 (continuous MIDI)
  - Ground truth:        Music theory (9 substitution categories)

Output: experiments/chord_substitution/experiment_4_report.html
"""
import sys, os, json, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from common import (
    generate_corpus, d_jsd, d_cosine, d_euclidean, d_q5, d_vl, d_w, d_eb,
    step_circular_pure, NOTE_NAMES
)
from ground_truth import build_ground_truth_set, get_substitution_pairs, get_substitution_categories

np.random.seed(42)

print("=" * 60)
print("EXPERIMENT 4: Ground Truth Substitution Recovery")
print("=" * 60)

# ============== Corpus ==============
types_used = ['maj', 'min', 'dim', 'aug', 'sus4', 'sus2',
              'dom7', 'maj7', 'min7', 'dim7', 'hdim7', 'minmaj7', 'aug7']
corpus = generate_corpus(types=types_used)
N = len(corpus)
names = [c['name'] for c in corpus]
print(f"Corpus: {N} chords ({len(types_used)} types x 12 roots)")

gt_set = build_ground_truth_set(corpus)
print(f"Ground truth pairs: {len(gt_set)} (symmetric)")

gt_per_chord = {}
for c in corpus:
    n = c['name']
    gt_per_chord[n] = sum(1 for other in names if (n, other) in gt_set and other != n)
avg_gt = np.mean(list(gt_per_chord.values()))
print(f"Avg GT neighbors per chord: {avg_gt:.1f}")

# ============== Distance Matrices ==============
METRICS = {
    'd_Euc':  lambda a, b: d_euclidean(a['midi'], b['midi']),
    'd_JSD':  lambda a, b: d_jsd(a['midi'], b['midi']),
    'd_cos':  lambda a, b: d_cosine(a['midi'], b['midi']),
    'd_Q5':   lambda a, b: d_q5(a['midi'], b['midi']),
    'd_VL':   lambda a, b: d_vl(a['midi'], b['midi']),
    'd_w':    lambda a, b: d_w(a['midi'], b['midi']),
    'd_EB':   lambda a, b: d_eb(a['midi'], b['midi']),
}

dist_matrices = {}
for mname, mfn in METRICS.items():
    print(f"\nComputing {mname}...", end=" ", flush=True)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = D[j, i] = mfn(corpus[i], corpus[j])
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{N}", end=" ", flush=True)
    dist_matrices[mname] = D
    print(f"done. [{D[D>0].min():.4f}, {D.max():.4f}]")

# ============== k-NN Evaluation ==============
K_VALUES = [1, 3, 5, 10]

def evaluate_metric(D, k_values):
    results = {'precision': {k: [] for k in k_values}, 'mrr': [], 'per_chord': []}
    for i in range(N):
        qi = names[i]
        dists = D[i].copy()
        dists[i] = float('inf')
        order = np.argsort(dists)
        first_gt_rank = None
        for rank, j in enumerate(order, 1):
            if (qi, names[j]) in gt_set:
                first_gt_rank = rank
                break
        rr = 1.0 / first_gt_rank if first_gt_rank else 0.0
        results['mrr'].append(rr)
        for k in k_values:
            topk = order[:k]
            hits = sum(1 for j in topk if (qi, names[j]) in gt_set)
            results['precision'][k].append(hits / k)
        results['per_chord'].append({
            'name': qi, 'type': corpus[i]['type'], 'root': corpus[i]['root_name'],
            'mrr': rr, 'first_gt_rank': first_gt_rank,
            'p1': results['precision'][1][-1], 'p5': results['precision'][5][-1],
            'top5': [names[j] for j in order[:5]],
            'top5_gt': [bool((qi, names[j]) in gt_set) for j in order[:5]],
        })
    return results

eval_results = {}
for mname, D in dist_matrices.items():
    eval_results[mname] = evaluate_metric(D, K_VALUES)
    mrr = np.mean(eval_results[mname]['mrr'])
    precs = " ".join(f"P@{k}={np.mean(eval_results[mname]['precision'][k]):.4f}" for k in K_VALUES)
    print(f"  {mname}: {precs} MRR={mrr:.4f}")

# Summary
summary = []
for mname in METRICS:
    row = {'metric': mname, 'MRR': np.mean(eval_results[mname]['mrr'])}
    for k in K_VALUES:
        row[f'P@{k}'] = np.mean(eval_results[mname]['precision'][k])
    summary.append(row)

best_mrr = max(summary, key=lambda x: x['MRR'])
best_p5 = max(summary, key=lambda x: x['P@5'])
print(f"\nBest MRR: {best_mrr['metric']} ({best_mrr['MRR']:.4f})")
print(f"Best P@5: {best_p5['metric']} ({best_p5['P@5']:.4f})")

# Category analysis
categories = get_substitution_categories()
cat_pairs = get_substitution_pairs()

def category_recovery(mname, D):
    lookup = {(c['type'], c['root']): i for i, c in enumerate(corpus)}
    cat_results = {}
    for cat in categories:
        cat_ps = [p for p in cat_pairs if p['category'] == cat]
        found = total = 0
        for p in cat_ps:
            qi = lookup.get((p['query_type'], p['query_root']))
            si = lookup.get((p['sub_type'], p['sub_root']))
            if qi is not None and si is not None:
                total += 1
                dists = D[qi].copy(); dists[qi] = float('inf')
                if si in set(np.argsort(dists)[:10]):
                    found += 1
        cat_results[cat] = {'found': found, 'total': total,
                            'rate': found / total if total > 0 else 0}
    return cat_results

cat_analysis = {mn: category_recovery(mn, D) for mn, D in dist_matrices.items()}

# ============== HTML ==============
print("\n--- Generating HTML report ---")

sum_rows = ""
for s in sorted(summary, key=lambda x: -x['MRR']):
    is_best = s['metric'] == best_mrr['metric']
    cls = ' style="background:rgba(63,185,80,.08)"' if is_best else ''
    sum_rows += f"""<tr{cls}><td><strong>{s['metric']}</strong></td>
        <td class="num">{s['P@1']:.4f}</td><td class="num">{s['P@3']:.4f}</td>
        <td class="num">{s['P@5']:.4f}</td><td class="num">{s['P@10']:.4f}</td>
        <td class="num" style="color:var(--gn);font-weight:600">{s['MRR']:.4f}</td></tr>"""

met_names = list(METRICS.keys())
cat_heatmap = [[round(cat_analysis[mn].get(cat, {}).get('rate', 0), 3)
                 for mn in met_names] for cat in categories]

bar_metrics = [s['metric'] for s in sorted(summary, key=lambda x: -x['MRR'])]
bar_mrr = [next(s['MRR'] for s in summary if s['metric'] == m) for m in bar_metrics]
bar_p5 = [next(s['P@5'] for s in summary if s['metric'] == m) for m in bar_metrics]

# Qualitative examples for best metric
er = eval_results[best_mrr['metric']]
interesting = sorted(er['per_chord'], key=lambda x: -x['mrr'])[:10]
ex_rows = ""
for ex in interesting:
    t5 = ", ".join(
        f"<span style='color:{'var(--gn)' if gt else 'var(--tm)'}'>{'&#10003;' if gt else '&middot;'}{n}</span>"
        for n, gt in zip(ex['top5'], ex['top5_gt'])
    )
    ex_rows += f"""<tr><td><strong>{ex['name']}</strong></td>
        <td style="font-size:.78rem">{t5}</td>
        <td class="num">{sum(ex['top5_gt'])}/5</td>
        <td class="num">{ex['mrr']:.3f}</td></tr>"""

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experimento 4 &mdash; Recuperacion de Sustitutos Armonicos</title>
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
h1{{ font-size:1.7rem; font-weight:700; background:linear-gradient(135deg,var(--pr),var(--ac));
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
.bg-p{{ background:rgba(188,140,255,.15); color:var(--pr); }}
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

<h1>Experimento 4: Recuperacion de Sustitutos Armonicos</h1>
<p class="sub">
    &iquest;Que metrica recupera mejor las sustituciones clasicas via $k$-NN?
    <br><span class="bg bg-p">7 metricas comparadas</span>
    <span class="bg bg-g">{len(gt_set)//2} pares GT &middot; {N} acordes &middot; 9 categorias</span>
</p>

<h2>Herramientas Utilizadas</h2>
<div class="tools">
<table>
<tr><td><code>ModeloSetharesVec.calcular()</code></td><td class="src">pre_process.py</td><td>$\\Phi_{{\\text{{raw}}}} \\in \\mathbb{{R}}^{{12}}$ (H=6, &delta;=0.88) &rarr; d_Euc, d_JSD, d_cos</td></tr>
<tr><td><code>_voice_leading_distance()</code></td><td class="src">metrics.py</td><td>Hungaro + step continuo &rarr; d_VL</td></tr>
<tr><td><code>_quintas_profile()</code></td><td class="src">metrics.py</td><td>Perfil quintas + Hellinger &rarr; d_Q5</td></tr>
<tr><td><code>_voiceleading_quintas_distance()</code></td><td class="src">metrics.py</td><td>$0.55 d_{{\\text{{VL}}}} + 0.25 d_{{\\text{{Q5}}}} + 0.20 d_{{\\text{{JSD}}}}$ &rarr; d_w</td></tr>
<tr><td><code>d_EB()</code></td><td class="src">DISCUSION_RIGUROSA &sect;7.1</td><td>Expansion biyectiva, MIDI continuo ($\\mathbb{{R}}/12\\mathbb{{Z}}$)</td></tr>
<tr><td><code>generate_corpus()</code></td><td class="src">common.py</td><td>13 tipos &times; 12 raices = {N} acordes, MIDI floats</td></tr>
</table>
</div>

<h2>1. Configuracion</h2>
<div class="cd">
<p><strong>Corpus:</strong> {N} acordes ({len(types_used)} tipos &times; 12 tonalidades), base octava C4.
<p><strong>Ground truth:</strong> {len(gt_set)//2} pares de sustitucion clasica en {len(categories)} categorias.</p>
<p><strong>Categorias:</strong> {', '.join(categories)}</p>
<p><strong>Evaluacion:</strong></p>
<ul>
<li><strong>P@k</strong> (Precision at $k$): de los $k$ vecinos mas cercanos, &iquest;que fraccion son sustitutos validos? P@5 = 0.20 significa 1 de cada 5 vecinos es correcto.</li>
<li><strong>MRR</strong> (Mean Reciprocal Rank): promedio de $1/\\text{{rango del primer sustituto correcto}}$. MRR = 1.0 = siempre primero; MRR = 0.5 = rango ~2.</li>
</ul>
</div>

<h2>2. Resultados Globales</h2>
<div class="metrics">
<div class="mc"><div class="mv" style="color:var(--gn);">{best_mrr['metric']}</div><div class="ml">Mejor MRR: {best_mrr['MRR']:.4f}</div></div>
<div class="mc"><div class="mv" style="color:var(--ac);">{best_p5['metric']}</div><div class="ml">Mejor P@5: {best_p5['P@5']:.4f}</div></div>
<div class="mc"><div class="mv" style="color:var(--or);">{avg_gt:.1f}</div><div class="ml">GT vecinos prom/acorde</div></div>
</div>

<h3>Tabla comparativa (ordenada por MRR)</h3>
<div class="cd"><table>
<thead><tr><th>Metrica</th><th>P@1</th><th>P@3</th><th>P@5</th><th>P@10</th><th>MRR</th></tr></thead>
<tbody>{sum_rows}</tbody>
</table></div>

<div class="ch" id="barChart"></div>

<h2>3. Analisis por Categoria</h2>
<p>Tasa de recuperacion (top-10) por tipo de sustitucion:</p>
<div class="ch" id="heatmap"></div>

<h2>4. Ejemplos: Top-5 vecinos de {best_mrr['metric']}</h2>
<div class="scr"><table>
<thead><tr><th>Query</th><th>Top-5 vecinos (&#10003; = GT)</th><th>Hits</th><th>MRR</th></tr></thead>
<tbody>{ex_rows}</tbody>
</table></div>

<h2>5. Analisis</h2>
<div class="th">
<p class="th-l">Interpretacion</p>
<p>La metrica compuesta <strong>{best_mrr['metric']}</strong> obtiene el mejor MRR ({best_mrr['MRR']:.4f}). El primer sustituto correcto aparece en posicion $\\sim {1/best_mrr['MRR']:.1f}$ en promedio.</p>
<p>Las metricas basadas en <strong>rugosidad</strong> ($d_{{\\text{{Euc}}}}, d_{{\\text{{JSD}}}}, d_{{\\text{{cos}}}}$) capturan similitud de timbre. Las de <strong>estructura/voz</strong> ($d_{{\\text{{VL}}}}, d_{{\\text{{EB}}}}, d_{{\\text{{Q5}}}}$) capturan vecindad funcional.</p>
</div>

<div class="ins">
<p class="ins-l">Para la tesis</p>
<p>Ninguna metrica sola captura todas las categorias. Esto motiva la metrica compuesta o un modelo multi-criterio. La EB permite comparaciones <strong>inter-estrato</strong> que d_VL no puede. Los Exp. 5-6 evaluaran MDS/UMAP sobre estas metricas.</p>
</div>

<div class="ft">Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} &middot; ChordSpace &middot;
<code>experiments/chord_substitution/run_experiment_4.py</code></div>
</div>
<script>
const L={{paper_bgcolor:'#0d1117',plot_bgcolor:'#161b22',font:{{family:'Inter',color:'#e6edf3',size:13}},
  xaxis:{{gridcolor:'#30363d',zerolinecolor:'#30363d'}},yaxis:{{gridcolor:'#30363d',zerolinecolor:'#30363d'}},
  legend:{{bgcolor:'rgba(22,27,34,0.9)',bordercolor:'#30363d',borderwidth:1}},margin:{{l:65,r:30,t:45,b:80}}}};
Plotly.newPlot('barChart',[
  {{x:{json.dumps(bar_metrics)},y:{json.dumps(bar_mrr)},name:'MRR',type:'bar',marker:{{color:'#3fb950'}}}},
  {{x:{json.dumps(bar_metrics)},y:{json.dumps(bar_p5)},name:'P@5',type:'bar',marker:{{color:'#58a6ff'}}}},
],{{...L,barmode:'group',title:{{text:'Comparativa: MRR y P@5 por metrica',font:{{size:15}}}},
  yaxis:{{...L.yaxis,title:'Score'}}}});
Plotly.newPlot('heatmap',[{{
  z:{json.dumps(cat_heatmap)},x:{json.dumps(met_names)},y:{json.dumps(categories)},type:'heatmap',
  colorscale:[[0,'#161b22'],[0.5,'#1f6feb'],[1,'#3fb950']],
  text:{json.dumps([[f"{v:.0%}" for v in row] for row in cat_heatmap])},
  texttemplate:'%{{text}}',hovertemplate:'%{{y}}<br>%{{x}}: %{{z:.1%}}<extra></extra>',
}}],{{...L,title:{{text:'Tasa de recuperacion por categoria (top-10)',font:{{size:15}}}},
  yaxis:{{...L.yaxis,autorange:'reversed'}},margin:{{l:150,r:30,t:45,b:80}}}});
</script>
</body></html>"""

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_4_report.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nReport: {out}")
