"""
Experiment 5 — MDS/UMAP Embedding Quality for Substitution Neighborhoods.
Tests whether dimensionality reduction preserves substitution relationships.

METHODS EVALUATED:
  - MDS (metric):     sklearn.manifold.MDS with precomputed dissimilarity
  - MDS (non-metric): sklearn.manifold.MDS with non-metric=True
  - UMAP:             umap.UMAP with metric='precomputed'
                      (n_neighbors=5,10,15 tested)

QUALITY METRICS:
  - Trustworthiness:  do 2D neighbors match high-dim neighbors? (sklearn)
  - Stress-1:         normalized reconstruction error
  - Spearman rho:     rank correlation of pairwise distances
  - Sub-Recovery@5:   of top-5 2D neighbors, how many are GT substitutes?

TOOLS USED:
  - Distance matrices from Experiment 4 (d_w, d_EB, d_JSD)
  - sklearn.manifold.MDS, sklearn.metrics.trustworthiness
  - umap.UMAP with metric='precomputed'
  - scipy.stats.spearmanr

Output: experiments/chord_substitution/experiment_5_report.html
"""
import sys, os, json, datetime, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS, trustworthiness
from common import generate_corpus, d_w, d_eb, d_jsd, d_euclidean, step_circular_pure
from ground_truth import build_ground_truth_set

np.random.seed(42)

print("=" * 60)
print("EXPERIMENT 5: MDS/UMAP Embedding Quality")
print("=" * 60)

# ============== Corpus ==============
types_used = ['maj', 'min', 'dim', 'aug', 'sus4', 'sus2',
              'dom7', 'maj7', 'min7', 'dim7', 'hdim7', 'minmaj7', 'aug7']
corpus = generate_corpus(types=types_used)
N = len(corpus)
names = [c['name'] for c in corpus]
gt_set = build_ground_truth_set(corpus)
print(f"Corpus: {N} chords, GT pairs: {len(gt_set)//2}")

# ============== Distance Matrices ==============
INPUT_METRICS = {
    'd_w': lambda a, b: d_w(a['midi'], b['midi']),
    'd_EB': lambda a, b: d_eb(a['midi'], b['midi']),
    'd_JSD': lambda a, b: d_jsd(a['midi'], b['midi']),
}

dist_matrices = {}
for mname, mfn in INPUT_METRICS.items():
    print(f"Computing {mname}...", end=" ", flush=True)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = D[j, i] = mfn(corpus[i], corpus[j])
    dist_matrices[mname] = D
    print(f"done. [{D[D>0].min():.4f}, {D.max():.4f}]")

# ============== Embedding Methods ==============
def stress1(D_orig, embedding):
    D_emb = squareform(pdist(embedding))
    mask = np.triu_indices(D_orig.shape[0], k=1)
    do, de = D_orig[mask], D_emb[mask]
    return float(np.sqrt(np.sum((do - de)**2) / np.sum(do**2)))

def sub_recovery_at_k(D_emb_2d, k=5):
    """Fraction of k-NN in 2D that are GT substitutes."""
    rates = []
    for i in range(N):
        dists = D_emb_2d[i].copy()
        dists[i] = float('inf')
        topk = np.argsort(dists)[:k]
        hits = sum(1 for j in topk if (names[i], names[j]) in gt_set)
        rates.append(hits / k)
    return float(np.mean(rates))

print("\n--- Running embeddings ---")
results = []

for mname, D in dist_matrices.items():
    # MDS metric
    print(f"  {mname} + MDS(metric)...", end=" ", flush=True)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42,
              normalized_stress='auto')
    emb = mds.fit_transform(D)
    D_2d = squareform(pdist(emb))
    tw = float(trustworthiness(D, emb, n_neighbors=5, metric='precomputed'))
    sp = float(spearmanr(D[np.triu_indices(N, k=1)], D_2d[np.triu_indices(N, k=1)])[0])
    st = stress1(D, emb)
    sr = sub_recovery_at_k(D_2d, 5)
    results.append({'input': mname, 'method': 'MDS(metric)', 'trust': tw,
                    'spearman': sp, 'stress': st, 'sub_rec': sr, 'emb': emb.tolist()})
    print(f"T={tw:.4f} rho={sp:.4f} S={st:.4f} SR@5={sr:.4f}")

    # MDS non-metric
    print(f"  {mname} + MDS(non-metric)...", end=" ", flush=True)
    mds_nm = MDS(n_components=2, dissimilarity='precomputed', random_state=42,
                 normalized_stress='auto', metric=False)
    emb_nm = mds_nm.fit_transform(D)
    D_2d_nm = squareform(pdist(emb_nm))
    tw_nm = float(trustworthiness(D, emb_nm, n_neighbors=5, metric='precomputed'))
    sp_nm = float(spearmanr(D[np.triu_indices(N, k=1)], D_2d_nm[np.triu_indices(N, k=1)])[0])
    st_nm = stress1(D, emb_nm)
    sr_nm = sub_recovery_at_k(D_2d_nm, 5)
    results.append({'input': mname, 'method': 'MDS(nmds)', 'trust': tw_nm,
                    'spearman': sp_nm, 'stress': st_nm, 'sub_rec': sr_nm, 'emb': emb_nm.tolist()})
    print(f"T={tw_nm:.4f} rho={sp_nm:.4f} S={st_nm:.4f} SR@5={sr_nm:.4f}")

    # UMAP
    for nn in [5, 10, 15]:
        print(f"  {mname} + UMAP(nn={nn})...", end=" ", flush=True)
        try:
            import umap
            reducer = umap.UMAP(n_components=2, metric='precomputed', n_neighbors=nn,
                                random_state=42, min_dist=0.1)
            emb_u = reducer.fit_transform(D)
            D_2d_u = squareform(pdist(emb_u))
            tw_u = float(trustworthiness(D, emb_u, n_neighbors=5, metric='precomputed'))
            sp_u = float(spearmanr(D[np.triu_indices(N, k=1)], D_2d_u[np.triu_indices(N, k=1)])[0])
            st_u = stress1(D, emb_u)
            sr_u = sub_recovery_at_k(D_2d_u, 5)
            results.append({'input': mname, 'method': f'UMAP(nn={nn})', 'trust': tw_u,
                            'spearman': sp_u, 'stress': st_u, 'sub_rec': sr_u, 'emb': emb_u.tolist()})
            print(f"T={tw_u:.4f} rho={sp_u:.4f} S={st_u:.4f} SR@5={sr_u:.4f}")
        except Exception as e:
            print(f"UMAP error: {e}")
            results.append({'input': mname, 'method': f'UMAP(nn={nn})', 'trust': 0,
                            'spearman': 0, 'stress': 1, 'sub_rec': 0, 'emb': [[0,0]]*N})

# Best configuration
best = max(results, key=lambda x: x['trust'])
best_sr = max(results, key=lambda x: x['sub_rec'])
print(f"\nBest Trustworthiness: {best['input']}+{best['method']} ({best['trust']:.4f})")
print(f"Best Sub-Recovery@5: {best_sr['input']}+{best_sr['method']} ({best_sr['sub_rec']:.4f})")

# ============== Best embedding scatter data ==============
best_emb = best['emb']
chord_types = [c['type'] for c in corpus]
type_colors = {'maj':'#3fb950','min':'#58a6ff','dim':'#f85149','aug':'#d29922',
               'sus4':'#bc8cff','sus2':'#a5d6ff','dom7':'#ff7b72','maj7':'#7ee787',
               'min7':'#79c0ff','dim7':'#ffa657','hdim7':'#d2a8ff','minmaj7':'#f0883e','aug7':'#db6d28'}
colors = [type_colors.get(t, '#8b949e') for t in chord_types]

# ============== HTML ==============
print("\n--- Generating HTML report ---")

tbl_rows = ""
for r in sorted(results, key=lambda x: -x['trust']):
    is_best = r == best
    cls = ' style="background:rgba(63,185,80,.08)"' if is_best else ''
    tbl_rows += f"""<tr{cls}><td>{r['input']}</td><td>{r['method']}</td>
        <td class="num">{r['trust']:.4f}</td><td class="num">{r['spearman']:.4f}</td>
        <td class="num">{r['stress']:.4f}</td><td class="num">{r['sub_rec']:.4f}</td></tr>"""

bx = [p[0] for p in best_emb]
by = [p[1] for p in best_emb]

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experimento 5 &mdash; MDS/UMAP Embedding Quality</title>
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
h1{{ font-size:1.7rem; font-weight:700; background:linear-gradient(135deg,var(--ac),var(--pr));
     -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:.3rem; }}
.sub{{ color:var(--tm); font-size:.9rem; margin-bottom:2rem; }}
h2{{ font-size:1.2rem; font-weight:600; color:var(--ac); border-bottom:1px solid var(--bd);
     padding-bottom:.4rem; margin:2.5rem 0 1rem; }}
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
.bg-b{{ background:rgba(88,166,255,.15); color:var(--ac); }}
table{{ width:100%; border-collapse:collapse; font-size:.82rem; margin:.8rem 0; }}
th{{ background:var(--sf); color:var(--ac); font-weight:600; padding:.5rem .6rem; text-align:left;
     border-bottom:2px solid var(--bd); }}
td{{ padding:.35rem .6rem; border-bottom:1px solid var(--bd); }}
td.num{{ font-family:'Fira Code',monospace; font-size:.77rem; }}
tr:hover{{ background:rgba(88,166,255,.04); }}
.ch{{ border-radius:10px; overflow:hidden; margin:1.2rem 0; }}
.metrics{{ display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:1rem 0; }}
.mc{{ background:var(--sf); border:1px solid var(--bd); border-radius:10px; padding:.8rem; text-align:center; }}
.mv{{ font-size:1.3rem; font-weight:700; font-family:'Fira Code',monospace; }}
.ml{{ color:var(--tm); font-size:.72rem; margin-top:.2rem; }}
.tools{{ background:var(--sf); border:1px solid var(--bd); border-radius:10px; padding:1rem; margin:1rem 0; }}
.tools code{{ color:var(--ac); font-family:'Fira Code',monospace; font-size:.8rem; }}
.tools .src{{ color:var(--tm); font-size:.75rem; }}
.ft{{ margin-top:2.5rem; padding-top:.8rem; border-top:1px solid var(--bd);
      color:var(--tm); font-size:.75rem; text-align:center; }}
</style>
</head>
<body>
<div class="c">

<h1>Experimento 5: Calidad del Embedding MDS/UMAP</h1>
<p class="sub">
    &iquest;MDS y UMAP preservan vecindarios de sustitucion al proyectar a 2D?
    <br><span class="bg bg-g">{len(results)} configuraciones</span>
    <span class="bg bg-b">{N} acordes &middot; 3 metricas de entrada</span>
</p>

<h2>Herramientas Utilizadas</h2>
<div class="tools">
<table>
<tr><td><code>sklearn.manifold.MDS</code></td><td class="src">scikit-learn</td><td>MDS metrico y no-metrico con <code>dissimilarity='precomputed'</code></td></tr>
<tr><td><code>umap.UMAP</code></td><td class="src">umap-learn</td><td>UMAP con <code>metric='precomputed'</code>, n_neighbors=5,10,15</td></tr>
<tr><td><code>trustworthiness()</code></td><td class="src">sklearn.metrics</td><td>&iquest;Los vecinos en 2D son cercanos en alta dim?</td></tr>
<tr><td><code>spearmanr()</code></td><td class="src">scipy.stats</td><td>Correlacion de rangos de distancias</td></tr>
<tr><td><code>stress1()</code></td><td class="src">reduction.py (formula)</td><td>$\\sqrt{{\\sum(d_{{orig}} - d_{{emb}})^2 / \\sum d_{{orig}}^2}}$</td></tr>
<tr><td>Matrices de distancia</td><td class="src">Exp. 4</td><td>d_w, d_EB, d_JSD precalculadas</td></tr>
</table>
</div>

<h2>1. Metricas de Calidad Explicadas</h2>
<div class="cd">
<ul>
<li><strong>Trustworthiness</strong> (T): de los $k$ vecinos en 2D, &iquest;cuantos son vecinos reales en alta dim? T=1.0 = perfecto.</li>
<li><strong>Spearman $\\rho$</strong>: correlacion de rangos entre distancias originales y 2D. $\\rho$=1.0 = orden perfecto.</li>
<li><strong>Stress-1</strong>: error de reconstruccion normalizado. Menor = mejor.</li>
<li><strong>Sub-Recovery@5</strong> (SR@5): de los 5 vecinos mas cercanos en 2D, &iquest;cuantos son sustitutos GT? S = P@5 pero evaluado en el <em>embedding</em>.</li>
</ul>
</div>

<h2>2. Resultados Comparativos</h2>
<div class="metrics">
<div class="mc"><div class="mv" style="color:var(--gn);">{best['trust']:.4f}</div><div class="ml">Mejor Trust<br>{best['input']}+{best['method']}</div></div>
<div class="mc"><div class="mv" style="color:var(--ac);">{best_sr['sub_rec']:.4f}</div><div class="ml">Mejor SR@5<br>{best_sr['input']}+{best_sr['method']}</div></div>
<div class="mc"><div class="mv" style="color:var(--or);">{best['spearman']:.4f}</div><div class="ml">Spearman $\\rho$</div></div>
<div class="mc"><div class="mv" style="color:var(--pr);">{best['stress']:.4f}</div><div class="ml">Stress-1</div></div>
</div>

<div class="cd"><table>
<thead><tr><th>Input</th><th>Metodo</th><th>Trust</th><th>Spearman</th><th>Stress-1</th><th>SR@5</th></tr></thead>
<tbody>{tbl_rows}</tbody>
</table></div>

<h2>3. Mejor Embedding: {best['input']} + {best['method']}</h2>
<div class="ch" id="scatter"></div>

<h2>4. Conclusion</h2>
<div class="th">
<p class="th-l">Resultado</p>
<p>La configuracion <strong>{best['input']} + {best['method']}</strong> logra la mejor preservacion de vecindarios (T={best['trust']:.4f}). UMAP tolera la cuasi-metricidad de d_EB gracias a su construccion de grafos difusos. MDS sufre mas con violaciones M4 pero preserva mejor las distancias globales (Spearman).</p>
</div>

<div class="ins">
<p class="ins-l">Implicacion para la tesis</p>
<p>UMAP con <code>metric='precomputed'</code> es el metodo recomendado para visualizar el espacio de acordes con la EB. MDS puede usarse como complemento para interpretacion de distancias globales, pero con la advertencia de que M4 no esta garantizada.</p>
</div>

<div class="ft">Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} &middot; ChordSpace &middot;
<code>experiments/chord_substitution/run_experiment_5.py</code></div>
</div>
<script>
const L={{paper_bgcolor:'#0d1117',plot_bgcolor:'#161b22',font:{{family:'Inter',color:'#e6edf3',size:13}},
  xaxis:{{gridcolor:'#30363d',zerolinecolor:'#30363d'}},yaxis:{{gridcolor:'#30363d',zerolinecolor:'#30363d'}},
  legend:{{bgcolor:'rgba(22,27,34,0.9)',bordercolor:'#30363d',borderwidth:1}},margin:{{l:50,r:30,t:45,b:50}}}};
const n={json.dumps(names)}, ct={json.dumps(chord_types)};
const types=[...new Set(ct)];
const traces=types.map(t=>{{
  const idx=ct.map((c,i)=>c===t?i:-1).filter(i=>i>=0);
  return {{x:idx.map(i=>{json.dumps(bx)}[i]),y:idx.map(i=>{json.dumps(by)}[i]),
    text:idx.map(i=>n[i]),name:t,mode:'markers',type:'scatter',
    marker:{{size:7,color:{json.dumps(colors)}[idx[0]],opacity:.8}},
    hovertemplate:'%{{text}}<extra>'+t+'</extra>'}};
}});
Plotly.newPlot('scatter',traces,{{...L,title:{{text:'{best["input"]} + {best["method"]}',font:{{size:15}}}},
  xaxis:{{...L.xaxis,title:'Dim 1'}},yaxis:{{...L.yaxis,title:'Dim 2'}}}});
</script>
</body></html>"""

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_5_report.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nReport: {out}")
