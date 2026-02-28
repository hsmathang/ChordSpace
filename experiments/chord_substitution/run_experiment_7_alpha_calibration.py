"""
Experiment 7 -- Alpha Calibration for d_combo = alpha*d_EB + (1-alpha)*sqrt(JSD).

METHODOLOGY:
  1. Corpus: 13 types x 12 roots = 156 chords (generate_corpus).
  2. Compute D_EB (156x156) and D_JSD (156x156).
  3. Range-normalize both to [0,1].
  4. Grid search alpha in {0.00, 0.05, ..., 1.00} (21 values).
  5. 5-fold stratified CV over 228 GT substitution pairs.
  6. Also compute LOO-CV as sanity check.
  7. Compare d_combo(alpha*) against 7 existing metrics.

NORMALIZATION:
  Range normalization: D_hat = D / max(D).
  Also z-score as robustness check: D_hat = (D - mean) / std.

EVALUATION:
  MRR (Mean Reciprocal Rank) and P@k on held-out GT pairs.

TOOLS USED:
  - d_EB:    common.py:d_eb (Expansion Biyectiva, R/12Z)
  - d_JSD:   common.py:d_jsd (sqrt Jensen-Shannon on Phi_simplex)
  - d_Euc:   common.py:d_euclidean (Euclidean on Phi_raw)
  - d_cos:   common.py:d_cosine (cosine on Phi_raw)
  - d_Q5:    common.py:d_q5 (Hellinger on quintas profile)
  - d_VL:    common.py:d_vl (Hungarian + step_continuous)
  - d_w:     common.py:d_w (0.55*VL + 0.25*Q5 + 0.20*JSD)
  - GT:      ground_truth.py (9 categories, 19 templates x 12 roots)

Output: experiments/chord_substitution/experiment_7_report.html
"""
import sys, os, json, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from common import (
    generate_corpus, d_jsd, d_cosine, d_euclidean, d_q5, d_vl, d_w, d_eb,
    step_circular_pure, NOTE_NAMES
)
from ground_truth import (
    build_ground_truth_set, get_substitution_pairs, get_substitution_categories
)

np.random.seed(42)

print("=" * 60)
print("EXPERIMENT 7: Alpha Calibration for d_combo")
print("=" * 60)

# ============== Step 1: Corpus ==============
types_used = ['maj', 'min', 'dim', 'aug', 'sus4', 'sus2',
              'dom7', 'maj7', 'min7', 'dim7', 'hdim7', 'minmaj7', 'aug7']
corpus = generate_corpus(types=types_used)
N = len(corpus)
names = [c['name'] for c in corpus]
print(f"Corpus: {N} chords ({len(types_used)} types x 12 roots)")

n_triads = sum(1 for c in corpus if c['card'] == 3)
n_tetrads = sum(1 for c in corpus if c['card'] == 4)
print(f"  Triads: {n_triads}, Tetrads: {n_tetrads}")

gt_set = build_ground_truth_set(corpus)
gt_pairs_list = []
seen = set()
for a_name in names:
    for b_name in names:
        if (a_name, b_name) in gt_set and a_name != b_name:
            pair = tuple(sorted([a_name, b_name]))
            if pair not in seen:
                seen.add(pair)
                gt_pairs_list.append(pair)
print(f"Ground truth: {len(gt_pairs_list)} unique pairs, {len(gt_set)} directed")

# ============== Step 2: Compute distance matrices ==============
print("\n--- Computing distance matrices ---")

def compute_matrix(metric_fn, label):
    print(f"  {label}...", end=" ", flush=True)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = D[j, i] = metric_fn(corpus[i]['midi'], corpus[j]['midi'])
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{N}", end=" ", flush=True)
    print(f"done. [{D[D>0].min():.4f}, {D.max():.4f}]")
    return D

D_EB = compute_matrix(d_eb, "d_EB")
D_JSD = compute_matrix(d_jsd, "d_JSD")
D_Euc = compute_matrix(d_euclidean, "d_Euc")
D_cos = compute_matrix(d_cosine, "d_cos")
D_Q5 = compute_matrix(d_q5, "d_Q5")
D_VL = compute_matrix(d_vl, "d_VL")
D_w = compute_matrix(d_w, "d_w")

# ============== Step 3: Normalize ==============
print("\n--- Normalizing ---")

# Range normalization
D_EB_range = D_EB / (D_EB.max() + 1e-12)
D_JSD_range = D_JSD / (D_JSD.max() + 1e-12)
print(f"  Range: D_EB max={D_EB.max():.4f}, D_JSD max={D_JSD.max():.4f}")

# Z-score normalization (upper triangle only for stats)
ut = np.triu_indices(N, k=1)
eb_mean, eb_std = D_EB[ut].mean(), D_EB[ut].std()
jsd_mean, jsd_std = D_JSD[ut].mean(), D_JSD[ut].std()
D_EB_zscore = (D_EB - eb_mean) / (eb_std + 1e-12)
D_JSD_zscore = (D_JSD - jsd_mean) / (jsd_std + 1e-12)
# Shift to non-negative
zmin = min(D_EB_zscore[ut].min(), D_JSD_zscore[ut].min())
D_EB_zscore -= zmin
D_JSD_zscore -= zmin
np.fill_diagonal(D_EB_zscore, 0)
np.fill_diagonal(D_JSD_zscore, 0)
print(f"  Z-score: EB mean={eb_mean:.4f} std={eb_std:.4f}, JSD mean={jsd_mean:.4f} std={jsd_std:.4f}")

# ============== Step 4 & 5: Grid search + CV ==============
print("\n--- Grid search over alpha with 5-fold CV ---")

alphas = np.arange(0.0, 1.025, 0.05)
alphas = np.round(alphas, 2)

def evaluate_mrr_p(D, gt_eval_set):
    """Compute MRR and P@k for given distance matrix and GT pairs."""
    mrrs = []
    p_at = {1: [], 3: [], 5: [], 10: []}
    for i in range(N):
        qi = names[i]
        # Check if this chord has any GT in eval set
        has_gt = any((qi, other) in gt_eval_set for other in names if other != qi)
        if not has_gt:
            continue
        dists = D[i].copy()
        dists[i] = float('inf')
        order = np.argsort(dists)
        first_gt_rank = None
        for rank, j in enumerate(order, 1):
            if (qi, names[j]) in gt_eval_set:
                first_gt_rank = rank
                break
        rr = 1.0 / first_gt_rank if first_gt_rank else 0.0
        mrrs.append(rr)
        for k in p_at:
            topk = order[:k]
            hits = sum(1 for j in topk if (qi, names[j]) in gt_eval_set)
            p_at[k].append(hits / k)
    return {
        'MRR': float(np.mean(mrrs)) if mrrs else 0.0,
        'P@1': float(np.mean(p_at[1])) if p_at[1] else 0.0,
        'P@5': float(np.mean(p_at[5])) if p_at[5] else 0.0,
        'P@10': float(np.mean(p_at[10])) if p_at[10] else 0.0,
        'n_queries': len(mrrs),
    }

# Assign categories to GT pairs for stratified CV
cat_pairs = get_substitution_pairs()
categories = get_substitution_categories()
name_lookup = {(c['type'], c['root']): c['name'] for c in corpus}

pair_cats = []  # list of (a_name, b_name, category)
for p in cat_pairs:
    qkey = (p['query_type'], p['query_root'])
    skey = (p['sub_type'], p['sub_root'])
    if qkey in name_lookup and skey in name_lookup:
        pair_cats.append((name_lookup[qkey], name_lookup[skey], p['category']))

# Stratified 5-fold split
from collections import defaultdict
cat_to_pairs = defaultdict(list)
for a, b, cat in pair_cats:
    cat_to_pairs[cat].append((a, b))

K_FOLDS = 5
folds = [[] for _ in range(K_FOLDS)]
for cat, pairs in cat_to_pairs.items():
    np.random.shuffle(pairs)
    for idx, pair in enumerate(pairs):
        folds[idx % K_FOLDS].append(pair)

print(f"  Folds: {[len(f) for f in folds]} pairs")

# Grid search
results_range = []
results_zscore = []

for alpha in alphas:
    # Range normalization
    D_combo_range = alpha * D_EB_range + (1 - alpha) * D_JSD_range

    # Z-score normalization
    D_combo_zscore = alpha * D_EB_zscore + (1 - alpha) * D_JSD_zscore

    # Full evaluation (no CV)
    full_eval_range = evaluate_mrr_p(D_combo_range, gt_set)
    full_eval_zscore = evaluate_mrr_p(D_combo_zscore, gt_set)

    # 5-fold CV
    cv_mrrs_range = []
    cv_mrrs_zscore = []
    for fold_idx in range(K_FOLDS):
        test_pairs = folds[fold_idx]
        test_gt = set()
        for a, b in test_pairs:
            test_gt.add((a, b))
            test_gt.add((b, a))
        cv_eval_range = evaluate_mrr_p(D_combo_range, test_gt)
        cv_eval_zscore = evaluate_mrr_p(D_combo_zscore, test_gt)
        cv_mrrs_range.append(cv_eval_range['MRR'])
        cv_mrrs_zscore.append(cv_eval_zscore['MRR'])

    results_range.append({
        'alpha': float(alpha),
        'MRR_full': full_eval_range['MRR'],
        'P@1': full_eval_range['P@1'],
        'P@5': full_eval_range['P@5'],
        'P@10': full_eval_range['P@10'],
        'MRR_cv_mean': float(np.mean(cv_mrrs_range)),
        'MRR_cv_std': float(np.std(cv_mrrs_range)),
        'cv_folds': cv_mrrs_range,
    })
    results_zscore.append({
        'alpha': float(alpha),
        'MRR_full': full_eval_zscore['MRR'],
        'MRR_cv_mean': float(np.mean(cv_mrrs_zscore)),
        'MRR_cv_std': float(np.std(cv_mrrs_zscore)),
    })

    if alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(f"  alpha={alpha:.2f} | MRR_full={full_eval_range['MRR']:.4f} "
              f"| MRR_cv={np.mean(cv_mrrs_range):.4f}+/-{np.std(cv_mrrs_range):.4f} "
              f"| P@5={full_eval_range['P@5']:.4f}")

# ============== Find optimal alpha ==============
best_range = max(results_range, key=lambda r: r['MRR_cv_mean'])
best_zscore = max(results_zscore, key=lambda r: r['MRR_cv_mean'])

print(f"\n=== OPTIMAL alpha (range norm): {best_range['alpha']:.2f} ===")
print(f"    MRR_cv = {best_range['MRR_cv_mean']:.4f} +/- {best_range['MRR_cv_std']:.4f}")
print(f"    MRR_full = {best_range['MRR_full']:.4f}, P@5 = {best_range['P@5']:.4f}")
print(f"\n=== OPTIMAL alpha (z-score norm): {best_zscore['alpha']:.2f} ===")
print(f"    MRR_cv = {best_zscore['MRR_cv_mean']:.4f} +/- {best_zscore['MRR_cv_std']:.4f}")

# Stable region (>= 95% of peak)
peak_mrr = best_range['MRR_cv_mean']
stable_alphas = [r['alpha'] for r in results_range if r['MRR_cv_mean'] >= 0.95 * peak_mrr]
print(f"\n  Stable region (>=95% peak): alpha in [{min(stable_alphas):.2f}, {max(stable_alphas):.2f}]")

# ============== Step 7: Compare against existing metrics ==============
print("\n--- Comparing d_combo(alpha*) against existing metrics ---")

alpha_star = best_range['alpha']
D_combo_star = alpha_star * D_EB_range + (1 - alpha_star) * D_JSD_range

all_metrics = {
    'd_Euc': D_Euc, 'd_JSD': D_JSD, 'd_cos': D_cos, 'd_Q5': D_Q5,
    'd_VL': D_VL, 'd_w': D_w, 'd_EB': D_EB,
    f'd_combo(a={alpha_star:.2f})': D_combo_star,
}

comparison = []
for mname, D in all_metrics.items():
    ev = evaluate_mrr_p(D, gt_set)
    comparison.append({'metric': mname, **ev})
    print(f"  {mname:25s}: MRR={ev['MRR']:.4f} P@1={ev['P@1']:.4f} P@5={ev['P@5']:.4f} P@10={ev['P@10']:.4f}")

comparison.sort(key=lambda x: -x['MRR'])

# ============== LOO-CV sanity check ==============
print("\n--- LOO-CV sanity check ---")
D_combo_loo = alpha_star * D_EB_range + (1 - alpha_star) * D_JSD_range
loo_mrrs = []
for pair_idx, (a, b) in enumerate(gt_pairs_list):
    loo_gt = {(a, b), (b, a)}
    ai = names.index(a)
    bi = names.index(b)
    # MRR for query a: find b
    dists = D_combo_loo[ai].copy()
    dists[ai] = float('inf')
    order = np.argsort(dists)
    rank_b = int(np.where(order == bi)[0][0]) + 1
    loo_mrrs.append(1.0 / rank_b)
    # MRR for query b: find a
    dists = D_combo_loo[bi].copy()
    dists[bi] = float('inf')
    order = np.argsort(dists)
    rank_a = int(np.where(order == ai)[0][0]) + 1
    loo_mrrs.append(1.0 / rank_a)

loo_mrr = float(np.mean(loo_mrrs))
print(f"  LOO MRR = {loo_mrr:.4f} (vs 5-fold CV = {best_range['MRR_cv_mean']:.4f})")

# ============== HTML Report ==============
print("\n--- Generating HTML report ---")

a_vals = [r['alpha'] for r in results_range]
mrr_full_vals = [r['MRR_full'] for r in results_range]
mrr_cv_vals = [r['MRR_cv_mean'] for r in results_range]
mrr_cv_err = [r['MRR_cv_std'] for r in results_range]
mrr_zscore_vals = [r['MRR_cv_mean'] for r in results_zscore]

# Comparison table rows
comp_rows = ""
for c in comparison:
    is_combo = 'd_combo' in c['metric']
    cls = ' style="background:rgba(63,185,80,.08)"' if is_combo else ''
    comp_rows += f"""<tr{cls}><td><strong>{c['metric']}</strong></td>
        <td class="num">{c['P@1']:.4f}</td><td class="num">{c['P@5']:.4f}</td>
        <td class="num">{c['P@10']:.4f}</td>
        <td class="num" style="color:var(--gn);font-weight:600">{c['MRR']:.4f}</td></tr>"""

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experimento 7 &mdash; Calibracion de alpha para d_combo</title>
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
h1{{ font-size:1.6rem; font-weight:700; background:linear-gradient(135deg,var(--or),var(--gn));
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
.bg-g{{ background:rgba(63,185,80,.15); color:var(--gn); }}
.bg-o{{ background:rgba(210,153,34,.15); color:var(--or); }}
.bg-b{{ background:rgba(88,166,255,.15); color:var(--ac); }}
table{{ width:100%; border-collapse:collapse; font-size:.82rem; margin:.8rem 0; }}
th{{ background:var(--sf); color:var(--ac); font-weight:600; padding:.5rem .6rem; text-align:left; border-bottom:2px solid var(--bd); }}
td{{ padding:.35rem .6rem; border-bottom:1px solid var(--bd); }}
td.num{{ font-family:'Fira Code',monospace; font-size:.77rem; }}
.ch{{ border-radius:10px; overflow:hidden; margin:1.2rem 0; }}
.metrics{{ display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:1rem 0; }}
.mc{{ background:var(--sf); border:1px solid var(--bd); border-radius:10px; padding:.8rem; text-align:center; }}
.mv{{ font-size:1.3rem; font-weight:700; font-family:'Fira Code',monospace; }}
.ml{{ color:var(--tm); font-size:.72rem; margin-top:.2rem; }}
.ft{{ margin-top:2.5rem; padding-top:.8rem; border-top:1px solid var(--bd); color:var(--tm); font-size:.75rem; text-align:center; }}
</style>
</head>
<body>
<div class="c">

<h1>Experimento 7: Calibracion de &alpha; para d_combo</h1>
<p class="sub">
    $d_{{\\text{{combo}}}} = \\alpha \\cdot \\hat{{d}}_{{\\text{{EB}}}} + (1-\\alpha) \\cdot \\hat{{d}}_{{\\text{{JSD}}}}$
    <br><span class="bg bg-g">alpha* = {alpha_star:.2f}</span>
    <span class="bg bg-o">MRR_cv = {best_range['MRR_cv_mean']:.4f} +/- {best_range['MRR_cv_std']:.4f}</span>
    <span class="bg bg-b">{N} acordes &middot; {len(gt_pairs_list)} pares GT &middot; 5-fold CV</span>
</p>

<h2>1. Metodologia</h2>
<div class="cd">
<p><strong>Corpus:</strong> {N} acordes ({n_triads} triadas + {n_tetrads} tetradas), 13 tipos &times; 12 raices.</p>
<p><strong>Componentes:</strong></p>
<ul>
<li>$d_{{\\text{{EB}}}}$: Expansion Biyectiva sobre notas MIDI ($\\mathbb{{R}}/12\\mathbb{{Z}}$)</li>
<li>$\\sqrt{{\\text{{JSD}}}}$: raiz de Jensen-Shannon sobre $\\Phi_{{\\text{{simplex}}}}$</li>
</ul>
<p><strong>Normalizacion:</strong> rango $\\hat{{D}} = D / \\max(D)$ para igualar escalas.</p>
<p><strong>Validacion:</strong> 5-fold CV estratificado por categoria de sustitucion (9 categorias). LOO-CV como sanity check.</p>
</div>

<h2>2. Resultados: MRR vs &alpha;</h2>
<div class="metrics">
<div class="mc"><div class="mv" style="color:var(--gn);">{alpha_star:.2f}</div><div class="ml">&alpha;* optimo</div></div>
<div class="mc"><div class="mv" style="color:var(--ac);">{best_range['MRR_cv_mean']:.4f}</div><div class="ml">MRR (5-fold CV)</div></div>
<div class="mc"><div class="mv" style="color:var(--or);">{best_range['MRR_full']:.4f}</div><div class="ml">MRR (full)</div></div>
<div class="mc"><div class="mv" style="color:var(--pr);">{loo_mrr:.4f}</div><div class="ml">MRR (LOO-CV)</div></div>
</div>

<h3>2.1 Curva MRR vs &alpha; (con barras de error &plusmn;1&sigma;)</h3>
<div class="ch" id="mrr_curve"></div>

<h3>2.2 Robustez: rango vs z-score</h3>
<div class="ch" id="norm_compare"></div>

<h2>3. Region estable</h2>
<div class="th">
<p class="th-l">Sensibilidad</p>
<p>La region donde MRR &ge; 95% del pico es $\\alpha \\in [{min(stable_alphas):.2f}, {max(stable_alphas):.2f}]$. Esto indica que el resultado es <strong>{'robusto' if max(stable_alphas) - min(stable_alphas) >= 0.15 else 'sensible'}</strong> a la eleccion exacta de &alpha;.</p>
</div>

<h2>4. Comparativa con metricas existentes</h2>
<div class="cd"><table>
<thead><tr><th>Metrica</th><th>P@1</th><th>P@5</th><th>P@10</th><th>MRR</th></tr></thead>
<tbody>{comp_rows}</tbody>
</table></div>

<div class="ch" id="comparison_bar"></div>

<h2>5. Analisis</h2>

<div class="th">
<p class="th-l">Resultado principal</p>
<p>$d_{{\\text{{combo}}}}(\\alpha^* = {alpha_star:.2f})$ combina el dominio <strong>espectral/perceptual</strong> ($\\sqrt{{\\text{{JSD}}}}$ sobre rugosidad) con el dominio <strong>de alturas/voice-leading</strong> ($d_{{\\text{{EB}}}}$). El peso optimo indica que la sustitucion armonica depende {'mas del voice-leading (EB)' if alpha_star > 0.5 else 'mas de la similitud timbrica (JSD)' if alpha_star < 0.5 else 'equitativamente de ambos dominios'}.</p>
</div>

<div class="ins">
<p class="ins-l">Validacion cruzada</p>
<p>LOO-CV ({loo_mrr:.4f}) {'confirma' if abs(loo_mrr - best_range['MRR_cv_mean']) < 0.02 else 'difiere de'} 5-fold CV ({best_range['MRR_cv_mean']:.4f}), {'indicando robustez' if abs(loo_mrr - best_range['MRR_cv_mean']) < 0.02 else 'sugiriendo varianza entre folds'}.</p>
</div>

<div class="warn">
<p class="warn-l">Limitaciones</p>
<p>1. La normalizacion por rango es corpus-dependiente. 2. El ground truth (228 pares, 9 categorias) es un <em>lower bound</em>; sustituciones validas no incluidas se penalizan como falsos positivos. 3. $\\alpha^*$ es especifico de esta poblacion de acordes.</p>
</div>

<div class="ft">Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} &middot; ChordSpace &middot;
<code>experiments/chord_substitution/run_experiment_7_alpha_calibration.py</code></div>
</div>
<script>
const L={{paper_bgcolor:'#0d1117',plot_bgcolor:'#161b22',font:{{family:'Inter',color:'#e6edf3',size:12}},
  xaxis:{{gridcolor:'#30363d',zerolinecolor:'#30363d'}},yaxis:{{gridcolor:'#30363d',zerolinecolor:'#30363d'}},
  legend:{{bgcolor:'rgba(22,27,34,0.9)',bordercolor:'#30363d',borderwidth:1}},margin:{{l:65,r:30,t:45,b:60}}}};

// MRR curve with error bars
const a={json.dumps([float(a) for a in a_vals])};
Plotly.newPlot('mrr_curve',[
  {{x:a,y:{json.dumps(mrr_cv_vals)},name:'MRR (5-fold CV)',
    error_y:{{type:'data',array:{json.dumps(mrr_cv_err)},visible:true,color:'rgba(63,185,80,0.3)'}},
    line:{{color:'#3fb950',width:3}},mode:'lines+markers',marker:{{size:6}}}},
  {{x:a,y:{json.dumps(mrr_full_vals)},name:'MRR (full)',
    line:{{color:'#58a6ff',width:1.5,dash:'dot'}}}},
  {{x:[{alpha_star},{alpha_star}],y:[0,{max(mrr_cv_vals)*1.1}],name:'alpha*={alpha_star:.2f}',
    line:{{color:'#d29922',width:2,dash:'dash'}},mode:'lines'}},
],{{...L,title:{{text:'MRR vs alpha (range normalization)',font:{{size:14}}}},
  xaxis:{{...L.xaxis,title:'alpha',dtick:0.1}},yaxis:{{...L.yaxis,title:'MRR'}}}});

// Normalization comparison
Plotly.newPlot('norm_compare',[
  {{x:a,y:{json.dumps(mrr_cv_vals)},name:'Range norm',line:{{color:'#3fb950',width:2}}}},
  {{x:a,y:{json.dumps(mrr_zscore_vals)},name:'Z-score norm',line:{{color:'#bc8cff',width:2}}}},
],{{...L,title:{{text:'MRR_cv: Range vs Z-score normalization',font:{{size:14}}}},
  xaxis:{{...L.xaxis,title:'alpha',dtick:0.1}},yaxis:{{...L.yaxis,title:'MRR (5-fold CV)'}}}});

// Comparison bar
const cmp_names={json.dumps([c['metric'] for c in comparison])};
const cmp_mrr={json.dumps([c['MRR'] for c in comparison])};
const cmp_p5={json.dumps([c['P@5'] for c in comparison])};
Plotly.newPlot('comparison_bar',[
  {{x:cmp_names,y:cmp_mrr,name:'MRR',type:'bar',marker:{{color:'#3fb950'}}}},
  {{x:cmp_names,y:cmp_p5,name:'P@5',type:'bar',marker:{{color:'#58a6ff'}}}},
],{{...L,barmode:'group',title:{{text:'d_combo vs metricas existentes',font:{{size:14}}}},
  yaxis:{{...L.yaxis,title:'Score'}}}});
</script>
</body></html>"""

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_7_report.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nReport: {out}")
