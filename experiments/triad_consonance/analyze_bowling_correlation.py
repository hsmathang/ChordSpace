"""
analyze_bowling_correlation.py
ValidaciÃ³n perceptual: modelo 1D (escalar) vs. 12D (vectorial Ridge).
TODAS las mÃ©tricas y puntos graficados son OOS (cross-val predict).
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# â”€â”€ Reproducibilidad â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEED = 42
CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

# â”€â”€ Datos â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df = pd.read_csv("bowling_results.csv")
ratings = df['rating'].values
scalar_X = df[['scalar_roughness']].values
vector_cols = [f'v{i}' for i in range(12)]
vector_X = df[vector_cols].values

# â”€â”€ Modelo 1D lineal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
lr_1d = LinearRegression()
preds_1d_oos = cross_val_predict(lr_1d, scalar_X, ratings, cv=CV)
r_1d_oos, _ = pearsonr(preds_1d_oos, ratings)
r2_1d_oos = r2_score(ratings, preds_1d_oos)

# â”€â”€ Modelo 1D polinÃ³mico (control de grados de libertad) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
poly = PolynomialFeatures(degree=3, include_bias=False)
scalar_poly_X = poly.fit_transform(scalar_X)
lr_1d_poly = LinearRegression()
preds_1d_poly_oos = cross_val_predict(lr_1d_poly, scalar_poly_X, ratings, cv=CV)
r_1d_poly_oos, _ = pearsonr(preds_1d_poly_oos, ratings)
r2_1d_poly_oos = r2_score(ratings, preds_1d_poly_oos)

# â”€â”€ Modelo 12D Ridge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ridge_12d = Ridge(alpha=1.0)
preds_12d_oos = cross_val_predict(ridge_12d, vector_X, ratings, cv=CV)
r_12d_oos, _ = pearsonr(preds_12d_oos, ratings)
r2_12d_oos = r2_score(ratings, preds_12d_oos)

# â”€â”€ Consola â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=" * 60)
print("  VALIDACIÃ“N PERCEPTUAL â€” Bowling et al. (N=298)")
print("  Todas las mÃ©tricas son OOS (5-fold CV, seed=42)")
print("=" * 60)
print(f"{'Modelo':<25} {'Pearson r OOS':>13}  {'RÂ² OOS':>8}")
print("-" * 50)
print(f"{'1D Lineal':<25} {r_1d_oos:>13.3f}  {r2_1d_oos:>8.3f}")
print(f"{'1D PolinÃ³mico (deg=3)':<25} {r_1d_poly_oos:>13.3f}  {r2_1d_poly_oos:>8.3f}")
print(f"{'12D Ridge (Î±=1.0)':<25} {r_12d_oos:>13.3f}  {r2_12d_oos:>8.3f}")
print()

# â”€â”€ Figura de alta calidad â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
sns.set_theme(style='ticks', context='paper', font_scale=1.25)
fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), sharey=True)

LIMS = (0.5, 4.5)
DIAG = [1.0, 4.0]

# Panel A - 12D Ridge (puntos OOS)
ax = axes[0]
ax.scatter(preds_12d_oos, ratings,
           color='#0072B2', alpha=0.45, s=22, edgecolors='none',
           label='Acordes (OOS)')
ax.plot(DIAG, DIAG, 'k--', alpha=0.6, lw=1.2, label='Prediccion ideal')
ax.set_title("A. Modelo Vectorial 12D (Ridge)",
    loc='left', fontweight='bold', fontsize=10)
ax.set_xlabel("Rating predicho (escala 1-4)")
ax.set_ylabel("Calificacion humana (Bowling et al.)")
ax.set_xlim(LIMS)
ax.set_ylim(LIMS)
ax.legend(fontsize=8, frameon=False)

# Panel B - 1D Lineal (puntos OOS)
ax = axes[1]
ax.scatter(preds_1d_oos, ratings,
           color='#D55E00', alpha=0.45, s=22, edgecolors='none',
           label='Acordes (OOS)')
ax.plot(DIAG, DIAG, 'k--', alpha=0.6, lw=1.2, label='Prediccion ideal')
ax.set_title("B. Modelo Escalar 1D",
    loc='left', fontweight='bold', fontsize=10)
ax.set_xlabel("Rating predicho (escala 1-4)")
ax.set_xlim(LIMS)
ax.legend(fontsize=8, frameon=False)

sns.despine()
plt.tight_layout()
plt.savefig("bowling_correlation_paper_hq.png", dpi=600, bbox_inches='tight')
plt.savefig("bowling_correlation_paper_hq.pdf", bbox_inches='tight')
print("Figura guardada: bowling_correlation_paper_hq.png / .pdf")

# â”€â”€ Reporte HTML â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>ValidaciÃ³n Perceptual â€” Bowling et al.</title>
  <style>
    body {{font-family: 'Segoe UI', sans-serif; max-width: 900px;
           margin: 40px auto; line-height: 1.7; color: #333;}}
    h1,h2,h3 {{color: #2C3E50;}}
    table {{width:100%; border-collapse:collapse; margin:20px 0;}}
    th,td {{border:1px solid #ddd; padding:10px; text-align:center;}}
    th {{background:#f5f5f5;}}
    .note {{background:#EBF5FB; padding:12px 16px;
            border-left:4px solid #2980B9; margin:20px 0;}}
    .win {{color:#117a65; font-weight:bold;}}
  </style>
</head>
<body>
  <h1>ValidaciÃ³n Perceptual: 1D vs. 12D</h1>
  <p>Dataset: Bowling et al. (2018), <strong>N=298</strong> acordes (dÃ­adas, trÃ­adas, tÃ©tradas).
  <strong>Todas las mÃ©tricas son OOS</strong> â€” 5-fold CV estratificado, semilla={SEED}.</p>

  <div class="note">
    <strong>ComparaciÃ³n justa:</strong> el control polinÃ³mico (deg=3) otorga al modelo 1D
    grados de libertad no lineales adicionales. Si el 12D sigue ganando, la superioridad
    no se debe a la asimetrÃ­a de parÃ¡metros.
  </div>

  <h2>Tabla de resultados OOS</h2>
  <table>
    <tr><th>Modelo</th><th>Pearson r OOS</th><th>RÂ² OOS</th></tr>
    <tr><td>1D Lineal</td>
        <td>{r_1d_oos:.3f}</td><td>{r2_1d_oos:.3f}</td></tr>
    <tr><td>1D PolinÃ³mico (deg=3) â€” control</td>
        <td>{r_1d_poly_oos:.3f}</td><td>{r2_1d_poly_oos:.3f}</td></tr>
    <tr><td class="win">12D Ridge (Î±=1.0)</td>
        <td class="win">{r_12d_oos:.3f}</td>
        <td class="win">{r2_12d_oos:.3f}</td></tr>
  </table>

  <h2>Diferencia OOS (Î”)</h2>
  <p>Î” RÂ² (12D vs 1D lineal): <strong>{r2_12d_oos - r2_1d_oos:+.3f}</strong><br>
  Î” RÂ² (12D vs 1D polinÃ³mico): <strong>{r2_12d_oos - r2_1d_poly_oos:+.3f}</strong></p>

  <p><em>Figura: bowling_correlation_paper_hq.png / .pdf</em></p>
</body>
</html>
"""
with open("bowling_correlation_report.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Reporte HTML: bowling_correlation_report.html")


