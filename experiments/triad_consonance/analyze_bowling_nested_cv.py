import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, GridSearchCV, cross_validate
from scipy.stats import pearsonr

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("bowling_results.csv")

def evaluate_subset_nested_cv(subset_df, name, n_splits=5, n_repeats=10):
    """
    Evaluates 1D (Linear & Poly) vs 12D (Ridge) using Repeated K-Fold Cross-Validation.
    For models requiring tuning (Poly degree, Ridge alpha), it uses GridSearchCV
    within the CV loop (Nested CV) to prevent data leakage during hyperparameter selection.
    """
    ratings = subset_df['rating'].values
    scalar_X = subset_df[['scalar_roughness']].values
    vector_cols = [f'v{i}' for i in range(12)]
    vector_X = subset_df[vector_cols].values

    # Define validation strategy: Repeated K-Fold for stable variance estimates
    cv_outer = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    cv_inner = 3 # 3-fold for inner hyperparameter tuning

    results = {'name': name, 'n': len(subset_df)}

    # --- 1D Baseline: Simple Linear ---
    lr = LinearRegression()
    scores_1d_lin = cross_validate(lr, scalar_X, ratings, cv=cv_outer, scoring='r2', return_estimator=True)
    results['r2_1d_lin_mean'] = np.mean(scores_1d_lin['test_score'])
    results['r2_1d_lin_std'] = np.std(scores_1d_lin['test_score'])

    # --- 1D Baseline: Polynomial (Tuned Degree) ---
    poly_pipe = Pipeline([('poly', PolynomialFeatures()), ('lr', LinearRegression())])
    poly_param_grid = {'poly__degree': [2, 3, 4]} 
    poly_search = GridSearchCV(poly_pipe, poly_param_grid, cv=cv_inner, scoring='r2')
    scores_1d_poly = cross_validate(poly_search, scalar_X, ratings, cv=cv_outer, scoring='r2')
    
    results['r2_1d_poly_mean'] = np.mean(scores_1d_poly['test_score'])
    results['r2_1d_poly_std'] = np.std(scores_1d_poly['test_score'])

    # --- 12D Model: Ridge (Tuned Alpha) ---
    ridge = Ridge()
    ridge_param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    ridge_search = GridSearchCV(ridge, ridge_param_grid, cv=cv_inner, scoring='r2')
    scores_12d = cross_validate(ridge_search, vector_X, ratings, cv=cv_outer, scoring='r2', return_estimator=True)
    
    results['r2_12d_mean'] = np.mean(scores_12d['test_score'])
    results['r2_12d_std'] = np.std(scores_12d['test_score'])

    # Extract the most frequently selected best alpha across all outer folds
    best_alphas = [est.best_params_['alpha'] for est in scores_12d['estimator']]
    results['best_alpha_mode'] = max(set(best_alphas), key=best_alphas.count)

    return results

print("===== RUNNING NESTED CV METHODOLOGICAL EVALUATION =====")
print("Evaluating models with RepeatedKFold (5 splits, 10 repeats) and inner GridSearchCV...")

subsets = [
    evaluate_subset_nested_cv(df, "All Chords"),
    evaluate_subset_nested_cv(df[df['k'] == 2], "Dyads (k=2)"),
    evaluate_subset_nested_cv(df[df['k'] == 3], "Triads (k=3)"),
    evaluate_subset_nested_cv(df[df['k'] == 4], "Tetrads (k=4)")
]

print("\n--- R² RESULTS BY CARDINALITY (Mean ± SD) ---")
print(f"{'Subset':<15} | {'N':<5} | {'1D Lin R²':<18} | {'1D Poly R²':<18} | {'12D Ridge R²':<18} | {'Mode Alpha':<10}")
print("-" * 90)
for sub in subsets:
    lin_str = f"{sub['r2_1d_lin_mean']:.3f} ± {sub['r2_1d_lin_std']:.3f}"
    poly_str = f"{sub['r2_1d_poly_mean']:.3f} ± {sub['r2_1d_poly_std']:.3f}"
    ridge_str = f"{sub['r2_12d_mean']:.3f} ± {sub['r2_12d_std']:.3f}"
    print(f"{sub['name']:<15} | {sub['n']:<5} | {lin_str:<18} | {poly_str:<18} | {ridge_str:<18} | {sub['best_alpha_mode']:<10}")

print("\nNested CV evaluation complete. Output ready for thesis report.")
