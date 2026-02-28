import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_predict

df = pd.read_csv('experiments/triad_consonance/bowling_results.csv')
ratings = df['rating'].values

scalar_X = df[['scalar_roughness']].values
lr_1d = LinearRegression()
preds_1d_cv = cross_val_predict(lr_1d, scalar_X, ratings, cv=5)
r_1d_cv_pearson, _ = pearsonr(preds_1d_cv, ratings)
r2_1d = lr_1d.fit(scalar_X, ratings).score(scalar_X, ratings)
print("1D: r_cv =", r_1d_cv_pearson, "R2 =", r2_1d)

vector_X = df[[f'v{i}' for i in range(12)]].values
ridge_12d = Ridge(alpha=1.0)
preds_12d_cv = cross_val_predict(ridge_12d, vector_X, ratings, cv=5)
r_12d_cv_pearson, _ = pearsonr(preds_12d_cv, ratings)
ridge_12d.fit(vector_X, ratings)
r2_12d = ridge_12d.score(vector_X, ratings)
print("12D: r_cv =", r_12d_cv_pearson, "R2 =", r2_12d)
