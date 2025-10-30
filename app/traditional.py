import numpy as np
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


def lasso_feature_selection(df, features, target_col, alpha=0.01):

    try:
        X = df[features].values
        y = df[target_col].values
        lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        lasso.fit(X, y)
        selected = [features[i] for i,coef in enumerate(lasso.coef_) if abs(coef) > 1e-6]
        return selected
    except Exception:
        return []
