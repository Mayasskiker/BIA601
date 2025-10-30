import numpy as np
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

def pca_feature_selection(df, features, n_components=3):

    try:
        X = df[features].values
        n_components = min(n_components, X.shape[1], X.shape[0]-1)
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X)
        top_features = set()
        for comp in pca.components_:
            idx = int(np.argmax(np.abs(comp)))
            top_features.add(features[idx])
        return list(top_features)
    except Exception:
        return []