from sklearn.feature_selection import chi2
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def chi2_feature_selection(df, features, target_col, k=5):

    try:
        X_raw = df[features].astype(float).values
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X_raw)

        y = df[target_col].values

        k = min(k, len(features))

        chi2_vals, p = chi2(X, y)
        idxs = np.argsort(chi2_vals)[-k:]
        return [features[i] for i in idxs if chi2_vals[i] > 0]
    except Exception:
        return []
