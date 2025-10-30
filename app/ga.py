import numpy as np
import random
import time
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder


def _is_discrete_series(s, threshold=20):
    return (s.dtype == object) or (s.nunique() < threshold)

def compute_precomputed_metrics(df, features, target_col, sample_n=20000, discrete_threshold=20):
    X_full = df[features]
    y_full = df[target_col]

    nrows = len(df)
    if nrows > sample_n:
        df_sample = df.sample(n=sample_n, random_state=42)
        X = df_sample[features].values
        y = df_sample[target_col].values
    else:
        X = X_full.values
        y = y_full.values

    discrete_target = _is_discrete_series(y_full, threshold=discrete_threshold)
    if discrete_target:
        if y.dtype.kind not in 'biufc':
            try:
                le = LabelEncoder()
                y = le.fit_transform(y)
            except Exception:
                pass

    try:
        if discrete_target:
            mi_all = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
        else:
            mi_all = mutual_info_regression(X, y, discrete_features='auto', random_state=42)
    except Exception:
        mi_all = np.zeros(X.shape[1], dtype=float)

    try:
        corr = np.corrcoef(X, rowvar=False)
        corr_abs = np.nan_to_num(np.abs(corr))
    except Exception:
        corr_abs = np.zeros((X.shape[1], X.shape[1]), dtype=float)

    return np.asarray(mi_all, dtype=float), np.asarray(corr_abs, dtype=float)

def compute_feature_importance_table(df, features, target_col, sample_n=20000):

    import pandas as pd
    X_full = df[features]
    y_full = df[target_col]


    if len(df) > sample_n:
        df_sample = df.sample(n=sample_n, random_state=42)
        X = df_sample[features].values
        y = df_sample[target_col].values
    else:
        X = X_full.values
        y = y_full.values

    discrete_target = _is_discrete_series(y_full)
    if discrete_target and y.dtype.kind not in 'biufc':
        try:
            le = LabelEncoder()
            y = le.fit_transform(y)
        except Exception:
            pass

    try:
        if discrete_target:
            mi_all = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
        else:
            mi_all = mutual_info_regression(X, y, discrete_features='auto', random_state=42)
    except Exception:
        mi_all = np.zeros(X.shape[1], dtype=float)

    corr_with_target = []
    for col in features:
        try:
            corr = df[col].corr(df[target_col])
            corr_with_target.append(float(np.nan_to_num(corr)))
        except Exception:
            corr_with_target.append(0.0)

    import pandas as pd
    tbl = pd.DataFrame({
        "feature": features,
        "mi": mi_all,
        "corr_with_target": corr_with_target,
        "n_unique": [int(df[f].nunique()) for f in features]
    })
    tbl = tbl.sort_values(by="mi", ascending=False).reset_index(drop=True)
    return tbl
