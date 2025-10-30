import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score

def _is_classification(y):

    try:
        if isinstance(y, pd.Series):
            y_arr = y.values
        else:
            y_arr = y

        if y_arr.dtype == object:
            return True
        n_unique = int(np.unique(y_arr).size)
        return n_unique <= 20 or n_unique <= (len(y_arr) * 0.05)
    except Exception:
        return True

def get_final_evaluation(df_train, df_test, feature_list, target_col, random_state=42):

    start = time.time()

    if not feature_list or len(feature_list) == 0:
        return {
            "num_features": 0, "metric": None, "score_mean": None,
            "score_std": None, "time_seconds": 0.0, "error": "no features"
        }

    valid_features = [f for f in feature_list if f in df_train.columns]
    if not valid_features:
         return {
            "num_features": 0, "metric": None, "score_mean": None,
            "score_std": None, "time_seconds": 0.0, "error": "features not found"
        }

    X_train = df_train[valid_features].values
    y_train = df_train[target_col].values
    X_test = df_test[valid_features].values
    y_test = df_test[target_col].values

    model = None
    scoring_metric = None
    score = 0.0

    try:
        is_clf = _is_classification(y_train)

        if is_clf:
            if y_train.dtype.kind not in 'biufc':
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                try:
                    y_test = le.transform(y_test)
                except ValueError:
                    known_labels_mask = np.isin(y_test, le.classes_)
                    X_test = X_test[known_labels_mask]
                    y_test = y_test[known_labels_mask]
                    y_test = le.transform(y_test)

            n_classes = len(np.unique(y_train))
            model = LogisticRegression(max_iter=500, solver='liblinear', random_state=random_state)

            if n_classes == 2:
                scoring_metric = 'roc_auc'
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_pred_proba)
            else:
                scoring_metric = 'accuracy'
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)

        else: # (Regression)
            scoring_metric = 'r2'
            model = Ridge(random_state=random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)

    except Exception as e:
        return {
            "num_features": len(valid_features), "metric": scoring_metric,
            "score_mean": None, "score_std": None,
            "time_seconds": time.time() - start, "error": str(e)
        }

    return {
        "num_features": len(valid_features),
        "metric": scoring_metric,
        "score_mean": float(score),
        "score_std": 0.0,
        "time_seconds": time.time() - start
    }