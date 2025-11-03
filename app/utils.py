import pandas as pd
import numpy as np

def get_target_column(df):

    candidates = [
        'target', 'label', 'class', 'outcome', 'y', 'output', 'result',
        'survived', 'diagnosis', 'response', 'is_fraud', 'clicked'
    ]
    cols = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand in cols:
            return df.columns[cols.index(cand)]
    if len(df.columns) > 1:
        return df.columns[-1]
    return df.columns[0] if len(df.columns) == 1 else None

def preprocess_data(df,corr_threshold, missing_threshold=0.6, verbose=False):

    df = df.copy()
    df = df.drop_duplicates()
    target_col = get_target_column(df)
    if target_col is None:
        raise ValueError("لم يتم التعرف على عمود الهدف تلقائياً. الرجاء التحقق من الملف.")

    miss_frac = df.isnull().mean()
    drop_cols = miss_frac[miss_frac > missing_threshold].index.tolist()
    if verbose and drop_cols:
        print("Dropping columns with too many missing:", drop_cols)
    df = df.drop(columns=drop_cols)

    features = []
    for col in df.columns:
        if col == target_col:
            continue

        if df[col].dtype.kind in 'biufc':
            med = df[col].median()
            df[col] = df[col].fillna(med)
        else:
            mode = df[col].mode(dropna=True)
            fill_val = mode.iloc[0] if mode.size > 0 else "NA"
            df[col] = df[col].fillna(fill_val)

    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == object or df[col].dtype.name == 'category':
            df[col], _ = pd.factorize(df[col], sort=True)
        features.append(col)

    nunique = df[features].nunique()
    zero_var = nunique[nunique <= 1].index.tolist()
    if zero_var:
        if verbose:
            print("Dropping zero variance columns:", zero_var)
        df = df.drop(columns=zero_var)
        features = [f for f in features if f not in zero_var]


    return df, features, target_col
