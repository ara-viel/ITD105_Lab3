# regression/model_B_repeated.py
"""
Model B - Regression (Linear Regression + repeated random train/test splits)
Dataset: PRSA (Beijing PM2.5)
Saves:
 - model file: regression_model_B_repeated.joblib
 - metrics file: regression_model_B_repeated_metrics.json

Run:
    python regression/model_B_repeated.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PRSA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
OUT_MODEL = Path(__file__).resolve().parent / "regression_model_B_repeated.joblib"
OUT_METRICS = Path(__file__).resolve().parent / "regression_model_B_repeated_metrics.json"

def load_prsa(maybe_path=None, nrows=None):
    if maybe_path:
        df = pd.read_csv(maybe_path)
    else:
        df = pd.read_csv(PRSA_URL, nrows=nrows)
    return df

def prepare_prsa(df):
    required = ['year','month','day','hour','pm2.5','DEWP','TEMP','PRES','cbwd','Iws','Is','Ir']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in PRSA dataset: {missing}")
    df = df.dropna(subset=['pm2.5'])
    X = df[['year','month','day','hour','DEWP','TEMP','PRES','Iws','Is','Ir','cbwd']].copy()
    y = df['pm2.5'].astype(float).copy()
    numeric_features = ['year','month','day','hour','DEWP','TEMP','PRES','Iws','Is','Ir']
    categorical_features = ['cbwd']
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preproc = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return X, y, preproc

def main():
    print("Loading PRSA dataset (first 20000 rows by default)...")
    df = load_prsa(None, nrows=20000)
    X, y, preproc = prepare_prsa(df)

    repeats = 10
    test_size = 0.2
    random_state_base = 42

    mse_list, mae_list, r2_list = [], [], []

    print(f"Running {repeats} repeated random splits...")
    for i in range(repeats):
        rs = random_state_base + i
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs)
        pipeline = Pipeline([('preproc', preproc), ('reg', LinearRegression())])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mse_list.append(float(mean_squared_error(y_test, y_pred)))
        mae_list.append(float(mean_absolute_error(y_test, y_pred)))
        r2_list.append(float(r2_score(y_test, y_pred)))

    metrics = {
        "method": "RepeatedRandomSplits",
        "repeats": repeats,
        "test_size": test_size,
        "mse_mean": float(np.mean(mse_list)),
        "mse_std": float(np.std(mse_list)),
        "mae_mean": float(np.mean(mae_list)),
        "mae_std": float(np.std(mae_list)),
        "r2_mean": float(np.mean(r2_list)),
        "r2_std": float(np.std(r2_list)),
    }

    with open(OUT_METRICS, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # Fit final pipeline on full (subsampled) data and save
    final_pipeline = Pipeline([('preproc', preproc), ('reg', LinearRegression())])
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, OUT_MODEL)

    print(f"Saved model to {OUT_MODEL}")
    print(f"Saved metrics to {OUT_METRICS}")
    print("Done.")

if __name__ == "__main__":
    main()
