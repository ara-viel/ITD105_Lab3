# regression/model_A_split.py
"""
Model A - Regression (Linear Regression + single Train/Test split 80:20)
Dataset: PRSA (Beijing PM2.5)
Saves:
 - model file: regression_model_A_split.joblib
 - metrics file: regression_model_A_split_metrics.json

Run:
    python regression/model_A_split.py
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
OUT_MODEL = Path(__file__).resolve().parent / "regression_model_A_split.joblib"
OUT_METRICS = Path(__file__).resolve().parent / "regression_model_A_split_metrics.json"

def load_prsa(maybe_path=None, nrows=None):
    if maybe_path:
        df = pd.read_csv(maybe_path)
    else:
        df = pd.read_csv(PRSA_URL, nrows=nrows)
    return df

def prepare_prsa(df):
    # keep necessary columns
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
    print("Loading PRSA dataset (first 20000 rows for speed)...")
    df = load_prsa(None, nrows=20000)
    X, y, preproc = prepare_prsa(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([('preproc', preproc), ('reg', LinearRegression())])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = dict(method="SingleTrainTest", test_size=0.2, mse=mse, mae=mae, r2=r2)
    with open(OUT_METRICS, "w") as fh:
        json.dump(metrics, fh, indent=2)

    joblib.dump(pipeline, OUT_MODEL)
    print(f"Saved model to {OUT_MODEL}")
    print(f"Saved metrics to {OUT_METRICS}")
    print("Done.")

if __name__ == "__main__":
    main()
