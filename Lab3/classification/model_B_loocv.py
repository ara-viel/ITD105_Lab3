# classification/model_B_loocv.py
"""
Model B - Classification (Logistic Regression + Leave-One-Out CV)
Dataset: Pima Indians Diabetes
Saves:
 - model file: classification_model_B_loocv.joblib
 - metrics file: classification_model_B_loocv_metrics.json

Run:
    python classification/model_B_loocv.py
Notes:
 - LOOCV can be slow (n ~ 768); expect more time than K-Fold.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report

PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
OUT_MODEL = Path(__file__).resolve().parent / "classification_model_B_loocv.joblib"
OUT_METRICS = Path(__file__).resolve().parent / "classification_model_B_loocv_metrics.json"

def load_pima(maybe_path=None):
    if maybe_path:
        df = pd.read_csv(maybe_path, header=None)
    else:
        df = pd.read_csv(PIMA_URL, header=None)
    if df.shape[1] == 9:
        df.columns = ['preg','plas','pres','skin','test','mass','pedi','age','class']
    return df

def preprocess_pipeline():
    numeric_cols = ['preg','plas','pres','skin','test','mass','pedi','age']
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preproc = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols)
    ])
    return preproc

def main():
    print("Loading Pima dataset...")
    df = load_pima(None)
    X = df[['preg','plas','pres','skin','test','mass','pedi','age']]
    y = df['class'].astype(int)

    preproc = preprocess_pipeline()
    clf = LogisticRegression(solver='liblinear', max_iter=1000)
    pipeline = Pipeline([('preproc', preproc), ('clf', clf)])

    loo = LeaveOneOut()
    print("Running Leave-One-Out cross-validation (may take longer)...")
    acc_scores = cross_val_score(pipeline, X, y, cv=loo, scoring='accuracy', n_jobs=-1)

    y_proba = cross_val_predict(pipeline, X, y, cv=loo, method='predict_proba', n_jobs=-1)[:,1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "method": "LOOCV",
        "n_splits": len(y),
        "accuracy_mean": float(np.mean(acc_scores)),
        "accuracy_std": float(np.std(acc_scores)),
    }
    try:
        metrics["log_loss"] = float(log_loss(y, y_proba))
    except Exception as e:
        metrics["log_loss"] = None
        metrics["log_loss_error"] = str(e)

    try:
        metrics["roc_auc"] = float(roc_auc_score(y, y_proba))
    except Exception as e:
        metrics["roc_auc"] = None
        metrics["roc_auc_error"] = str(e)

    metrics["confusion_matrix"] = confusion_matrix(y, y_pred).tolist()
    metrics["classification_report"] = classification_report(y, y_pred, output_dict=True)

    with open(OUT_METRICS, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("Training final pipeline on entire dataset and saving model...")
    pipeline.fit(X, y)
    joblib.dump(pipeline, OUT_MODEL)

    print(f"Saved model to {OUT_MODEL}")
    print(f"Saved metrics to {OUT_METRICS}")
    print("Done.")

if __name__ == "__main__":
    main()
