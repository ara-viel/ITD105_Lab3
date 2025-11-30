# train_classification.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict, LeaveOneOut
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report, roc_auc_score
import joblib

# 1) Load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
colnames = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
df = pd.read_csv(url, header=None, names=colnames)

# quick cleaning: replace zeros in some columns considered missing (optional)
for c in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
    df[c] = df[c].replace(0, np.nan)
# simple imputation (median)
df.fillna(df.median(), inplace=True)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# two model setups: both use LogisticRegression
clf = LogisticRegression(max_iter=1000, solver='liblinear')

# Model A: K-Fold CV (k=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_kf = cross_val_predict(clf, X, y, cv=kf, method='predict')
y_proba_kf = cross_val_predict(clf, X, y, cv=kf, method='predict_proba')[:,1]

# Model B: Leave-One-Out CV
loo = LeaveOneOut()
y_pred_loo = cross_val_predict(clf, X, y, cv=loo, method='predict')
y_proba_loo = cross_val_predict(clf, X, y, cv=loo, method='predict_proba')[:,1]

# metrics function
def classification_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, np.vstack([1-y_proba, y_proba]).T)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, digits=4)
    auc = roc_auc_score(y_true, y_proba)
    return {'accuracy':acc, 'log_loss':ll, 'confusion_matrix':cm, 'classification_report':cr, 'roc_auc':auc}

metrics_kf = classification_metrics(y, y_pred_kf, y_proba_kf)
metrics_loo = classification_metrics(y, y_pred_loo, y_proba_loo)

print("=== Model A (K-Fold) metrics ===")
print(metrics_kf['accuracy'], metrics_kf['log_loss'], metrics_kf['roc_auc'])
print(metrics_kf['confusion_matrix'])
print(metrics_kf['classification_report'])

print("=== Model B (LOOCV) metrics ===")
print(metrics_loo['accuracy'], metrics_loo['log_loss'], metrics_loo['roc_auc'])
print(metrics_loo['confusion_matrix'])
print(metrics_loo['classification_report'])

# Choose Model A for saving (example decision; see reasoning in write-up)
clf.fit(X, y)
joblib.dump(clf, "logreg_kfold_pima.joblib")
print("Saved model to logreg_kfold_pima.joblib")
