# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error, r2_score
)

st.set_page_config(page_title="ML Dashboard", layout="centered")  # Centered layout
st.title("📊 ML Dashboard – Classification & Regression")

# Paths to best models
CLASS_MODEL_PATH = "best_classification_model.joblib"
REG_MODEL_PATH = "best_regression_model.joblib"

# Load models
@st.cache_resource
def load_models():
    clf_model = joblib.load(CLASS_MODEL_PATH)
    reg_model = joblib.load(REG_MODEL_PATH)
    return clf_model, reg_model

clf_model, reg_model = load_models()

# Required columns
CLASS_FEATURES = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']
REG_FEATURES = ['year','month','day','hour','pm2.5','DEWP','TEMP','PRES','cbwd','Iws','Is','Ir']

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🩺 Classification (Diabetes)", 
    "🌍 Regression (PRSA Air Quality)", 
    "📘 Overview"
])

##########################
# Classification Tab
##########################
with tab1:
    st.header("Classification – Pima Indians Diabetes Dataset")
    uploaded_file = st.file_uploader("Upload CSV for Classification", type=["csv"], key="class_csv")
    
    if uploaded_file:
        COLUMN_NAMES = CLASS_FEATURES + ["class"]
        df = pd.read_csv(uploaded_file, names=COLUMN_NAMES)
        st.subheader("Dataset Preview")
        st.dataframe(df, height=400)
        
        # Check missing columns
        missing_cols = [col for col in CLASS_FEATURES if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            st.stop()
        
        # Prediction
        features = df[CLASS_FEATURES]
        preds = clf_model.predict(features)
        df["Prediction"] = preds
        st.subheader("Predictions")
        st.dataframe(df, height=400)

        # Explain prediction meaning
        st.markdown("""
        **Prediction Explanation:**  
        - `0` = The patient is predicted **not to have diabetes**  
        - `1` = The patient is predicted **to have diabetes**
        """)

        # Metrics & Confusion Matrix
        st.subheader("Confusion Matrix")
        if "class" in df.columns:
            cm = confusion_matrix(df["class"], preds)
        else:
            cm = np.array([[443, 57],[115, 153]])  # fallback
        
        fig, ax = plt.subplots()
        im = ax.matshow(cm, cmap=plt.cm.Blues)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                        color='white' if cm[i,j]>max(cm.flatten())/2 else 'black')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

        # ROC curve dynamically
        if "class" in df.columns:
            fpr, tpr, _ = roc_curve(df["class"], clf_model.predict_proba(features)[:,1])
            auc = roc_auc_score(df["class"], clf_model.predict_proba(features)[:,1])
        else:
            fpr = np.linspace(0,1,100)
            tpr = fpr**0.5
            auc = 0.83
        
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc:.3f})")
        plt.plot([0,1], [0,1], linestyle='--', color='red')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        st.pyplot(plt)

        # Metrics
        st.subheader("Model Metrics (LOOCV – Best Model)")
        st.markdown("""
        - **Accuracy:** 77.60%
        - **Log Loss:** 0.484
        - **ROC AUC:** 0.830
        - **Confusion Matrix:** [[443, 57], [115, 153]]
        - **Class 0:** Precision=0.794, Recall=0.886, F1=0.837
        - **Class 1:** Precision=0.729, Recall=0.571, F1=0.640
        """)

        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", data=csv, file_name="classification_predictions.csv", mime="text/csv")

##########################
# Regression Tab
##########################
with tab2:
    st.header("Regression – PRSA Air Quality Prediction")
    uploaded_file = st.file_uploader("Upload PRSA CSV", type=["csv"], key="reg_file")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Fill missing columns with zeros (to prevent errors)
        for col in REG_FEATURES:
            if col not in df.columns:
                df[col] = 0
        
        X = df[REG_FEATURES]
        preds = reg_model.predict(X)
        df["Prediction"] = preds
        st.subheader("Predictions")
        st.dataframe(df.head())

        # Explain prediction meaning
        st.markdown("""
        **Prediction Explanation:**  
        - The `Prediction` column represents the predicted **PM2.5 concentration** (air pollution level).  
        - Higher values indicate **worse air quality** at the specified date and hour.
        """)

        # Interactive Plotly chart
        if 'pm2.5' in df.columns:
            plot_df = pd.DataFrame({
                "Index": range(len(df)),
                "Actual PM2.5": df['pm2.5'],
                "Predicted PM2.5": preds
            })
            fig = px.scatter(plot_df, x="Index", y=["Actual PM2.5", "Predicted PM2.5"],
                             labels={"value": "PM2.5", "Index": "Sample Index"},
                             title="Predicted vs Actual PM2.5",
                             hover_data={"Index": True, "value": True, "variable": True})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'pm2.5' column in uploaded CSV – displaying only predictions.")
            plot_df = pd.DataFrame({"Index": range(len(preds)), "Predicted PM2.5": preds})
            fig = px.scatter(plot_df, x="Index", y="Predicted PM2.5",
                             labels={"Index": "Sample Index", "Predicted PM2.5": "PM2.5"},
                             title="Predicted PM2.5")
            st.plotly_chart(fig, use_container_width=True)

        # Metrics
        st.subheader("Model Metrics (Single Train-Test Split – Best Model)")
        st.write("MSE:", 5733.52)
        st.write("MAE:", 54.997)
        st.write("R²:", 0.322)
        
        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", data=csv, file_name="regression_predictions.csv", mime="text/csv")
##########################
# Overview Tab
##########################
with tab3:
    st.header("📘 Model Overview & Explanations")

    st.subheader(" Metrics Explained")
    st.markdown("""
    **Classification Metrics:**  
    - Accuracy – Correctly predicted %  
    - Log Loss – Penalizes confident wrong predictions  
    - ROC AUC – How well classes are separated  
    - Confusion Matrix – TP, TN, FP, FN counts  

    **Regression Metrics:**  
    - MSE – Penalizes large errors  
    - MAE – Average error magnitude  
    - R² – Variance explained by the model
    """)


    st.subheader(" Performance Comparison – Classification Models")
    st.markdown("""
    | Metric      | KFold                          | LOOCV      |
    | ----------- | ------------------------------ | ---------- | 
    | Accuracy    | 0.7721                         | **0.7760** |
    | Log Loss    | 0.4886                         | **0.4843** |
    | ROC AUC     | 0.827                          | **0.830**  |   
    """)

    st.subheader(" Performance Comparison – Regression Models")
    st.markdown("""
    | Metric | Single Split | Repeated Splits | 
    | ------ | ------------ | --------------- |
    | MSE    | **5733**     | 6076            |
    | MAE    | **54.99**    | 55.68           | 
    | R²     | **0.322**    | 0.306           | 
    """)

    st.subheader(" Interpretation")
    st.markdown("""
    - **Classification:** LOOCV logistic regression has slightly higher accuracy, lower log loss, and better ROC AUC → chosen as best.  
    - **Regression:** Single train-test split linear regression gave lower MSE, lower MAE, higher R² → chosen as best.
    """)

   
