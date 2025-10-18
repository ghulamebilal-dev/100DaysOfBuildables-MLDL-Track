import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Wine Classifier (RF vs XGB)", layout="centered")
st.title("Wine Classifier — Random Forest vs XGBoost")
st.write("Choose model, input features, and get predictions.")

# Load models (if available)
models = {}
try:
    models['RandomForest'] = joblib.load("best_rf.joblib")
except Exception:
    try:
        models['RandomForest'] = joblib.load("model.joblib")
    except Exception:
        models['RandomForest'] = None
try:
    models['XGBoost'] = joblib.load("best_xgb.joblib")
except Exception:
    models['XGBoost'] = None

# Load dataset to get feature ranges
df = pd.read_csv("wine_dataset.csv")
for t in ['target','class','Type']:
    if t in df.columns:
        target = t
        break
else:
    target = df.columns[-1]
features = df.drop(columns=[target]).select_dtypes(include=[np.number]).columns.tolist()

model_choice = st.sidebar.selectbox("Choose model", options=list(models.keys()))
st.sidebar.header("Input features")
input_data = {}
for feat in features:
    col = df[feat]
    min_v, max_v = float(col.min()), float(col.max())
    mean_v = float(col.mean())
    input_data[feat] = st.sidebar.slider(feat, min_value=min_v, max_value=max_v, value=mean_v)

X_input = pd.DataFrame([input_data])
st.subheader("Input preview")
st.write(X_input)

if st.button("Predict"):
    model = models.get(model_choice)
    if model is None:
        st.error(f"No model file found for {model_choice}. Please train models first (see train_model.py).")
    else:
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        st.success(f"Predicted class: {pred}")
        st.write("Prediction probabilities:")
        for cls, p in zip(model.classes_, proba):
            st.write(f"Class {cls}: {p:.4f}")

# Show simple comparison (if both models exist)
if models.get('RandomForest') is not None and models.get('XGBoost') is not None:
    st.sidebar.markdown('---')
    st.sidebar.write('Both models loaded. See comparison in `wine_extended_tuning.ipynb`.')