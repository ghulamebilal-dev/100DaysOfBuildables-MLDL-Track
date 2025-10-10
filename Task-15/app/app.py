import streamlit as st
import joblib, os
import pandas as pd

@st.cache_resource
def load_artifact():
    base = os.path.join(os.path.dirname(__file__), '..')
    path = os.path.join(base, 'models', 'best_pipeline.pkl')
    obj = joblib.load(path)
    return obj

artifact = load_artifact()
pipeline = artifact['pipeline']
meta = artifact['meta']
model_name = artifact['model_name']

st.title("Ensemble Demo — Real-time Prediction")
st.write("Model loaded:", model_name)

# Build input form
st.sidebar.header("Input features")
inputs = {}
for col in meta['num_cols']:
    inputs[col] = st.sidebar.number_input(col, value=0.0)
for col in meta['cat_cols']:
    # categories are not saved; use example choices
    choices = ['Lahore','Karachi','Islamabad']
    inputs[col] = st.sidebar.selectbox(col, choices)

if st.sidebar.button("Predict"):
    df = pd.DataFrame([inputs])
    st.subheader("Input preview")
    st.write(df)
    pred = pipeline.predict(df)[0]
    st.write("Prediction:", int(pred))
    if hasattr(pipeline.named_steps['clf'], 'predict_proba'):
        proba = pipeline.predict_proba(df)[0]
        st.write("Probabilities:", proba)
