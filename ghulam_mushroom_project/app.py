
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import joblib, os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

st.set_page_config(page_title="Mushroom Classification", layout="wide")
st.title("Ghulame Bilal — Mushroom Classification (Edible vs Poisonous)")

DATA_PATH = Path(__file__).resolve().parent / "data" / "mushrooms_sample.csv"

df = None
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
else:
    uploaded = st.file_uploader("Upload mushrooms CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

if df is None:
    st.error("No dataset. Place 'mushrooms_sample.csv' in data/ or upload file.")
    st.stop()

st.header("Dataset preview")
st.dataframe(df.head())

st.header("Cap color vs Edibility")
if 'cap-color' in df.columns and 'class' in df.columns:
    counts = df.groupby(['cap-color','class']).size().reset_index(name='count')
    fig = px.bar(counts, x='cap-color', y='count', color='class', barmode='group', title='Cap color vs Edibility')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Required columns ('cap-color','class') not found.")

st.header("Modeling")
if st.button("Train model"):
    X = df.drop(columns=['class'])
    y = df['class']
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    pre = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)])
    model = Pipeline(steps=[('pre', pre), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    st.write("Accuracy:", acc)
    st.write("F1 score:", f1)
    st.write("Classification report:")
    st.text(classification_report(y_test, preds))
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/mushroom_model.pkl')
    st.success("Model trained and saved to models/mushroom_model.pkl")

st.header("Predict single sample")
if os.path.exists('models/mushroom_model.pkl'):
    model = joblib.load('models/mushroom_model.pkl')
    sample = {}
    for col in df.drop(columns=['class']).columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            sample[col] = st.number_input(col, value=float(df[col].iloc[0]))
        else:
            opts = df[col].unique().tolist()
            sample[col] = st.selectbox(col, options=opts, index=0)
    if st.button("Predict"):
        input_df = pd.DataFrame([sample])
        pred = model.predict(input_df)[0]
        st.write("Prediction:", pred)
else:
    st.info("No saved model found. Train a model first.")
