import streamlit as st
import pandas as pd
import joblib  # ✅ joblib, not pickle!
import numpy as np

# Load trained models
logistic_model = joblib.load('models/logistic_model.pkl')
decision_tree_model = joblib.load('models/decision_tree_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')

# Mapping from label number to class name
label_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Sidebar model selection
st.sidebar.title("Choose Model")
model_choice = st.sidebar.selectbox("Select a classifier:", ("Logistic Regression", "Decision Tree", "KNN"))

# Input sliders
st.title("Iris Flower Classifier 🌸")
st.subheader("Input Flower Features Below")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Feature array
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict based on model choice
if model_choice == "Logistic Regression":
    model = logistic_model
elif model_choice == "Decision Tree":
    model = decision_tree_model
else:
    model = knn_model

prediction = model.predict(input_data)
predicted_class = label_map[int(prediction[0])]

# Show prediction
st.markdown("### 🧠 Model Prediction")
st.write(f"**Predicted Class:** {predicted_class}")

# Show accuracy
st.markdown("### 📊 Model Accuracy")
if model_choice == "Logistic Regression":
    st.write("Accuracy: 0.97")
elif model_choice == "Decision Tree":
    st.write("Accuracy: 1.00")
else:
    st.write("Accuracy: 0.97")
