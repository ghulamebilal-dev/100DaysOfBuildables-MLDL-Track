# Deploying an Ensemble ML Model with Streamlit: From Tuning to Production

## Introduction
This article explains how I trained, tuned, and deployed a wine classification model using Random Forest and XGBoost, followed by a Streamlit web app for real-time predictions.

## Dataset
I used the UCI Wine dataset. It has numeric features and a multiclass target.

## Preprocessing
- Kept numeric features only
- Stratified train-test split

## Model selection and validation
- Baseline Random Forest with 5-fold Stratified CV
- GridSearchCV for Random Forest
- RandomizedSearchCV for wider/faster search
- XGBoost as a comparison model (tuned with RandomizedSearchCV)

## Results
- Show CV improvements, test set accuracy, and F1 scores.

## Deploying with Streamlit
Step-by-step guide (see deployment.md) and notes on model hosting and production readiness.

## Conclusion
Tuning reduced variance and improved generalization. XGBoost provided competitive performance in this task.

---

*(Include plots and code snippets from `wine_extended_tuning.ipynb` when publishing.)*