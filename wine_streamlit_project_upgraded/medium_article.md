# Deploying a Tuned Random Forest Classifier for Wine Classification

This is a draft article suitable for Medium. It explains the steps taken: dataset selection, preprocessing, K-Fold cross-validation, hyperparameter tuning using GridSearchCV, model evaluation, and deploying the final model with Streamlit.

(Shortened for assignment — expand when publishing.)

## Introduction
Explain the goal: train a robust wine classifier and deploy it as a web app.

## Dataset
UCI Wine dataset (features: 13 numeric features; target: wine class)

## Methods
1. Exploratory data analysis
2. Baseline Random Forest with 5-fold CV
3. Hyperparameter tuning with GridSearchCV
4. Evaluation on holdout set
5. Save model and create Streamlit app

## Results
Summarize improvements: GridSearchCV improved CV accuracy and produced more stable results across folds. Test set accuracy improved vs baseline.

## How to run
1. Install requirements: `pip install -r requirements.txt`
2. Run the notebook or `python train_model.py`
3. Run Streamlit: `streamlit run app.py`

## Reflection
See reflection.md for more details.
