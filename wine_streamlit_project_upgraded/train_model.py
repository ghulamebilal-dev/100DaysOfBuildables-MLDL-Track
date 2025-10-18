# train_model.py - alternative script to train and save the model (mirrors notebook)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("wine_dataset.csv")
# detect target
for t in ['target','class','Type']:
    if t in df.columns:
        target = t
        break
else:
    target = df.columns[-1]

X = df.drop(columns=[target]).select_dtypes(include=[np.number])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
joblib.dump(best, "model.joblib")
print("Trained model saved to model.joblib")
