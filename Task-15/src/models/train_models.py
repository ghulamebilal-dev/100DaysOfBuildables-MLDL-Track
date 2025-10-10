import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib, os
from src.data.preprocessing import build_preprocessor

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sample_dataset.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
os.makedirs(OUTPUT_OUT, exist_ok=True)

def evaluate(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe.named_steps['clf'], 'predict_proba') else None
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, proba)) if proba is not None else None
    }

def main():
    df = pd.read_csv(DATA_PATH)
    TARGET = 'target'
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preproc, meta = build_preprocessor(X_train)

    models = {}
    # Decision Tree
    dt = Pipeline([('preproc', preproc), ('clf', DecisionTreeClassifier(random_state=42))])
    dt.fit(X_train, y_train)
    models['DecisionTree'] = dt

    # Random Forest
    rf = Pipeline([('preproc', preproc), ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf

    # XGBoost
    xgb = Pipeline([('preproc', preproc), ('clf', XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=4))])
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb

    results = {}
    for name, pipe in models.items():
        results[name] = evaluate(pipe, X_test, y_test)
        print(name, results[name])

    # choose best by f1
    best_name = max(results, key=lambda n: results[n]['f1'])
    best_pipe = models[best_name]
    print('Best model:', best_name)

    # save pipeline and metadata
    save_obj = {'pipeline': best_pipe, 'meta': meta, 'model_name': best_name}
    joblib.dump(save_obj, os.path.join(OUTPUT_DIR, 'best_pipeline.pkl'))
    # save metrics summary
    import json
    with open(os.path.join(OUTPUT_OUT, 'metrics_summary.json'), 'w') as f:
        json.dump({'results': results, 'best': best_name}, f, indent=2)
    print('Saved pipeline and metrics.')

if __name__ == '__main__':
    main()
