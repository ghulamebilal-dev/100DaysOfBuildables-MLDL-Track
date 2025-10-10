import joblib, os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_pipeline(path):
    obj = joblib.load(path)
    return obj['pipeline'], obj['meta'], obj['model_name']

def get_feature_names(preproc, meta):
    # numeric names
    num_cols = meta['num_cols']
    cat_cols = meta['cat_cols']
    # get onehot names
    ohe = None
    for name, trans, cols in preproc.transformers_:
        if name == 'cat':
            # transformer is a Pipeline
            ohe = trans.named_steps['ohe']
    ohe_names = list(ohe.get_feature_names_out(cat_cols)) if ohe is not None else []
    return num_cols + ohe_names

def plot_importances(model, names, out_path, title='feature importances'):
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).ravel()
    else:
        raise ValueError('Model has no feature_importances_ or coef_')
    idx = np.argsort(importances)[::-1][:20]
    names_sorted = [names[i] for i in idx]
    vals = importances[idx]
    plt.figure(figsize=(8,6))
    plt.barh(range(len(vals))[::-1], vals[::-1])
    plt.yticks(range(len(vals)), names_sorted[::-1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    BASE = Path(__file__).parents[1]
    pipeline_path = BASE / 'models' / 'best_pipeline.pkl'
    pipeline, meta, model_name = load_pipeline(pipeline_path)
    preproc = pipeline.named_steps['preproc']
    clf = pipeline.named_steps['clf']
    names = get_feature_names(preproc, meta)
    out_dir = BASE / 'outputs'
    out_dir.mkdir(exist_ok=True)
    plot_importances(clf, names, out_dir / f'{model_name}_feature_importance.png', title=f'{model_name} importances')
    print('Saved feature importance to outputs/')

if __name__ == '__main__':
    main()
