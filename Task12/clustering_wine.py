#!/usr/bin/env python3
"""
clustering_wine.py
Reproducible script to run KMeans and DBSCAN on parsed_wine_dataset.csv
Saves results into an output folder and creates a zip archive.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
import json, zipfile

def main(input_csv=\"parsed_wine_dataset.csv\", outdir=\"wine_with_code_results\", n_init=20):
    base = Path(outdir)
    base.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    if 'target' not in df.columns:
        raise ValueError(\"Expected 'target' column in CSV\")
    y = df['target'].values
    X = df.drop(columns=['target']).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)
    n_clusters = len(np.unique(y))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=n_init).fit(Xs)
    kl = kmeans.labels_
    # DBSCAN grid search
    best = {'score': -999, 'eps': None, 'min_samples': None, 'labels': None}
    eps_values = np.linspace(0.1,3.0,30)
    for eps in eps_values:
        for ms in [3,5,7,10]:
            db = DBSCAN(eps=eps, min_samples=ms).fit(Xs)
            labels = db.labels_
            unique_labels = set(labels)
            ncl = len([l for l in unique_labels if l!=-1])
            if ncl>=2:
                try:
                    sc = silhouette_score(Xs, labels)
                except:
                    sc = -999
                if sc > best['score']:
                    best.update({'score': sc, 'eps': eps, 'min_samples': ms, 'labels': labels.copy()})
    if best['eps'] is None:
        db = DBSCAN(eps=0.8,min_samples=5).fit(Xs)
        bl = db.labels_
        best.update({'eps':0.8,'min_samples':5,'labels':bl,'score':None})
    db_labels = best['labels']
    # metrics
    def safe(x):
        return float(x) if x is not None else None
    metrics = [
        {\"method\":f\"KMeans (k={n_clusters})\",\"ARI\":safe(adjusted_rand_score(y,kl)),
         \"AMI\":safe(adjusted_mutual_info_score(y,kl)),\"silhouette\":safe(silhouette_score(Xs,kl))},
        {\"method\":f\"DBSCAN (eps={best['eps']:.3f}, min_samples={best['min_samples']})\",
         \"ARI\":safe(adjusted_rand_score(y,db_labels)),\"AMI\":safe(adjusted_mutual_info_score(y,db_labels)),
         \"silhouette\":None}
    ]
    mask = np.array(db_labels) != -1
    if mask.sum()>=2 and len(set(db_labels)) - (1 if -1 in db_labels else 0) >=2:
        metrics[1]['silhouette'] = float(silhouette_score(Xs[mask], np.array(db_labels)[mask]))
    # save results
    pd.DataFrame(metrics).to_csv(base/\"metrics.csv\", index=False)
    pd.DataFrame({'kmeans_label':kl}).to_csv(base/\"kmeans_labels.csv\", index=False)
    pd.DataFrame({'dbscan_label':db_labels}).to_csv(base/\"dbscan_labels.csv\", index=False)
    pd.DataFrame(Xp, columns=['PC1','PC2']).to_csv(base/\"pca_2d_coords.csv\", index=False)
    df.to_csv(base/\"parsed_wine_dataset.csv\", index=False)
    with open(base/\"summary.txt\",\"w\") as f:
        json.dump({\"dbscan_best\":{\"eps\":best['eps'],\"min_samples\":best['min_samples'],\"silhouette\":best['score']},\"n_clusters\":n_clusters}, f, indent=2)
    # plotting functions
    def plot_labels(lbls, title, fname):
        plt.figure(figsize=(8,6))
        for l in np.unique(lbls):
            mask = (lbls==l)
            label = \"noise\" if l==-1 else f\"cluster {l}\"
            plt.scatter(Xp[mask,0], Xp[mask,1], label=label, s=40)
        plt.title(title)
        plt.xlabel(\"PC1\"); plt.ylabel(\"PC2\"); plt.legend(); plt.grid(True)
        plt.savefig(base/fname, bbox_inches='tight'); plt.close()
    plot_labels(y.astype(int), \"True labels (PCA 2D)\", \"pca_true_labels.png\")
    plot_labels(kl, f\"KMeans (k={n_clusters}) (PCA 2D)\", \"pca_kmeans_labels.png\")
    plot_labels(np.array(db_labels), f\"DBSCAN (eps={best['eps']:.3f}, min_samples={best['min_samples']}) (PCA 2D)\", \"pca_dbscan_labels.png\")
    # create a zip
    zpath = Path(\"/mnt/data/wine_with_code_results.zip\")
    with zipfile.ZipFile(zpath, 'w') as zf:
        for file in sorted(base.glob('*')):
            zf.write(file, arcname=file.name)
    print(\"Wrote outputs to:\", base)
    print(\"Zip archive:\", zpath)

if __name__=='__main__':
    main()
