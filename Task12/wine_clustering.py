#!/usr/bin/env python3
# wine_clustering.py - reproduce clustering analysis for provided parsed_wine_dataset.csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path('results')
out_dir.mkdir(exist_ok=True)

df = pd.read_csv('parsed_wine_dataset.csv')
y = df['target'].values
X = df.drop(columns=['target']).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
pd.DataFrame(X_pca, columns=['PC1','PC2']).to_csv(out_dir / 'pca_2d_coords.csv', index=False)

n_clusters = len(np.unique(y))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
klabels = kmeans.fit_predict(X_scaled)
pd.DataFrame({'kmeans_label': klabels}).to_csv(out_dir / 'kmeans_labels.csv', index=False)

best = {'score': -10}
eps_values = np.linspace(0.1, 3.0, 30)
min_samples_values = [3,5,7,10]
for eps in eps_values:
    for ms in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=ms)
        labels = db.fit_predict(X_scaled)
        unique = set(labels)
        n_clusters_db = len([lab for lab in unique if lab != -1])
        if n_clusters_db >= 2:
            try:
                sc = silhouette_score(X_scaled, labels)
            except Exception:
                sc = -10
            if sc > best.get('score', -10):
                best.update({'score': sc, 'eps': eps, 'min_samples': ms, 'labels': labels.copy()})
if 'labels' not in best:
    db = DBSCAN(eps=0.8, min_samples=5)
    best['labels'] = db.fit_predict(X_scaled)

db_labels = best['labels']
pd.DataFrame({'dbscan_label': db_labels}).to_csv(out_dir / 'dbscan_labels.csv', index=False)

metrics = []
metrics.append({
    'method': f'KMeans (k={n_clusters})',
    'ARI': float(adjusted_rand_score(y, klabels)),
    'AMI': float(adjusted_mutual_info_score(y, klabels)),
    'silhouette': float(silhouette_score(X_scaled, klabels))
})

mask_valid = (np.array(db_labels) != -1)
if len(set(db_labels)) - (1 if -1 in db_labels else 0) >= 2 and mask_valid.sum() >= 2:
    try:
        db_sil = float(silhouette_score(X_scaled[mask_valid], np.array(db_labels)[mask_valid]))
    except Exception:
        db_sil = None
else:
    db_sil = None
metrics.append({
    'method': f"DBSCAN (eps={best.get('eps',0):.3f}, min_samples={best.get('min_samples',0)})",
    'ARI': float(adjusted_rand_score(y, db_labels)),
    'AMI': float(adjusted_mutual_info_score(y, db_labels)),
    'silhouette': db_sil
})
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(out_dir / 'metrics.csv', index=False)
pd.DataFrame(df).to_csv(out_dir / 'parsed_wine_dataset.csv', index=False)

def plot_and_save(labels, title, fname):
    plt.figure(figsize=(8,6))
    for lab in np.unique(labels):
        mask = (labels == lab)
        name = 'noise' if lab == -1 else f'cluster {lab}'
        plt.scatter(X_pca[mask,0], X_pca[mask,1], label=name, s=40)
    plt.title(title); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend(); plt.grid(True)
    plt.savefig(out_dir / fname, bbox_inches='tight'); plt.close()

plot_and_save(y.astype(int), 'True labels (PCA 2D)', 'pca_true_labels.png')
plot_and_save(klabels, f'KMeans (k={n_clusters}) (PCA 2D)', 'pca_kmeans_labels.png')
plot_and_save(np.array(db_labels), f"DBSCAN (eps={best.get('eps',0):.3f}, min_samples={best.get('min_samples',0)}) (PCA 2D)", 'pca_dbscan_labels.png')

print('Done. Results in results/ folder')
