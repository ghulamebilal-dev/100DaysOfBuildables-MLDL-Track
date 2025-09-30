Wine Clustering Analysis
=======================

Files included:
- wine_clustering.ipynb   : Jupyter notebook reproducing the analysis.
- results/parsed_wine_dataset.csv : Parsed dataset used by the notebook.
- results/metrics.csv     : Clustering metrics (ARI, AMI, silhouette).
- results/kmeans_labels.csv : KMeans cluster labels.
- results/dbscan_labels.csv : DBSCAN cluster labels.
- results/pca_2d_coords.csv : PCA 2D coordinates.
- results/pca_*.png       : PCA scatterplots for true labels, KMeans, DBSCAN.

How to run:
1. Place `parsed_wine_dataset.csv` in the same folder as the notebook or script.
2. For notebook: open `wine_clustering.ipynb` in Jupyter and run cells.
3. For script: run `python wine_clustering.py`. It will write outputs to `results/`.
