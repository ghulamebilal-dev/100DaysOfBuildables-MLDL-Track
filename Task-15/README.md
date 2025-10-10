# Ensemble Project (DecisionTree, RandomForest, XGBoost) — Minimal MVP

This is a minimal end-to-end example that:
- trains a Decision Tree, Random Forest, and XGBoost classifier on a sample dataset,
- compares metrics (accuracy, precision, recall, f1, roc_auc),
- plots feature importances for Random Forest and XGBoost,
- saves the best pipeline (preprocessing + model) to `models/best_pipeline.pkl`,
- provides a Streamlit app (`app/app.py`) for real-time predictions.

## What's included
- `data/sample_dataset.csv` — synthetic dataset used for training.
- `src/` — code for preprocessing and training.
- `models/best_pipeline.pkl` — trained pipeline selected by F1 score.
- `outputs/` — metrics CSV and feature importance images.
- `app/app.py` — Streamlit app to load the saved pipeline and predict.

## How to run locally (Linux / Windows WSL / macOS)
1. Create virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Train (optional, model already provided):
   ```
   python src/models/train_models.py
   ```
   This will create `models/best_pipeline.pkl` and `outputs/metrics_summary.csv`.

4. Run Streamlit app:
   ```
   streamlit run app/app.py
   ```
   The app will open in your browser (default port 8501).

## Using your own dataset
Replace `data/sample_dataset.csv` with your CSV, ensuring there is a `target` column.
Edit `src/models/train_models.py` to change `TARGET` variable if needed.