# Deployment Guide

This file explains how to deploy the Streamlit app (`app.py`) using three popular hosts: Streamlit Cloud, Render, and Heroku.

## Prepare repository
- Include `app.py`, `model.joblib`, `wine_dataset.csv`, `requirements.txt`, and optionally `Procfile` for Heroku.
- Commit and push to GitHub.

## Streamlit Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**, select the repo and branch, and set `app.py` as the entrypoint.
3. Click **Deploy**. Streamlit Cloud will install dependencies from `requirements.txt`.

## Render (Web Service)
1. Create a new Web Service on Render and connect your GitHub repo.
2. Set the build command to `pip install -r requirements.txt` and the start command to `streamlit run app.py --server.port $PORT`.
3. Add `PORT` as an environment variable if needed.

## Heroku
1. Create `Procfile` with: `web: streamlit run app.py --server.port $PORT`
2. Create app on Heroku, push code via Git, and set buildpacks to Python.
3. Deploy; Heroku will install packages from `requirements.txt`.

Notes:
- For model files larger than repo limits, use Git LFS or host models on cloud storage and download at runtime.
- Use environment variables for secrets and config.
