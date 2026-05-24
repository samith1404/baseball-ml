# ⚾ Advanced Baseball Analytics & Machine Learning Pipeline

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-00B2A9.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-orange.svg)](https://shap.readthedocs.io/)

A comprehensive, production-ready Machine Learning pipeline that analyzes historical baseball statistics and fan sentiment. Built as a Master's-level data analytics portfolio project, this repository emphasizes advanced feature engineering (Sabermetrics), hyperparameter optimization, and model interpretability (Explainable AI).

## 🚀 Features & Architecture

*   **Advanced Feature Engineering (Sabermetrics)**
    *   Calculates highly predictive baseball metrics from raw box-score data: **Batting Average (BA), On-Base Percentage (OBP), Slugging (SLG), OPS, Run Differential, and Pythagorean Win Expectancy.**
    *   Utilizes 3-year rolling averages to build temporal features for predicting future performance.
*   **Hyperparameter Optimization & ML**
    *   Leverages `RandomizedSearchCV` to optimize `XGBoost` Regressors (for player runs) and Classifiers (for team win prediction).
    *   Implements `LogisticRegression` with high-dimensional `TF-IDF` vectors (10k n-grams) for extremely fast, real-time fan tweet sentiment analysis.
*   **Explainable AI (XAI)**
    *   Integrates **SHAP** (SHapley Additive exPlanations) to provide local interpretability (waterfall plots) and global feature importance. 
*   **Interactive Streamlit Dashboard**
    *   Pre-trained models are persisted via `joblib` for rapid dashboard loading.
    *   Interactive "what-if" scenario planners for teams and players.
    *   Exploratory Data Analysis (EDA) visualizations using `Plotly`.

---

## 📂 Project Structure

```text
baseball-ml-project-main/
│
├── src/                        # Source Code
│   ├── data/                   # Engineered datasets (Generated)
│   ├── features/               # Feature Engineering Scripts
│   │   ├── features_player.py  # Sabermetrics (OPS, BA, SLG)
│   │   └── features_team.py    # Pythagorean Win Pct, Run Diff
│   └── models/
│       └── train.py            # Unified training & hyperparameter tuning
│
├── saved_models/               # Pre-trained models (.joblib)
│
├── app.py                      # Streamlit Dashboard Entrypoint
├── requirements.txt            # Python dependencies
└── ...                         # Raw Data (Batting.csv, Teams.csv, clean_tweets.csv)
```

## 📊 Key Findings & Results

1.  **Player Performance:** 
    *   The `XGBoost` regressor achieved an RMSE of ~23 runs. 
    *   **SHAP Analysis** confirmed that `OPS` (On-Base Plus Slugging) and historical `RBI` are the most significant predictors of a player's future run production, validating traditional baseball scouting heuristics with machine learning.
2.  **Team Win Prediction:**
    *   The model predicts whether a team will have a winning season (>0.500 pct) with **>90% Accuracy** and **~0.97 AUC**.
    *   The engineered *Pythagorean Win Expectancy* feature was the dominant factor, completely overshadowing raw hits or walks.
3.  **Fan Sentiment:**
    *   The `TF-IDF` + `Logistic Regression` pipeline achieved a **0.76 F1 Score** on Twitter data. Logistic Regression was specifically chosen over tree-based models here due to its O(N) inference speed on sparse text matrices, making the real-time Streamlit predictor instantaneous.

## 🛠 Setup & Installation

**1. Clone the repository and install dependencies:**
```bash
pip install -r requirements.txt
```

*(Note for Mac users: XGBoost requires OpenMP. `xgboost==2.1.4` is specified in `requirements.txt` to bypass standard macOS `libomp` linking issues).*

**2. Generate Features & Train Models:**
Run the pipelines to calculate Sabermetrics and tune the models (takes ~2-3 minutes).
```bash
python src/features/features_player.py
python src/features/features_team.py
python src/models/train.py
```

**3. Launch the Dashboard:**
```bash
streamlit run app.py
```

## 👨‍💻 Author
Built for a Data Analytics Master's Portfolio. Demonstrates end-to-end data science lifecycle from raw data ingestion to an interactive, interpretable AI deployment.
