import pandas as pd
import numpy as np
import os
import joblib
import logging
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_player_models():
    logging.info("--- Training Player Performance Model ---")
    data = pd.read_csv("src/data/player_features.csv")
    
    # Target is R (Runs)
    # Features include our new Sabermetrics
    feat_cols = ["AB", "H", "HR", "RBI", "BB", "SO", "BA", "OBP", "SLG", "OPS"]
    data = data.dropna(subset=feat_cols + ["R"])
    
    X = data[feat_cols]
    y = data["R"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost with RandomizedSearchCV for Hyperparameter tuning
    param_dist = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0]
    }
    
    xgb = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10, cv=3, 
                                scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1)
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    preds = best_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    logging.info(f"Best Params: {search.best_params_}")
    logging.info(f"Test RMSE: {rmse:.2f}, R2: {r2:.3f}")
    
    # Save the model
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(best_model, "saved_models/player_xgb_best.joblib")
    logging.info("Saved player model to saved_models/player_xgb_best.joblib")


def train_team_models():
    logging.info("--- Training Team Win Prediction Model ---")
    data = pd.read_csv("src/data/team_features.csv")
    
    feat_cols = ["R", "RA", "H", "HR", "BB", "SO", "RunDiff", "PythagWinPct"]
    X = data[feat_cols]
    y = data["Win_Season"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_dist = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200]
    }
    
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1, verbosity=0)
    search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10, cv=3, 
                                scoring='accuracy', random_state=42, n_jobs=-1)
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    preds = best_model.predict(X_test)
    probs = best_model.predict_proba(X_test)[:,1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    logging.info(f"Best Params: {search.best_params_}")
    logging.info(f"Test Accuracy: {acc:.3f}, AUC: {auc:.3f}")
    
    joblib.dump(best_model, "saved_models/team_xgb_best.joblib")
    logging.info("Saved team model to saved_models/team_xgb_best.joblib")


def train_sentiment_models():
    logging.info("--- Training Sentiment Analysis Model ---")
    tweets = pd.read_csv("clean_tweets.csv")
    
    if tweets["target"].max() > 1:
        tweets["target"] = tweets["target"].map({0: 0, 4: 1})
        
    tweets = tweets.dropna(subset=["target", "text"])
    
    # Use a bit more data since we aren't doing it live anymore
    sample = tweets.sample(min(80000, len(tweets)), random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        sample["text"], sample["target"], test_size=0.2, random_state=42
    )
    
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # We stick with Logistic Regression for fast inference on text
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_vec, y_train)
    
    preds = clf.predict(X_test_vec)
    f1 = f1_score(y_test, preds)
    
    logging.info(f"Test F1 Score: {f1:.3f}")
    
    joblib.dump(clf, "saved_models/sentiment_lr.joblib")
    joblib.dump(vectorizer, "saved_models/tfidf_vectorizer.joblib")
    logging.info("Saved sentiment models to saved_models/")


if __name__ == "__main__":
    train_player_models()
    train_team_models()
    train_sentiment_models()
    logging.info("All models trained and saved successfully.")
