import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

teams = pd.read_csv("Teams.csv").fillna(0)

cols = ["R","RA","H","HR","BB","SO","W","L"]
teams = teams[cols]

teams["Win"] = (teams["W"] > teams["L"]).astype(int)

X = teams[["R","RA","H","HR","BB","SO"]]
y = teams["Win"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=120),
    "XGBoost": XGBClassifier(n_estimators=120, max_depth=4)
}

print("\nTEAM WIN PREDICTION RESULTS\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)

    print(f"{name:15} Acc: {acc:.3f}  F1: {f1:.3f}  ROC: {roc:.3f}")
