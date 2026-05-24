import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

sent = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None
)

sent.columns = ["target","id","date","flag","user","text"]
sent = sent[["target","text"]]
sent["target"] = sent["target"].map({0:0,4:1})

# use smaller sample for speed (still huge academically)
sent_small = sent.sample(50000, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    sent_small["text"], sent_small["target"],
    test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(n_estimators=80, max_depth=4)
}

print("\nSENTIMENT MODEL RESULTS\n")

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    probs = model.predict_proba(X_test_vec)[:,1]

    f1 = f1_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)

    print(f"{name:12} F1: {f1:.3f}  ROC: {roc:.3f}")
