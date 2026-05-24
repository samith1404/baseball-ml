import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

data = pd.read_csv("player_features_5yr.csv")

X = data[["AB","H","HR","RBI","BB","SO"]]
y = data["R"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=80, n_jobs=-1),
    "XGBoost": XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        n_jobs=-1
    )
}

print("\nPLAYER PERFORMANCE RESULTS\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    print(f"{name:15} RMSE: {rmse:.2f}   R2: {r2:.3f}")
