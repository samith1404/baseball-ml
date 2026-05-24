import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

data = pd.read_csv("player_features_5yr.csv")

X = data[["AB","H","HR","RBI","BB","SO"]]
y = data["R"]

# use smaller sample for speed (still valid for project)
data_small = data.sample(12000, random_state=42)

X = data_small[["AB","H","HR","RBI","BB","SO"]]
y = data_small["R"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=60,
    max_depth=3,
    learning_rate=0.15,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)

print("XGBoost RMSE:", round(rmse,2))
print("XGBoost R2:", round(r2,3))
