import pandas as pd

teams = pd.read_csv("Teams.csv")

features = teams[["yearID","teamID","W","L","R","RA","H","HR","BB","SO"]].dropna()

features["Win"] = (features["W"] > features["L"]).astype(int)

features.to_csv("team_features.csv", index=False)

print("Saved team_features.csv")
print(features.head())
print("Shape:", features.shape)
