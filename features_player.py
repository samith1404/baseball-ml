import pandas as pd

batting = pd.read_csv("Batting.csv")

cols = ["playerID","yearID","AB","H","HR","RBI","BB","SO","R"]
batting = batting[cols].fillna(0)
batting = batting[batting["AB"] > 0]

batting = batting.sort_values(["playerID","yearID"])

rolling = (
    batting
    .groupby("playerID")[["AB","H","HR","RBI","BB","SO"]]
    .rolling(5)
    .mean()
    .reset_index()
)

batting_feat = pd.concat(
    [batting[["playerID","yearID","R"]].reset_index(drop=True),
     rolling.drop("playerID",axis=1)],
    axis=1
)

batting_feat = batting_feat.dropna()

batting_feat.to_csv("player_features_5yr.csv", index=False)

print(batting_feat.head())
print("Shape:", batting_feat.shape)
