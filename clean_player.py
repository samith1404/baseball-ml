import pandas as pd

batting = pd.read_csv("Batting.csv")

# Keep useful columns only
cols = ["playerID","yearID","AB","H","HR","RBI","BB","SO","R"]
batting = batting[cols]

# Remove rows with no at-bats
batting = batting[batting["AB"] > 0]

# Fill missing values
batting = batting.fillna(0)

print(batting.head())
print(batting.shape)
