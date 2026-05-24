import pandas as pd

teams = pd.read_csv("Teams.csv")

cols = ["yearID","teamID","W","L","R","RA","H","HR","BB","SO"]
teams = teams[cols]

teams = teams.fillna(0)

# Create win/loss label
teams["Win"] = (teams["W"] > teams["L"]).astype(int)

print(teams.head())
print(teams.shape)
    