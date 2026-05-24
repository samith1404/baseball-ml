import pandas as pd

print(pd.read_csv("Batting.csv").shape)
print(pd.read_csv("Teams.csv").shape)
print(pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None).shape)
