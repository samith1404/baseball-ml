import pandas as pd

df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)

df = df[[0,5]]
df.columns = ["target", "text"]

df = df.sample(100000, random_state=42)

df.to_csv("clean_tweets.csv", index=False)

print("CREATED clean_tweets.csv")
print(df.head())
print(df.shape)
