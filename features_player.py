import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_player_features(input_path="Batting.csv", output_path="src/data/player_features.csv"):
    logging.info("Loading Batting data...")
    batting = pd.read_csv(input_path)
    
    # Keep useful columns and fill NaNs
    cols = ["playerID", "yearID", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "SO", "HBP", "SF"]
    batting = batting[cols].fillna(0)
    
    # Filter out players with very few at-bats to avoid division by zero and noisy data
    batting = batting[batting["AB"] > 50]
    
    logging.info("Calculating advanced sabermetrics...")
    # Calculate Singles (1B)
    batting["1B"] = batting["H"] - batting["2B"] - batting["3B"] - batting["HR"]
    
    # Batting Average (BA)
    batting["BA"] = batting["H"] / batting["AB"]
    
    # On-Base Percentage (OBP)
    batting["OBP"] = (batting["H"] + batting["BB"] + batting["HBP"]) / (batting["AB"] + batting["BB"] + batting["HBP"] + batting["SF"])
    
    # Slugging Percentage (SLG)
    batting["SLG"] = (batting["1B"] + 2*batting["2B"] + 3*batting["3B"] + 4*batting["HR"]) / batting["AB"]
    
    # OPS (On-Base Plus Slugging)
    batting["OPS"] = batting["OBP"] + batting["SLG"]
    
    # Handle any potential NaNs from division by zero
    batting = batting.fillna(0)
    
    # Aggregate by player and year (some players have multiple stints in a year)
    batting = batting.groupby(["playerID", "yearID"]).sum().reset_index()
    
    # Recalculate rate stats after sum aggregation
    batting["BA"] = batting["H"] / batting["AB"]
    batting["OBP"] = (batting["H"] + batting["BB"] + batting["HBP"]) / (batting["AB"] + batting["BB"] + batting["HBP"] + batting["SF"])
    batting["SLG"] = (batting["1B"] + 2*batting["2B"] + 3*batting["3B"] + 4*batting["HR"]) / batting["AB"]
    batting["OPS"] = batting["OBP"] + batting["SLG"]
    batting = batting.fillna(0)

    # Calculate 3-year rolling averages to predict next year's Runs (R)
    logging.info("Calculating 3-year rolling averages...")
    batting = batting.sort_values(["playerID", "yearID"])
    
    feature_cols = ["AB", "H", "HR", "RBI", "BB", "SO", "BA", "OBP", "SLG", "OPS"]
    rolling = (
        batting.groupby("playerID")[feature_cols]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    # Shift rolling features by 1 to predict current year's R based on previous years
    rolling_shifted = rolling.groupby(batting["playerID"]).shift(1)
    
    batting_feat = pd.concat([batting[["playerID", "yearID", "R"]], rolling_shifted], axis=1)
    
    # Drop NaNs (first year for each player will be NaN because of shift)
    batting_feat = batting_feat.dropna()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    batting_feat.to_csv(output_path, index=False)
    logging.info(f"Saved advanced player features to {output_path} (Shape: {batting_feat.shape})")

if __name__ == "__main__":
    create_player_features()
