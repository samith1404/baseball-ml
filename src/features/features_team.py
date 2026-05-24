import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_team_features(input_path="Teams.csv", output_path="src/data/team_features.csv"):
    logging.info("Loading Teams data...")
    teams = pd.read_csv(input_path)
    
    # Select important columns
    cols = ["yearID", "teamID", "W", "L", "R", "RA", "H", "HR", "BB", "SO", "HA", "HRA", "BBA", "SOA"]
    features = teams[cols].dropna()
    
    # Calculate Run Differential
    features["RunDiff"] = features["R"] - features["RA"]
    
    # Pythagorean Win Expectancy: R^1.83 / (R^1.83 + RA^1.83)
    # Using 1.83 is the standard modern baseball exponent
    features["PythagWinPct"] = (features["R"]**1.83) / (features["R"]**1.83 + features["RA"]**1.83)
    
    # Calculate actual Win Pct
    features["WinPct"] = features["W"] / (features["W"] + features["L"])
    
    # Binary Target: Did they win more than they lost?
    features["Win_Season"] = (features["W"] > features["L"]).astype(int)
    
    features = features.fillna(0)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_csv(output_path, index=False)
    logging.info(f"Saved advanced team features to {output_path} (Shape: {features.shape})")

if __name__ == "__main__":
    create_team_features()
