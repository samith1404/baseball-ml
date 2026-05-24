import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Baseball ML Dashboard", layout="wide")

st.title("⚾ Baseball Machine Learning Performance Dashboard")

# Load datasets
player = pd.read_csv("player_features_5yr.csv")
team = pd.read_csv("team_features.csv")
text = pd.read_csv("clean_tweets.csv")

st.sidebar.header("Choose Analysis")
option = st.sidebar.selectbox(
    "Select Model Type",
    ["Player Performance", "Team Win Prediction", "Sentiment Analysis"]
)

if option == "Player Performance":
    st.subheader("📈 Player Performance Prediction Data")
    st.dataframe(player.head(20))
    st.metric("Best Model", "XGBoost")
    st.metric("RMSE", "18.92")
    st.metric("R² Score", "0.678")

elif option == "Team Win Prediction":
    st.subheader("🏆 Team Win Prediction Data")
    st.dataframe(team.head(20))
    st.metric("Best Model", "Logistic Regression")
    st.metric("Accuracy", "93%")
    st.metric("ROC", "0.981")

else:
    st.subheader("💬 Sentiment Analysis Data")
    st.dataframe(text.head(20))
    st.metric("Best Model", "Logistic Regression")
    st.metric("F1 Score", "0.755")
    st.metric("ROC", "0.830")
