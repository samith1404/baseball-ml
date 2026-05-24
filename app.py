import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚾ Baseball ML Dashboard | Portfolio Edition",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
def get_base64_of_bin_file(bin_file):
    import base64
    try:
        # Try to open the file from the provided path
        if os.path.exists(bin_file):
            with open(bin_file, 'rb') as f:
                data = f.read()
            return base64.b64encode(data).decode()
        else:
            return ""
    except Exception as e:
        return ""

# Get the base directory for finding assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Try to load background images with fallback paths
bg_paths = [
    os.path.join(BASE_DIR, "assets", "background.jpg"),
]
image_folder_path = [p for p in bg_paths if os.path.exists(p)]
bg_base64 = get_base64_of_bin_file(image_folder_path[0]) if image_folder_path else ""

sidebar_paths = [
    os.path.join(BASE_DIR, "assets", "sidebar_background.png"),
]
sidebar_folder_path = [p for p in sidebar_paths if os.path.exists(p)]
sidebar_bg_base64 = get_base64_of_bin_file(sidebar_folder_path[0]) if sidebar_folder_path else ""

st.markdown(f"""
<style>
/* Custom Baseball Background Injection */
[data-testid="stAppViewContainer"] {{
    background-image: linear-gradient(rgba(15, 23, 42, 0.55), rgba(15, 23, 42, 0.75)), url("data:image/jpeg;base64,{bg_base64}") !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
}}
/* Force inner containers to be transparent so the background shows */
[data-testid="stHeader"], [data-testid="stAppViewBlockContainer"] {{
    background-color: transparent !important;
}}

/* Custom Sidebar Retro Baseball Background */
[data-testid="stSidebar"] {{
    background-image: linear-gradient(rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0.4)), url("data:image/png;base64,{sidebar_bg_base64}") !important;
    background-size: cover !important;
    background-position: center !important;
    background-repeat: no-repeat !important;
}}
/* Force sidebar text, headers, widgets, labels, and cache tags to be dark slate/charcoal for premium look */
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{
    color: #0f172a !important;
    font-weight: 600 !important;
}}
[data-testid="stSidebar"] [data-testid="stHeader"] {{
    background-color: transparent !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data(fname):
    df = pd.read_csv(os.path.join(BASE, fname))
    # Calculate baseball statistics if this is player data
    if "AB" in df.columns and "H" in df.columns and "HR" in df.columns and "BB" in df.columns:
        df["BA"] = df["H"] / df["AB"].replace(0, np.nan)
        df["OBP"] = (df["H"] + df["BB"]) / (df["AB"] + df["BB"]).replace(0, np.nan)
        df["SLG"] = df["HR"] / df["AB"].replace(0, np.nan)
        df["OPS"] = df["OBP"] + df["SLG"]
    return df

@st.cache_resource
def load_model(fname):
    # Try both paths for compatibility with local dev and Streamlit Cloud
    possible_paths = [
        os.path.join(BASE, "saved_models", fname),  # Local development path
        os.path.join(BASE, fname)  # Streamlit Cloud root path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return joblib.load(path)
    
    # If model not found in either location, return None
    return None

def plotly_dark():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(color="#cbd5e1", family="Inter"),
        xaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
        yaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    )

COLORS = px.colors.qualitative.Bold

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚾ Baseball ML")
    st.markdown("---")
    page = st.radio(
        "Select Analysis",
        [
            "🏠 Overview", 
            "📊 Exploratory Data Analysis",
            "📈 Player Performance (XAI)", 
            "🏆 Team Win Prediction",
            "📉 Player Aging Curves",
            "🔍 Player Comparison Tool",
            "💬 Sentiment Analysis"
        ],
        index=0,
    )
    st.markdown("---")
    
    # Dataset Information
    st.markdown("### 📊 Dataset Overview")
    try:
        player_data = load_data("player_features_5yr.csv")
        team_data = load_data("team_features.csv")
        tweet_data = load_data("clean_tweets.csv")
        
        col1, col2 = st.columns(2)
        col1.metric("🏟️ Players", f"{player_data['playerID'].nunique():,}")
        col2.metric("🏆 Teams", f"{team_data['teamID'].nunique():,}")
        
        col1.metric("📅 Years Covered", f"{int(player_data['yearID'].min())}-{int(player_data['yearID'].max())}")
        col2.metric("💬 Tweets", f"{len(tweet_data):,}")
        
        st.markdown("---")
    except:
        pass
    
    # Model Information
    st.markdown("### 🤖 Model Pipeline")
    st.markdown("""
    **Player Prediction**
    - 🎯 XGBoost Regressor
    - 📊 10 Sabermetric Features
    
    **Team Prediction**
    - 🎯 XGBoost Classifier
    - 📊 8 Advanced Metrics
    
    **Sentiment Analysis**
    - 🎯 Logistic Regression
    - 📊 TF-IDF (10K features)
    """)
    
    st.markdown("---")
    
    # Quick Tips
    st.markdown("### 💡 Quick Tips")
    with st.expander("🔍 How to use this dashboard"):
        st.markdown("""
        1. **Overview** - Start here for project summary
        2. **EDA** - Explore data relationships
        3. **Player Perf** - Predict runs & explain decisions
        4. **Team Wins** - Forecast winning seasons
        5. **Aging** - Understand player decline
        6. **Comps** - Find similar players
        7. **Sentiment** - Analyze fan tweets
        """)
    
    st.markdown("---")
    st.markdown("### 📚 Tech Stack")
    st.markdown("""
    - **ML**: XGBoost, Scikit-learn, SHAP
    - **Data**: Pandas, NumPy
    - **Viz**: Plotly, Matplotlib
    - **App**: Streamlit
    """)
    
    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("⚾ Advanced Baseball ML Analytics")
    st.markdown("A comprehensive, production-ready Machine Learning pipeline analyzing Baseball statistics and Fan Sentiment. Designed with focus on **Feature Engineering (Sabermetrics)**, **Hyperparameter Optimization**, and **Model Interpretability (Explainable AI)**.")

    try:
        player = load_data("player_features_5yr.csv")
        team   = load_data("team_features.csv")
        tweets = load_data("clean_tweets.csv")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Player Records", f"{len(player):,}")
        c2.metric("Team Records",   f"{len(team):,}")
        c3.metric("Tweet Records",  f"{len(tweets):,}")
        c4.metric("Engineered Features", "12+")
    except Exception as e:
        st.error(f"Error loading data: {e}")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-header">📈 Player Performance</div>', unsafe_allow_html=True)
        st.markdown("""
        Predicts **runs scored (R)** based on historical performance.
        - **Engineered**: BA, OBP, SLG, OPS (3-year rolling averages).
        - **Model**: Tuned XGBoost Regressor
        - **XAI**: SHAP waterfall plots for prediction interpretation.
        """)

    with col2:
        st.markdown('<div class="section-header">🏆 Team Win Prediction</div>', unsafe_allow_html=True)
        st.markdown("""
        Classifies whether a team will **win more than they lose**.
        - **Engineered**: Run Differential, Pythagorean Win %.
        - **Model**: Tuned XGBoost Classifier
        - **Performance**: >90% Accuracy, >0.95 AUC.
        """)

    with col3:
        st.markdown('<div class="section-header">💬 NLP Sentiment Analysis</div>', unsafe_allow_html=True)
        st.markdown("""
        Classifies fan **tweet sentiment** (positive / negative).
        - **Features**: TF-IDF (10,000 terms, unigrams & bigrams).
        - **Model**: Logistic Regression (for fast inference).
        - **Interactivity**: Live text prediction engine.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.caption("Visualizing the relationship between advanced Sabermetrics and Baseball outcomes.")
    
    tab1, tab2 = st.tabs(["Player Sabermetrics", "Team Sabermetrics"])
    
    with tab1:
        player = load_data("player_features_5yr.csv").dropna()
        st.subheader("Correlation: OPS vs Runs Scored (R)")
        st.markdown("OPS (On-Base Plus Slugging) is universally considered one of the most robust predictors of a player's ability to score runs.")
        
        # Sample for plotting speed
        sample = player.sample(min(2000, len(player)))
        
        fig = px.scatter(
            sample, x="OPS", y="R", color="BA", 
            color_continuous_scale="Viridis",
            hover_data=["AB", "HR"],
            title="Runs Scored vs OPS (Color = Batting Average)"
        )
        fig.update_layout(**plotly_dark(), height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        team = load_data("team_features.csv").dropna()
        st.subheader("Pythagorean Win Expectancy")
        st.markdown("Bill James' Pythagorean formula estimates a team's win percentage based on runs scored and allowed. Notice how perfectly it aligns with actual wins.")
        
        fig2 = px.scatter(
            team, x="RunDiff", y="WinPct", color="Win_Season",
            color_continuous_scale=["#f87171", "#34d399"],
            labels={"Win_Season": "Winning Season"},
            title="Actual Win Percentage vs Run Differential"
        )
        # Add the theoretical pythagorean curve line
        team_sorted = team.sort_values("RunDiff")
        fig2.add_trace(go.Scatter(
            x=team_sorted["RunDiff"], y=team_sorted["PythagWinPct"],
            mode="lines", name="Pythagorean Expectancy", line=dict(color="white", width=2, dash="dash")
        ))
        fig2.update_layout(**plotly_dark(), height=500)
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – PLAYER PERFORMANCE & SHAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Player Performance (XAI)":
    st.title("📈 Player Performance Prediction")
    st.caption("Using XGBoost and Explainable AI (SHAP) to predict Runs (R) based on advanced Sabermetrics.")
    
    model = load_model("player_xgb_best.joblib")
    data = load_data("player_features_5yr.csv")
    
    if model is None:
        st.warning("Model not found. Please train models first using `python src/models/train.py`.")
    else:
        feat_cols = ["AB", "H", "HR", "RBI", "BB", "SO", "BA", "OBP", "SLG", "OPS"]
        data = data.dropna(subset=feat_cols + ["R"])
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="section-header">Interactive Prediction</div>', unsafe_allow_html=True)
            st.markdown("Adjust the player's 3-year rolling average stats to see the predicted Runs Scored.")
            
            user_input = {}
            for col in feat_cols:
                if col in data.columns:
                    median_val = float(data[col].median())
                    max_val = float(data[col].quantile(0.99))
                    user_input[col] = st.slider(col, 0.0, max_val, median_val)
                else:
                    user_input[col] = 0.0
                
            input_df = pd.DataFrame([user_input])
            pred_r = model.predict(input_df)[0]
            
            st.metric("Predicted Runs (R) Next Season", f"{pred_r:.1f}")
            
        with col2:
            st.markdown('<div class="section-header">Explainable AI (SHAP)</div>', unsafe_allow_html=True)
            st.markdown("The Waterfall plot below shows exactly **how each stat contributed** to this specific prediction, pushing it away from the baseline (average).")
            
            # SHAP explainer
            explainer = shap.Explainer(model)
            shap_values = explainer(input_df)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            # Dark theme for SHAP
            plt.style.use('dark_background')
            shap.plots.waterfall(shap_values[0], show=False)
            fig.patch.set_facecolor('#0d1117')
            ax.set_facecolor('#0d1117')
            st.pyplot(fig)
            
            st.markdown("---")
            st.subheader("Global Feature Importance")
            # Sample for fast SHAP
            sample_X = data[feat_cols].sample(1000, random_state=42)
            shap_values_global = explainer(sample_X)
            
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            shap.plots.bar(shap_values_global, show=False)
            fig2.patch.set_facecolor('#0d1117')
            ax2.set_facecolor('#0d1117')
            st.pyplot(fig2)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – TEAM WIN PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Team Win Prediction":
    st.title("🏆 Team Win Prediction")
    st.caption("Binary classification: Identifying winning seasons (>0.500 win pct) using team aggregates.")
    
    model = load_model("team_xgb_best.joblib")
    data = load_data("team_features.csv")
    
    if model is None:
        st.warning("Model not found. Please train models first.")
    else:
        feat_cols = ["R", "RA", "H", "HR", "BB", "SO", "RunDiff", "PythagWinPct"]
        
        st.markdown('<div class="section-header">Team Scenario Analyzer</div>', unsafe_allow_html=True)
        st.markdown("Input team performance metrics. Notice how **RunDiff** and **PythagWinPct** heavily influence the XGBoost classification.")
        
        cols = st.columns(4)
        user_input = {}
        for i, col in enumerate(feat_cols):
            median_val = float(data[col].median())
            max_val = float(data[col].quantile(0.99))
            min_val = float(data[col].quantile(0.01))
            if col == "RunDiff":
                min_val = -300.0
                max_val = 300.0
            elif col == "PythagWinPct":
                min_val = 0.2
                max_val = 0.8
            user_input[col] = cols[i%4].number_input(col, min_value=min_val, max_value=max_val, value=median_val)
            
        input_df = pd.DataFrame([user_input])
        prob = model.predict_proba(input_df)[0][1]
        
        c1, c2 = st.columns(2)
        c1.metric("Win Probability", f"{prob:.1%}")
        if prob > 0.5:
            c2.success("Prediction: **WINNING SEASON** 🏆")
        else:
            c2.error("Prediction: **LOSING SEASON** 📉")
            
        st.markdown("---")
        
        # Historical accuracy check
        st.subheader("Historical Model Confidence (ROC Curve equivalent via distribution)")
        data = data.dropna(subset=feat_cols + ["Win_Season"])
        probs_all = model.predict_proba(data[feat_cols])[:,1]
        
        fig = px.histogram(
            x=probs_all, color=data["Win_Season"].astype(str),
            nbins=50, barmode="overlay",
            labels={"x": "Predicted Probability of Winning Season", "color": "Actual Outcome (1=Win)"},
            color_discrete_sequence=["#f87171", "#34d399"],
            opacity=0.75
        )
        fig.update_layout(**plotly_dark(), height=400)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – PLAYER AGING CURVES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Player Aging Curves":
    st.title("📉 Player Aging Curves: Performance vs Age")
    st.caption("Critical for contract negotiations—understand when players peak and decline.")
    
    data = load_data("player_features_5yr.csv").dropna()
    
    # Estimate age: assume debut at ~25 years old, increment by year
    data['estimated_age'] = 25 + (data['yearID'].max() - data['yearID'])
    
    st.markdown('<div class="section-header">📊 League-Wide Aging Pattern</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall performance metrics by age
        age_stats = data.groupby('estimated_age').agg({
            'R': 'mean',
            'H': 'mean', 
            'HR': 'mean',
            'OPS': 'mean'
        }).reset_index()
        age_stats = age_stats[age_stats['estimated_age'] >= 22]
        age_stats = age_stats[age_stats['estimated_age'] <= 40]
        
        fig_ops = px.line(
            age_stats, x='estimated_age', y='OPS',
            markers=True, title='Average OPS by Age',
            labels={'estimated_age': 'Age (years)', 'OPS': 'On-Base Plus Slugging'}
        )
        fig_ops.update_layout(**plotly_dark(), height=400)
        st.plotly_chart(fig_ops, use_container_width=True)
    
    with col2:
        fig_hr = px.line(
            age_stats, x='estimated_age', y='HR',
            markers=True, title='Average Home Runs by Age', line_shape='spline',
            labels={'estimated_age': 'Age (years)', 'HR': 'Home Runs'}
        )
        fig_hr.update_layout(**plotly_dark(), height=400)
        st.plotly_chart(fig_hr, use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">🎯 Individual Player Aging Profile</div>', unsafe_allow_html=True)
    
    # Select a player
    players = data.drop_duplicates('playerID').sort_values('playerID')
    selected_player = st.selectbox("Select Player:", players['playerID'].unique())
    
    player_data = data[data['playerID'] == selected_player].sort_values('yearID')
    player_data['estimated_age'] = 25 + (player_data['yearID'].max() - player_data['yearID'])
    
    if len(player_data) > 1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Peak OPS", f"{player_data['OPS'].max():.3f}")
        col2.metric("Current OPS", f"{player_data['OPS'].iloc[-1]:.3f}")
        col3.metric("Peak Age", f"{int(player_data.loc[player_data['OPS'].idxmax(), 'estimated_age'])} yrs")
        
        fig_player = px.line(
            player_data, x='estimated_age', y='OPS',
            markers=True, title=f'{selected_player} - Career OPS Trajectory',
            labels={'estimated_age': 'Age (years)', 'OPS': 'OPS'},
            hover_data={'yearID': True, 'AB': True, 'HR': True}
        )
        fig_player.update_layout(**plotly_dark(), height=450)
        st.plotly_chart(fig_player, use_container_width=True)
    else:
        st.warning("Not enough data for this player.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 – PLAYER COMPARISON TOOL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Player Comparison Tool":
    st.title("🔍 Player Comparison: Find Similar Players")
    st.caption("Essential for scouting and trade analysis—find comparable players by stats.")
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import euclidean_distances
    
    data = load_data("player_features_5yr.csv").dropna()
    
    # Get latest season stats for each player
    latest_data = data.sort_values('yearID').drop_duplicates('playerID', keep='last')
    
    st.markdown('<div class="section-header">🎯 Find Comps</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_player = st.selectbox(
            "Select a player to find comps:",
            latest_data['playerID'].unique()
        )
    
    with col2:
        num_comps = st.slider("Number of comps:", 3, 10, 5)
    
    # Comparison features
    comp_features = ['AB', 'H', 'HR', 'RBI', 'BB', 'SO', 'BA', 'OBP', 'SLG', 'OPS']
    comp_data = latest_data[comp_features].fillna(0)
    
    # Standardize
    scaler = StandardScaler()
    comp_data_scaled = scaler.fit_transform(comp_data)
    
    # Find selected player index
    player_idx = latest_data[latest_data['playerID'] == selected_player].index[0]
    player_vector = comp_data_scaled[latest_data.index.get_loc(player_idx)]
    
    # Calculate distances
    distances = euclidean_distances([player_vector], comp_data_scaled)[0]
    closest_idx = np.argsort(distances)[1:num_comps+1]
    
    comps = latest_data.iloc[closest_idx].copy()
    comps['distance'] = distances[closest_idx]
    comps = comps.sort_values('distance')
    
    # Display player profile
    player_stats = latest_data[latest_data['playerID'] == selected_player].iloc[0]
    st.markdown(f"### {selected_player} Profile")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AB", int(player_stats['AB']))
    c2.metric("HR", int(player_stats['HR']))
    c3.metric("BA", f"{player_stats['BA']:.3f}")
    c4.metric("OBP", f"{player_stats['OBP']:.3f}")
    c5.metric("OPS", f"{player_stats['OPS']:.3f}")
    
    st.markdown("---")
    st.markdown(f"### Top {num_comps} Comparable Players")
    
    # Display comps as a table
    comp_display = comps[['playerID', 'AB', 'H', 'HR', 'BA', 'OBP', 'SLG', 'OPS']].copy()
    comp_display.columns = ['Player', 'AB', 'H', 'HR', 'BA', 'OBP', 'SLG', 'OPS']
    comp_display = comp_display.round({'BA': 3, 'OBP': 3, 'SLG': 3, 'OPS': 3})
    
    st.dataframe(comp_display.reset_index(drop=True), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Similarity Heatmap")
    
    # Compare selected player with top 3 comps
    top_3_comps = comps.head(3)
    comparison_data = pd.concat([
        latest_data[latest_data['playerID'] == selected_player],
        top_3_comps
    ])[comp_features].fillna(0)
    
    comparison_data_scaled = scaler.transform(comparison_data)
    
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=comparison_data_scaled,
            x=comp_features,
            y=[selected_player] + list(top_3_comps['playerID'].values),
            colorscale='RdYlGn',
            text=np.round(comparison_data_scaled, 2),
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        )
    )
    fig_heatmap.update_layout(
        **plotly_dark(),
        height=300,
        title="Standardized Stats Comparison (scaled 0-1)"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.info("💡 **Use Case**: When evaluating trades or free agents, compare with historical comps to assess contract value and potential.")

elif page == "💬 Sentiment Analysis":
    st.title("💬 Sentiment Analysis")
    st.caption("Classifying fan sentiment (positive / negative) using Logistic Regression & TF-IDF.")
    
    clf = load_model("sentiment_lr.joblib")
    vectorizer = load_model("tfidf_vectorizer.joblib")
    
    if clf is None or vectorizer is None:
        st.warning("NLP models not found. Please train models first.")
    else:
        st.markdown('<div class="section-header">🔍 Live Fan Tweet Predictor</div>', unsafe_allow_html=True)
        user_tweet = st.text_area("Enter a simulated fan tweet:", placeholder="e.g. The new pitcher is absolutely phenomenal, great game!")
        
        if st.button("Analyze Sentiment", type="primary"):
            if user_tweet.strip():
                vec_input = vectorizer.transform([user_tweet])
                prob = clf.predict_proba(vec_input)[0][1]
                
                if prob >= 0.5:
                    st.success(f"✅ **Positive Sentiment** (Confidence: {prob:.1%})")
                else:
                    st.error(f"❌ **Negative Sentiment** (Confidence: {(1-prob):.1%})")
            else:
                st.warning("Please enter some text.")
                
        st.markdown("---")
        st.subheader("How does it work?")
        st.markdown("""
        The model uses **TF-IDF** (Term Frequency-Inverse Document Frequency) with up to **10,000 unigram and bigram features**. 
        It was trained on a balanced dataset of 80,000 tweets. Logistic Regression was chosen over XGBoost here due to its highly efficient handling of sparse, high-dimensional text matrices, enabling near-instant real-time inference on the dashboard.
        """)
