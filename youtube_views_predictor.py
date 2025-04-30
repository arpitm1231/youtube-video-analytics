import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

# ✅ Load or Train Model
model_filename = "youtube_model.pkl"

if os.path.exists(model_filename):
    model = joblib.load(model_filename)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

# ✅ Streamlit UI
st.title("📊 YouTube Video Views Predictor")
st.write("Enter video details to predict the number of views.")

# 🎯 User Inputs
likes = st.number_input("Enter Number of Likes", min_value=0, value=1000, step=100)
comments = st.number_input("Enter Number of Comments", min_value=0, value=100, step=10)

# ✅ Auto-Calculate Ratios
total_interactions = likes + comments + 1  # Avoid division by zero
like_view_ratio = likes / total_interactions
comment_view_ratio = comments / total_interactions

# 🎯 Prediction Button
if st.button("Predict Views"):
    # Log Transform
    log_likes = np.log1p(likes)
    log_comments = np.log1p(comments)

    # Input for Model
    new_video = np.array([[log_likes, log_comments, like_view_ratio, comment_view_ratio]])

    # Predict in Log Scale & Convert Back
    predicted_log_views = model.predict(new_video)[0]
    predicted_views = int(np.expm1(predicted_log_views))

    st.success(f"📌 Predicted Views: **{predicted_views:,}**")  # Adds comma formatting

# ✅ Optional Model Retraining
if st.button("Retrain Model"):
    # 📌 Sample Training Data (Replace with Real YouTube API Data)
    data = {
        "Likes": [1000, 5000, 20000, 50000, 100000],
        "Comments": [100, 500, 2000, 5000, 10000],
        "Views": [10000, 50000, 200000, 500000, 1000000]
    }

    df = pd.DataFrame(data)

    # Apply Log Transformation
    df["Log_Views"] = np.log1p(df["Views"])
    df["Log_Likes"] = np.log1p(df["Likes"])
    df["Log_Comments"] = np.log1p(df["Comments"])
    df["Like_View_Ratio"] = df["Likes"] / df["Views"]
    df["Comment_View_Ratio"] = df["Comments"] / df["Views"]

    # Features & Target
    X = df[["Log_Likes", "Log_Comments", "Like_View_Ratio", "Comment_View_Ratio"]]
    y = df["Log_Views"]

    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save Model
    joblib.dump(model, model_filename)

    st.success("✅ Model Retrained Successfully!")
