# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression

# --- App Config ---
st.set_page_config(page_title="Happiness Score Predictor 😊", page_icon="😊")
st.title("😊 Predict the Happiness Score of a Country")
st.write("Adjust the sliders below to estimate a country's happiness score.")

# --- Paths ---
model_path = Path("models/model.joblib")
data_path = Path("data/raw/world_happiness.csv")

# --- If no model found, train automatically ---
if not model_path.exists():
    st.warning("Model not found — training a new one automatically...")

    df = pd.read_csv(data_path)
    features = [
        "GDP per capita",
        "Social support",
        "Healthy life expectancy",
        "Freedom to make life choices",
        "Generosity",
        "Perceptions of corruption"
    ]
    X = df[features]
    y = df["Score"]

    model = LinearRegression().fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features}, model_path)
    st.success("✅ Model trained successfully!")
else:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]

# --- User Input ---
st.subheader("Input Factors")

gdp = st.slider("GDP per capita", 0.0, 2.0, 1.0, 0.01)
support = st.slider("Social support", 0.0, 2.0, 1.0, 0.01)
health = st.slider("Healthy life expectancy", 0.0, 1.5, 0.8, 0.01)
freedom = st.slider("Freedom to make life choices", 0.0, 1.0, 0.5, 0.01)
generosity = st.slider("Generosity", 0.0, 1.0, 0.2, 0.01)
corruption = st.slider("Perceptions of corruption", 0.0, 1.0, 0.3, 0.01)

X = np.array([[gdp, support, health, freedom, generosity, corruption]])
pred = model.predict(X)[0]
st.success(f"🌏 Predicted Happiness Score: **{pred:.2f}**")

# --- Feature Importance ---
st.subheader("Feature Importance")
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
}).sort_values("Coefficient", ascending=True)

st.bar_chart(coef_df.set_index("Feature"))

st.caption("Linear Regression coefficients show each feature’s impact on the predicted happiness score.")