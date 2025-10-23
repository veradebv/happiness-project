import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# App title
st.set_page_config(page_title="Happiness Score Predictor 😊", page_icon="😊")
st.title("😊 Predict the Happiness Score of a Country")
st.write("Adjust the sliders below to estimate a country's happiness score.")

# Load trained model
model_path = Path("models/model.joblib")

if not model_path.exists():
    st.error("Model file not found! Please run `python src/train.py` first.")
else:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]

    # Input sliders for each feature
    inputs = []
    st.subheader("Input Factors")

    gdp = st.slider("GDP per capita", 0.0, 2.0, 1.0, 0.01)
    support = st.slider("Social support", 0.0, 2.0, 1.0, 0.01)
    health = st.slider("Healthy life expectancy", 0.0, 1.5, 0.8, 0.01)
    freedom = st.slider("Freedom to make life choices", 0.0, 1.0, 0.5, 0.01)
    generosity = st.slider("Generosity", 0.0, 1.0, 0.2, 0.01)
    corruption = st.slider("Perceptions of corruption", 0.0, 1.0, 0.3, 0.01)

    X = np.array([[gdp, support, health, freedom, generosity, corruption]])

    # Predict
    pred = model.predict(X)[0]
    st.success(f"🌏 Predicted Happiness Score: **{pred:.2f}**")
