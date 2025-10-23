# 🌎 Predicting the Happiness Score of a Country

[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-success?logo=streamlit)](https://happiness-project.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A data science project that predicts a country's **Happiness Score** based on socioeconomic indicators such as GDP, social support, and health expectancy — powered by a linear regression model and deployed with Streamlit.

---

## 🚀 Live Demo

🎯 **Try it here:**  
👉 [https://happiness-project.streamlit.app](https://happiness-project.streamlit.app)

Move the sliders and see how GDP, freedom, or generosity affect a country's predicted happiness score in real time.

---

## 📊 Project Overview

This project explores data from the **World Happiness Report** to understand which factors most influence a nation's happiness and builds a simple yet powerful **machine learning model** to predict happiness scores.

| Objective | Details |
|------------|----------|
| 🎯 **Goal** | Predict the Happiness Score from social and economic factors |
| 🧠 **Model** | Linear Regression (baseline) |
| 📈 **Dataset** | World Happiness Report (Kaggle) |
| 🧰 **Tech Stack** | Python, Pandas, scikit-learn, Streamlit, Matplotlib |
| ☁️ **Deployment** | Streamlit Cloud |

---

## 🧩 Dataset Description

Source: *World Happiness Report 2019*  
Each record represents a country with several features:

| Column | Description |
|:--|:--|
| `GDP per capita` | Economic indicator |
| `Social support` | Level of family/community support |
| `Healthy life expectancy` | Average health/longevity score |
| `Freedom to make life choices` | Measure of personal freedom |
| `Generosity` | Measure of charitable giving |
| `Perceptions of corruption` | Trust in institutions |
| `Score` | Target happiness score (0–10 scale) |

---

## 🧪 Project Pipeline

1. **Data Exploration (EDA)**  
   - Correlation heatmaps  
   - Pairwise plots  
   - Feature–target relationships  

2. **Model Training (`src/train.py`)**  
   - Linear Regression model  
   - Evaluation metrics (R², RMSE)  
   - Model saved as `models/model.joblib`

3. **Model Evaluation (`src/evaluate.py`)**  
   - Scatter plot: Actual vs. Predicted  
   - Saved visuals in `reports/figures/`

4. **Feature Importance (`src/feature_importance.py`)**  
   - Bar plot of model coefficients  
   - Identifies top contributors to happiness

5. **Streamlit Web App (`streamlit_app.py`)**  
   - Interactive sliders for all features  
   - Instant prediction updates  
   - Live bar chart of feature importance  

---

## 📈 Model Performance

| Metric | Value |
|:--|:--:|
| **R² Score** | ~0.79 |
| **RMSE** | ~0.35 |
| **Model Type** | Linear Regression |

---

## 🌿 Feature Importance Visualization

*(Automatically generated in the app)*  

| Feature | Relative Impact |
|:--|:--:|
| GDP per capita | ⬆️ Strong positive |
| Social support | ⬆️ Positive |
| Healthy life expectancy | ⬆️ Positive |
| Freedom to make life choices | ⬆️ Positive |
| Generosity | ⬇️ Weak |
| Perceptions of corruption | ⬇️ Negative |

---

## 🧰 Installation & Local Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/happiness-project.git
   cd happiness-project
