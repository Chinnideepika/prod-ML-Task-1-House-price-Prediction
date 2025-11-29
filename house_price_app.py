# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 21:55:18 2025

@author: Deepika
"""


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="House Price – Linear Regression", layout="centered")

st.title("🏠 House Price Prediction – Linear Regression")
st.write(
    "Task: Implement a linear regression model to predict house prices "
    "based on square footage, number of bedrooms, and bathrooms (with an extended version)."
)

# -------------------------
# File upload
# -------------------------
st.header("1. Upload Kaggle train.csv")
uploaded_file = st.file_uploader("Upload train.csv from the House Prices competition", type=["csv"])

if uploaded_file is None:
    st.info("Please upload train.csv to continue.")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)
st.write(f"Dataset loaded. Shape: {df.shape}")

if st.checkbox("Show first 5 rows"):
    st.dataframe(df.head())

# -------------------------
# Model choice
# -------------------------
st.header("2. Choose model type")

MODEL_SIMPLE = "Simple Model (GrLivArea, BedroomAbvGr, FullBath)"
MODEL_EXT = "Extended Model (+ LotArea, GarageCars, OverallQual, YearBuilt)"

model_choice = st.radio(
    "Select which model to train:",
    [MODEL_SIMPLE, MODEL_EXT],
    index=0
)

if model_choice == MODEL_SIMPLE:
    FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath"]
else:
    FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath",
                "LotArea", "GarageCars", "OverallQual", "YearBuilt"]

TARGET = "SalePrice"

st.write("**Features used:**", ", ".join(FEATURES))
st.write("**Target:**", TARGET)

# -------------------------
# Prepare data
# -------------------------
st.header("3. Prepare and train model")

# Handle missing values
X_df = df[FEATURES].copy()
y = df[TARGET].copy()

X_df = X_df.fillna(X_df.median())
mask = ~y.isnull()
X_clean = X_df[mask].values
y_clean = y[mask].values

st.write(f"Rows used for modeling: {X_clean.shape[0]} (dropped {len(df) - X_clean.shape[0]} rows with missing target).")

test_size = st.slider("Test size (percentage for testing)", min_value=10, max_value=40, value=20, step=5)
random_state = st.number_input("Random state (for reproducible split)", value=42, step=1)

if st.button("Train Linear Regression model"):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean,
        test_size=test_size / 100.0,
        random_state=int(random_state)
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model parameters")
    st.write("Intercept (b0):", model.intercept_)
    coef_df = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": model.coef_
    })
    st.table(coef_df)

    st.subheader("Evaluation on test set")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R²:** {r2:.3f}")

    # Plot Actual vs Predicted
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--')
    ax.set_xlabel("Actual SalePrice")
    ax.set_ylabel("Predicted SalePrice")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

    # Store model and feature info in session_state for predictions
    st.session_state["model"] = model
    st.session_state["features"] = FEATURES
    st.session_state["feature_medians"] = X_df.median()

# -------------------------
# Predict a single example
# -------------------------
st.header("4. Predict a new house price")

if "model" not in st.session_state:
    st.info("Train the model first to enable prediction.")
else:
    model = st.session_state["model"]
    FEATURES = st.session_state["features"]
    medians = st.session_state["feature_medians"]

    st.write("Enter feature values for a new house:")

    input_vals = {}
    for f in FEATURES:
        default_val = float(medians[f]) if f in medians.index else 0.0
        input_vals[f] = st.number_input(f, value=default_val)

    if st.button("Predict Price"):
        example = np.array([[input_vals[f] for f in FEATURES]])
        pred_price = model.predict(example)[0]
        st.success(f"Predicted SalePrice: {int(pred_price):,}")
