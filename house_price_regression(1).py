# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 21:42:42 2025

@author: Deepika
"""

# -*- coding: utf-8 -*-
"""
House Price Prediction using Linear Regression (7 features)
Task: Predict SalePrice using:
GrLivArea, BedroomAbvGr, FullBath, LotArea, GarageCars, OverallQual, YearBuilt
"""

# 0. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. LOAD DATASET
# Change this path if your train.csv is in a different location
df = pd.read_csv(
    r"C:\Users\Deepika\Downloads\house-prices-advanced-regression-techniques\train.csv"
)

print("Dataset shape (rows, cols):", df.shape)

# 2. CHOOSE FEATURES AND TARGET
FEATURES = [
    "GrLivArea",   
    "BedroomAbvGr",
    "FullBath",   
    "LotArea",    
    "GarageCars",  
    "OverallQual", 
    "YearBuilt"    
]
TARGET = "SalePrice"

print("\nPreview of chosen columns:")
print(df[FEATURES + [TARGET]].head())

# 3. HANDLE MISSING VALUES
# Take copies so we don't accidentally modify original df
X_df = df[FEATURES].copy()
y = df[TARGET].copy()

# Fill missing feature values with median (safe choice)
X_df = X_df.fillna(X_df.median())

# Drop rows where target (SalePrice) is missing (should be rare in train.csv)
mask = ~y.isnull()
X_clean = X_df[mask].values   # features as numpy array
y_clean = y[mask].values      # target as numpy array

print(f"\nRows used for modeling: {X_clean.shape[0]} (dropped {len(df) - X_clean.shape[0]} rows)")

# 4. SPLIT INTO TRAIN AND TEST SETS
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)
print("Train rows:", X_train.shape[0], "Test rows:", X_test.shape[0])

# 5. TRAIN LINEAR REGRESSION MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# 6. PRINT INTERCEPT AND COEFFICIENTS
print("\nModel parameters:")
print("Intercept (b0):", model.intercept_)
for fname, coef in zip(FEATURES, model.coef_):
    print(f"  {fname}: {coef:.4f}")

# 7. PREDICT ON TEST SET AND EVALUATE
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nEvaluation on hold-out test set:")
print(f"  RMSE: {rmse:.2f}")
print(f"  R²: {r2:.3f}")

# 8. PLOT ACTUAL VS PREDICTED
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
minv = min(y_test.min(), y_pred.min())
maxv = max(y_test.max(), y_pred.max())
plt.plot([minv, maxv], [minv, maxv], 'r--', linewidth=1)
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted SalePrice (7-feature model)")
plt.tight_layout()
plt.show()

# 9. EXAMPLE SINGLE PREDICTION
# Example house: 2000 sqft, 3 beds, 2 baths, 8000 lot area, 2 garage cars, quality 7, built in 1995
example = np.array([[2000, 3, 2, 8000, 2, 7, 1995]])
pred_price = model.predict(example)[0]
print("\nExample prediction for house:")
print("  2000 sqft, 3 bed, 2 bath, LotArea=8000, GarageCars=2, OverallQual=7, YearBuilt=1995")
print("Predicted Price:", int(pred_price))
