import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/processed/traffic_features.csv")

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop(["Timestamp", "Traffic_Demand"], axis=1)
y = df["Traffic_Demand"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Load Models
# -----------------------------
models = {
    "Random Forest": joblib.load("outputs/models/random_forest.pkl"),
    "XGBoost": joblib.load("outputs/models/xgboost.pkl"),
    "LightGBM": joblib.load("outputs/models/lightgbm.pkl"),
    "Ensemble": joblib.load("outputs/models/ensemble.pkl")
}

results = []

# -----------------------------
# Evaluate Models
# -----------------------------
for name, model in models.items():

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

# -----------------------------
# Save Report
# -----------------------------
results_df = pd.DataFrame(results)

os.makedirs("outputs/reports", exist_ok=True)

results_df.to_csv(
    "outputs/reports/model_comparison.csv",
    index=False
)

print(results_df)

print("\nModel comparison saved successfully.")