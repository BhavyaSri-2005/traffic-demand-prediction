import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from lightgbm import LGBMRegressor

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
# Train Model
# -----------------------------
model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("LightGBM Results")
print("----------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("outputs/models", exist_ok=True)

joblib.dump(
    model,
    "outputs/models/lightgbm.pkl"
)

print("LightGBM Model Saved Successfully")