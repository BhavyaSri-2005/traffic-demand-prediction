import pandas as pd
import numpy as np
import os

# -----------------------------
# Load Clean Dataset
# -----------------------------
df = pd.read_csv("data/processed/traffic_clean.csv")

# -----------------------------
# Create Rush Hour Feature
# -----------------------------
df["Rush_Hour"] = df["Hour"].apply(
    lambda x: 1 if x in [7,8,9,17,18,19] else 0
)

# -----------------------------
# Create Weekend Feature
# -----------------------------
# (Saturday = 2, Sunday = 3 after Label Encoding in this project.
# Adjust these values if your encoding is different.)
df["Weekend"] = df["Day_of_Week"].apply(
    lambda x: 1 if x in [2,3] else 0
)

# -----------------------------
# Heavy Rain Feature
# -----------------------------
df["Heavy_Rain"] = np.where(df["Rainfall"] >= 20,1,0)

# -----------------------------
# High Temperature Feature
# -----------------------------
df["High_Temperature"] = np.where(df["Temperature"] >= 35,1,0)

# -----------------------------
# Traffic Density Score
# -----------------------------
df["Traffic_Density_Score"] = (
    df["Large_Vehicle_Count"] +
    df["Traffic_Signals"] +
    df["Number_of_Lanes"]
)

# -----------------------------
# Weather Impact
# -----------------------------
weather_impact = {
    0:1,
    1:2,
    2:3,
    3:4
}

df["Weather_Impact"] = df["Weather_Conditions"].map(weather_impact)

# -----------------------------
# Save Dataset
# -----------------------------
os.makedirs("data/processed",exist_ok=True)

df.to_csv(
    "data/processed/traffic_features.csv",
    index=False
)

print("Feature Engineering Completed Successfully")
print(df.head())
print(df.shape)