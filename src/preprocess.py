import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/raw/traffic_dataset.csv")

print("Original Shape :", df.shape)

# -----------------------------
# Remove Duplicate Rows
# -----------------------------
df.drop_duplicates(inplace=True)

# -----------------------------
# Handle Missing Values
# -----------------------------
df.fillna({
    "Temperature": df["Temperature"].mean(),
    "Humidity": df["Humidity"].mean(),
    "Rainfall": df["Rainfall"].mean()
}, inplace=True)

# Fill missing values in categorical columns
categorical_columns = [
    "Road_Type",
    "Weather_Conditions",
    "Nearby_Landmarks",
    "Event_Indicator",
    "GeoHash_Location",
    "Day_of_Week"
]

for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# -----------------------------
# Convert Timestamp
# -----------------------------
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# -----------------------------
# Encode Categorical Columns
# -----------------------------
encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# -----------------------------
# Save Clean Dataset
# -----------------------------
os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/traffic_clean.csv",
    index=False
)

print("Clean Dataset Saved Successfully")
print(df.head())
print("Processed Shape :", df.shape)