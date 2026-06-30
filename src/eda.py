import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/processed/traffic_features.csv")

# Create output folder
os.makedirs("outputs/plots", exist_ok=True)

# -----------------------------
# 1. Traffic Demand Distribution
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(df["Traffic_Demand"], bins=30)
plt.title("Traffic Demand Distribution")
plt.xlabel("Traffic Demand")
plt.ylabel("Count")
plt.savefig("outputs/plots/traffic_distribution.png")
plt.close()

# -----------------------------
# 2. Traffic by Hour
# -----------------------------
hourly = df.groupby("Hour")["Traffic_Demand"].mean()

plt.figure(figsize=(10,5))
plt.plot(hourly.index, hourly.values, marker="o")
plt.title("Average Traffic by Hour")
plt.xlabel("Hour")
plt.ylabel("Average Traffic")
plt.savefig("outputs/plots/traffic_by_hour.png")
plt.close()

# -----------------------------
# 3. Traffic by Weather
# -----------------------------
weather = df.groupby("Weather_Conditions")["Traffic_Demand"].mean()

plt.figure(figsize=(8,5))
plt.bar(weather.index.astype(str), weather.values)
plt.title("Traffic by Weather")
plt.xlabel("Weather")
plt.ylabel("Average Traffic")
plt.savefig("outputs/plots/weather_analysis.png")
plt.close()

# -----------------------------
# 4. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(12,8))

corr = df.corr(numeric_only=True)

plt.imshow(corr, cmap="coolwarm")

plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.tight_layout()

plt.savefig("outputs/plots/heatmap.png")

plt.close()

print("EDA Completed Successfully")