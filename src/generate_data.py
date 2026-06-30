import pandas as pd
import numpy as np
import random
import os

# -----------------------------
# Number of records
# -----------------------------
num_records = 120000

np.random.seed(42)
random.seed(42)

# -----------------------------
# Timestamp
# -----------------------------
timestamps = pd.date_range(
    start="2024-01-01",
    periods=num_records,
    freq="5min"
)

# -----------------------------
# Basic Features
# -----------------------------
day_of_week = timestamps.day_name()

hour = timestamps.hour

road_type = np.random.choice(
    ["Highway", "City Road", "Residential"],
    num_records,
    p=[0.30,0.50,0.20]
)

number_of_lanes = np.random.choice(
    [2,4,6],
    num_records,
    p=[0.4,0.4,0.2]
)

traffic_signals = np.random.randint(0,6,num_records)

temperature = np.random.randint(18,41,num_records)

humidity = np.random.randint(30,96,num_records)

rainfall = np.round(np.random.uniform(0,40,num_records),2)

weather = np.random.choice(
    ["Sunny","Cloudy","Rainy","Foggy"],
    num_records,
    p=[0.45,0.25,0.20,0.10]
)

nearby_landmarks = np.random.choice(
    ["Mall","School","Hospital","Office","None"],
    num_records
)

event_indicator = np.random.choice(
    ["Yes","No"],
    num_records,
    p=[0.10,0.90]
)

large_vehicle_count = np.random.randint(0,40,num_records)

geohash_location = np.random.choice(
    ["GH001","GH002","GH003","GH004","GH005"],
    num_records
)

# -----------------------------
# Target Variable
# -----------------------------
traffic_demand = []

for i in range(num_records):

    demand = 100

    if hour[i] in [7,8,9,17,18,19]:
        demand += 150

    if weather[i] == "Rainy":
        demand += 80

    if weather[i] == "Foggy":
        demand += 40

    if road_type[i] == "Highway":
        demand += 120

    if event_indicator[i] == "Yes":
        demand += 100

    demand += large_vehicle_count[i] * 2

    demand += random.randint(-30,30)

    traffic_demand.append(max(50,demand))

# -----------------------------
# Create DataFrame
# -----------------------------
df = pd.DataFrame({

    "Timestamp":timestamps,

    "Day_of_Week":day_of_week,

    "Hour":hour,

    "Road_Type":road_type,

    "Number_of_Lanes":number_of_lanes,

    "Traffic_Signals":traffic_signals,

    "Temperature":temperature,

    "Humidity":humidity,

    "Rainfall":rainfall,

    "Weather_Conditions":weather,

    "Nearby_Landmarks":nearby_landmarks,

    "Event_Indicator":event_indicator,

    "Large_Vehicle_Count":large_vehicle_count,

    "GeoHash_Location":geohash_location,

    "Traffic_Demand":traffic_demand

})

# -----------------------------
# Save Dataset
# -----------------------------
os.makedirs("data/raw", exist_ok=True)

df.to_csv(
    "data/raw/traffic_dataset.csv",
    index=False
)

print("Dataset Generated Successfully")
print(df.head())
print(df.shape)