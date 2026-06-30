import os
import joblib
import pandas as pd

MODEL_PATH = "outputs/models/ensemble.pkl"

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


def predict_traffic(
    day,
    hour,
    road,
    lanes,
    signals,
    temperature,
    humidity,
    rainfall,
    weather,
    landmark,
    event,
    vehicles
):

    if model is None:
        raise FileNotFoundError("outputs/models/ensemble.pkl not found")

    day_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }

    road_map = {
        "Highway": 0,
        "City Road": 1,
        "Residential": 2
    }

    weather_map = {
        "Sunny": 0,
        "Cloudy": 1,
        "Rainy": 2,
        "Foggy": 3
    }

    landmark_map = {
        "Mall": 0,
        "School": 1,
        "Hospital": 2,
        "Office": 3,
        "None": 4
    }

    event_map = {
        "No": 0,
        "Yes": 1
    }

    input_df = pd.DataFrame([{
        "Day_of_Week": day_map[day],
        "Hour": hour,
        "Road_Type": road_map[road],
        "Number_of_Lanes": lanes,
        "Traffic_Signals": signals,
        "Temperature": temperature,
        "Humidity": humidity,
        "Rainfall": rainfall,
        "Weather_Conditions": weather_map[weather],
        "Nearby_Landmarks": landmark_map[landmark],
        "Event_Indicator": event_map[event],
        "Large_Vehicle_Count": vehicles,
        "GeoHash_Location": 0,
        "Rush_Hour": 1 if hour in [7,8,9,17,18,19] else 0,
        "Weekend": 1 if day in ["Saturday","Sunday"] else 0,
        "Heavy_Rain": 1 if rainfall >= 20 else 0,
        "High_Temperature": 1 if temperature >= 35 else 0,
        "Traffic_Density_Score": lanes + signals + vehicles,
        "Weather_Impact": weather_map[weather] + 1
    }])

    prediction = model.predict(input_df)

    return float(prediction[0])