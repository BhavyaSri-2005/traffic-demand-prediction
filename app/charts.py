import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------
# Hourly Traffic Trend
# ---------------------------------------
def traffic_trend_chart():

    df = pd.DataFrame({

        "Hour":[0,1,2,3,4,5,6,7,8,9,10,11,
                12,13,14,15,16,17,18,19,20,21,22,23],

        "Traffic":[
            60,55,50,45,40,60,
            120,210,280,300,290,260,
            240,235,250,280,320,360,
            380,340,260,180,120,80
        ]

    })

    fig = px.area(
        df,
        x="Hour",
        y="Traffic",
        title="🚗 Hourly Traffic Trend"
    )

    fig.update_layout(
        template="plotly_dark",
        height=420
    )

    return fig


# ---------------------------------------
# Weather Distribution
# ---------------------------------------
def weather_chart():

    weather = pd.DataFrame({

        "Weather":[
            "Sunny",
            "Cloudy",
            "Rainy",
            "Foggy"
        ],

        "Traffic":[
            180,
            220,
            320,
            260
        ]

    })

    fig = px.pie(

        weather,

        values="Traffic",

        names="Weather",

        hole=0.6,

        title="🌦 Weather Distribution"

    )

    fig.update_layout(

        template="plotly_dark",

        height=420

    )

    return fig


# ---------------------------------------
# Road Analysis
# ---------------------------------------
def road_chart():

    road = pd.DataFrame({

        "Road":[
            "Highway",
            "City Road",
            "Residential"
        ],

        "Traffic":[
            340,
            260,
            180
        ]

    })

    fig = px.bar(

        road,

        x="Road",

        y="Traffic",

        color="Traffic",

        text="Traffic",

        title="🛣 Road Analysis"

    )

    fig.update_layout(

        template="plotly_dark",

        height=420

    )

    return fig


# ---------------------------------------
# Feature Importance
# ---------------------------------------
def feature_importance_chart():

    features = pd.DataFrame({

        "Feature":[

            "Hour",

            "Rainfall",

            "Temperature",

            "Vehicles",

            "Signals",

            "Humidity"

        ],

        "Importance":[

            0.31,

            0.24,

            0.18,

            0.12,

            0.09,

            0.06

        ]

    })

    fig = px.bar(

        features,

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        title="⭐ Feature Importance"

    )

    fig.update_layout(

        template="plotly_dark",

        height=420

    )

    return fig


# ---------------------------------------
# Gauge Meter
# ---------------------------------------
def gauge_chart(prediction):

    fig = go.Figure(go.Indicator(

        mode="gauge+number",

        value=float(prediction),

        title={"text":"🚦 Traffic Demand"},

        gauge={

            "axis":{"range":[0,400]},

            "bar":{"color":"cyan"},

            "steps":[

                {"range":[0,180],"color":"green"},

                {"range":[180,300],"color":"orange"},

                {"range":[300,400],"color":"red"}

            ]

        }

    ))

    fig.update_layout(

        template="plotly_dark",

        height=400

    )

    return fig