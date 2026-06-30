import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from datetime import datetime
from prediction import predict_traffic

from charts import (
    traffic_trend_chart,
    weather_chart,
    road_chart,
    feature_importance_chart,
    gauge_chart
)
# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(
    page_title="TrafficSense AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# CUSTOM CSS
# ------------------------------
st.markdown("""
<style>

/* Hide Streamlit Menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Background */
.stApp{
background:linear-gradient(135deg,#0f172a,#111827,#1e293b);
color:white;
}

/* Title */
.title{
font-size:55px;
font-weight:800;
text-align:center;
color:#38bdf8;
}

/* Subtitle */
.subtitle{
font-size:22px;
text-align:center;
color:white;
margin-bottom:30px;
}

/* Glass Card */
.card{
background:rgba(255,255,255,0.08);
padding:25px;
border-radius:20px;
backdrop-filter:blur(12px);
box-shadow:0px 0px 25px rgba(0,255,255,0.25);
transition:0.3s;
}

.card:hover{
transform:scale(1.02);
box-shadow:0px 0px 35px cyan;
}

/* Metric */
.metric{
font-size:36px;
font-weight:bold;
color:#22d3ee;
text-align:center;
}

.metric-title{
font-size:18px;
color:white;
text-align:center;
}

/* Button */
.stButton>button{
width:100%;
height:55px;
font-size:18px;
font-weight:bold;
border-radius:12px;
background:linear-gradient(90deg,#06b6d4,#2563eb);
color:white;
border:none;
}

.stButton>button:hover{
background:linear-gradient(90deg,#8b5cf6,#ec4899);
}

[data-testid="stSidebar"]{
background:#020617;
}

</style>
""", unsafe_allow_html=True)
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3208/3208718.png",
        width=120
    )

    st.title("TrafficSense AI")

    selected = option_menu(

        menu_title="Navigation",

        options=[
            "Dashboard",
            "Prediction",
            "Analytics",
            "About"
        ],

        icons=[
            "house",
            "cpu",
            "bar-chart",
            "info-circle"
        ],

        default_index=0

    )
    st.markdown(
"""
<div class="title">
🚦 TrafficSense AI
</div>

<div class="subtitle">
Smart Traffic Demand Prediction using Artificial Intelligence
</div>
""",
unsafe_allow_html=True
)
    st.image(

"https://images.unsplash.com/photo-1449824913935-59a10b8d2000",

use_container_width=True

)
    st.write("")

c1,c2,c3,c4=st.columns(4)

with c1:

    st.markdown("""
<div class="card">

<div class="metric">
120K
</div>

<div class="metric-title">
Traffic Records
</div>

</div>
""",unsafe_allow_html=True)

with c2:

    st.markdown("""
<div class="card">

<div class="metric">
19
</div>

<div class="metric-title">
Features
</div>

</div>
""",unsafe_allow_html=True)

with c3:

    st.markdown("""
<div class="card">

<div class="metric">
4
</div>

<div class="metric-title">
ML Models
</div>

</div>
""",unsafe_allow_html=True)

with c4:

    st.markdown(f"""
<div class="card">

<div class="metric">
{datetime.now().strftime("%H:%M")}
</div>

<div class="metric-title">
Current Time
</div>

</div>
""",unsafe_allow_html=True)
    # ==========================================================
# AI DASHBOARD
# ==========================================================

st.write("")
st.markdown("## 📊 Live Traffic Analytics")

left, right = st.columns([2, 1])

# -------------------------------
# Traffic Trend Chart
# -------------------------------
with left:

    traffic = pd.DataFrame({
        "Hour":[0,1,2,3,4,5,6,7,8,9,10,11,
                12,13,14,15,16,17,18,19,20,21,22,23],

        "Traffic":[
            50,45,40,35,30,45,90,180,250,280,260,240,
            230,220,240,260,300,340,360,320,260,180,120,80
        ]
    })

    fig = px.area(
        traffic,
        x="Hour",
        y="Traffic",
        title="🚗 Hourly Traffic Trend"
    )

    fig.update_layout(
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Weather Analysis
# -------------------------------
with right:

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

    fig2 = px.pie(
        weather,
        values="Traffic",
        names="Weather",
        hole=0.6,
        title="🌦 Weather Distribution"
    )

    fig2.update_layout(
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(fig2, use_container_width=True)

st.write("")
st.markdown("## 🤖 AI Insights")

c1, c2, c3 = st.columns(3)

with c1:

    st.info("""

### 🚦 Peak Hour

7 AM – 10 AM

5 PM – 8 PM

""")

with c2:

    st.success("""

### 🌤 Best Weather

Sunny

Cloudy

""")

with c3:

    st.warning("""

### ⚠ Congestion

Rainfall

Special Events

Heavy Vehicles

""")
    st.write("")
st.markdown("## 🛣 Road Analytics")

road = pd.DataFrame({

"Road":[

"Highway",

"City",

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

title="Traffic by Road Type"

)

fig.update_layout(

template="plotly_dark",

height=400

)

st.plotly_chart(fig,use_container_width=True)
st.markdown("## ⭐ Important Features")

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

0.25,

0.18,

0.12,

0.09,

0.05

]

})

fig = px.bar(

features,

x="Importance",

y="Feature",

orientation="h",

color="Importance",

title="Feature Importance"

)

fig.update_layout(

template="plotly_dark",

height=450

)

st.plotly_chart(fig,use_container_width=True)
st.markdown("## 🧠 AI Recommendation")

st.success("""
🚗 Avoid travelling during **7–10 AM** and **5–8 PM**.

🌦 Rainy weather increases congestion.

🚦 Highways carry the highest traffic demand.

🤖 Ensemble Machine Learning predicts traffic more accurately than a single model.

📍 Traffic management authorities can use these predictions for signal optimization.
""")
# ==========================================================
# AI PREDICTION CENTER
# ==========================================================

st.markdown("## 🤖 AI Traffic Prediction")

# Load model
try:
    model = joblib.load("outputs/models/ensemble.pkl")
except:
    model = None
    st.warning("⚠️ Model not found. Prediction is disabled.")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        day = st.selectbox(
            "📅 Day",
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        )

        hour = st.slider(
            "🕒 Hour",
            0,
            23,
            8
        )

        road = st.selectbox(
            "🛣 Road Type",
            ["Highway","City Road","Residential"]
        )

        lanes = st.selectbox(
            "🚗 Number of Lanes",
            [2,4,6]
        )

        signals = st.slider(
            "🚦 Traffic Signals",
            0,
            5,
            2
        )

    with col2:

        weather = st.selectbox(
            "🌦 Weather",
            ["Sunny","Cloudy","Rainy","Foggy"]
        )

        temperature = st.slider(
            "🌡 Temperature",
            15,
            45,
            30
        )

        humidity = st.slider(
            "💧 Humidity",
            20,
            100,
            70
        )

        rainfall = st.slider(
            "🌧 Rainfall",
            0.0,
            40.0,
            5.0
        )

        vehicles = st.slider(
            "🚚 Heavy Vehicles",
            0,
            40,
            10
        )

    predict = st.form_submit_button("🚀 Predict Traffic")
    if predict:

      from prediction import predict_traffic

    prediction = predict_traffic(
            day,
            hour,
            road,
            lanes,
            signals,
            temperature,
            humidity,
            rainfall,
            weather,
            "None",
            "No",
            vehicles
        )

    st.write("Prediction:", prediction)
    st.markdown("---")

    if prediction < 180:

            color = "#16a34a"
            status = "🟢 LOW TRAFFIC"

    elif predict < 300:

            color = "#f59e0b"
            status = "🟡 MODERATE TRAFFIC"

    else:

            color = "#dc2626"
            status = "🔴 HEAVY TRAFFIC"

    st.markdown(f"""
        <div style="
        background:{color};
        padding:30px;
        border-radius:20px;
        text-align:center;
        color:white;
        ">

        <h1>{status}</h1>

        <h2>{prediction:.2f}</h2>

        </div>

        """, unsafe_allow_html=True)
    # ===========================================
# AI Gauge Meter
# ===========================================

st.markdown("## 🚦 Traffic Demand Meter")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=float(prediction),

    title={
        "text":"Traffic Demand Score"
    },

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

    height=420

)

st.plotly_chart(fig,use_container_width=True)
st.markdown("## 🤖 AI Recommendation")

if prediction < 180:

    st.success("""

✅ Low Traffic

• Roads are clear

• Safe to travel

• No congestion expected

""")

elif prediction < 300:

    st.warning("""

⚠ Moderate Traffic

• Slight congestion

• Travel carefully

• Expect small delays

""")

else:

    st.error("""

🚨 Heavy Traffic

• Avoid this route

• Use alternate roads

• Delay expected

""")
    st.markdown("## 📊 Prediction History")

history = pd.DataFrame({

"Time":[

"08:00",

"09:30",

"12:00",

"15:00",

"18:00"

],

"Prediction":[

145,

220,

180,

265,

340

],

"Status":[

"Low",

"Moderate",

"Low",

"Moderate",

"Heavy"

]

})

st.dataframe(

history,

use_container_width=True,

hide_index=True

)
st.markdown("## 📥 Download Report")

report = pd.DataFrame({

"Prediction":[predict],

"Day":[day],

"Hour":[hour],

"Weather":[weather],

"Temperature":[temperature],

"Humidity":[humidity],

"Rainfall":[rainfall],

"Road":[road]

})

csv = report.to_csv(index=False).encode("utf-8")

st.download_button(

label="📄 Download Prediction Report",

data=csv,

file_name="traffic_prediction_report.csv",

mime="text/csv"

)
st.write("Prediction:", prediction)