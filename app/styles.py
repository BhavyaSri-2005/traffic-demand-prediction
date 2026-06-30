def load_css():
    return """
    <style>

    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}

    .stApp{
        background:linear-gradient(135deg,#0f172a,#111827,#1e293b);
        color:white;
    }

    .title{
        font-size:50px;
        font-weight:800;
        text-align:center;
        color:#38bdf8;
    }

    .subtitle{
        text-align:center;
        color:white;
        font-size:20px;
        margin-bottom:20px;
    }

    .glass{
        background:rgba(255,255,255,0.08);
        border-radius:20px;
        padding:20px;
        backdrop-filter:blur(12px);
        box-shadow:0px 0px 20px rgba(0,255,255,.25);
    }

    .card{
        background:#16213E;
        border-radius:20px;
        padding:20px;
        text-align:center;
        color:white;
        box-shadow:0px 0px 20px cyan;
    }

    .metric{
        font-size:35px;
        font-weight:bold;
        color:#22d3ee;
    }

    .stButton>button{
        width:100%;
        height:55px;
        border:none;
        border-radius:15px;
        font-size:18px;
        font-weight:bold;
        background:linear-gradient(90deg,#00c6ff,#0072ff);
        color:white;
    }

    .stButton>button:hover{
        background:linear-gradient(90deg,#9333ea,#ec4899);
    }

    </style>
    """