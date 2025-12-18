from datetime import datetime, timedelta
import streamlit as st
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import logging
import numpy as np
import pandas as pd

st.markdown("""
<style>
    /* Center the content */
    .main {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh; /* Full viewport height */
        background-color: var(--background);
    }
    .stApp {
        background-color: #f1f5f9;
    }
    /* Center the app header */
    .app-header {
        text-align: center;
        margin: auto;
        max-width: 2000px; /* Increased width from 800px */
        padding: 20px; /* Add padding */
    }
    /* Adjust the button alignment */
    .stButton > button {
        margin: auto;
        display: block;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
    /* Custom font settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    :root {
        --primary-red: #FF4A4A; /* Your group's color */
        --danger-red: #FF4A4A; /* Matching the primary color */
        --warning-amber: #f59e0b;
        --safe-green: #16a34a;
        --background: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #000000; /* Black font for primary text */
        --text-secondary: #000000; /* Black font for secondary text */
    }

    body, .stApp {
        font-family: 'Inter', sans-serif; /* Use Inter font */
        background-color: var(--background);
        color: var(--text-primary); /* Set default text color to black */
    }

    .app-header {
        background: var(--primary-red); /* Apply the custom color */
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        font-family: 'Inter', sans-serif; /* Ensure consistent font */
    }

    .app-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700; /* Bold font */
    }

    .app-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-weight: 400; /* Regular font */
    }

    .stButton > button {
        background: var(--primary-red); /* Button color */
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600; /* Semi-bold font */
        width: 100%;
        transition: all 0.2s;
        font-family: 'Inter', sans-serif; /* Ensure consistent font */
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 74, 74, 0.4); /* Hover effect */
    }
</style>
""", unsafe_allow_html=True)


def generate_simulated_ecg_features():
    """Generate simulated ECG-derived features for prediction"""
    return {
        'hrv_sdnn': np.random.normal(45, 15),  # HRV SDNN in ms
        'hrv_rmssd': np.random.normal(35, 12),  # HRV RMSSD in ms
        'hr_mean': np.random.normal(75, 10),    # Mean heart rate
        'hr_variability': np.random.normal(8, 3),
        'qt_interval': np.random.normal(400, 30),  # QT interval in ms
        'st_deviation': np.random.normal(0, 0.5),  # ST segment deviation
    }


def calculate_hypoglycemia_probability(features, time_of_day, history):
    """
    Calculate hypoglycemia probability based on ECG features.
    This is a demonstration model - in production, this would be the FM-TS transformer.
    
    Risk factors:
    - Low HRV (SDNN < 30) increases risk
    - High HR variability can indicate autonomic response
    - Time of day (higher risk in early morning and post-meal)
    - Recent trend in predictions
    """
    base_prob = 0.15  # Baseline 15% probability
    
    # HRV contribution (lower HRV = higher risk)
    hrv_factor = max(0, (50 - features['hrv_sdnn']) / 100)
    
    # Heart rate contribution
    hr_factor = abs(features['hr_mean'] - 70) / 200
    
    # Time of day factor (higher risk 2-4am and 2-4pm)
    hour = time_of_day.hour
    if 2 <= hour <= 4 or 14 <= hour <= 16:
        time_factor = 0.15
    elif 6 <= hour <= 8:  # Dawn phenomenon
        time_factor = 0.1
    else:
        time_factor = 0
    
    # Add some realistic variability
    noise = np.random.normal(0, 0.05)
    
    # Temporal smoothing based on history
    if history:
        trend = np.mean(history[-5:]) if len(history) >= 5 else np.mean(history)
        smoothing = 0.3 * trend
    else:
        smoothing = 0
    
    # Calculate final probability
    prob = base_prob + hrv_factor + hr_factor + time_factor + noise + smoothing
    
    # Clamp between 0 and 1
    return max(0.05, min(0.95, prob))


def get_risk_level(probability):
    """Determine risk level from probability"""
    if probability < 0.25:
        return "LOW", "risk-low", "✓ Low Risk"
    elif probability < 0.50:
        return "MEDIUM", "risk-medium", "⚠ Moderate Risk"
    else:
        return "HIGH", "risk-high", "⚠ High Risk - Consider Action"


def create_forecast_chart(data, current_time):
    """Create the real-time forecast chart"""
    fig = go.Figure()
    
    # Add risk zones as background
    fig.add_hrect(y0=0.5, y1=1.0, fillcolor="rgba(220, 38, 38, 0.1)", 
                  layer="below", line_width=0, annotation_text="High Risk Zone",
                  annotation_position="top right")
    fig.add_hrect(y0=0.25, y1=0.5, fillcolor="rgba(245, 158, 11, 0.1)", 
                  layer="below", line_width=0)
    fig.add_hrect(y0=0.0, y1=0.25, fillcolor="rgba(22, 163, 74, 0.1)", 
                  layer="below", line_width=0)
    
    # Main probability line
    fig.add_trace(go.Scatter(
        x=data['time'],
        y=data['probability'],
        mode='lines+markers',
        name='p(HG in FW)',
        line=dict(color='#dc2626', width=3),
        marker=dict(size=6, color='#dc2626'),
        hovertemplate='Time: %{x}<br>Risk: %{y:.1%}<extra></extra>'
    ))
    
    # Add threshold lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="#dc2626", 
                  annotation_text="High Risk Threshold (50%)", annotation_position="right")
    fig.add_hline(y=0.25, line_dash="dash", line_color="#f59e0b",
                  annotation_text="Moderate Risk (25%)", annotation_position="right")
    
    # Current time marker
    if len(data) > 0:
        fig.add_vline(x=data['time'].iloc[-1], line_dash="dot", line_color="#2563eb",
                      annotation_text="Now", annotation_position="top")
    
    fig.update_layout(
        title=dict(
            text="<b>p(HG in FW) - Hypoglycemia Probability Forecast</b>",
            font=dict(size=18, color='#1e293b')
        ),
        xaxis_title="Time",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif", color='#1e293b'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=80, b=60),
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
    
    return fig

# filepath: /Users/jenniferdanielonwuchekwa/code/Dati94/hypopredict-frontend/app/app.py
def show_welcome_page():
    st.markdown("""
    <div class="app-header">
        <h1 style="margin-bottom:0;">
        🫀 HypoPredict
        <span style="font-size:0.4em; font-weight:600; color:white;">
            &nbsp;<em>Estimate the risk of hypoglycemia</em>
        </span>
        </h1>
        <p style="margin-top:4px;">
        <em>
           We estimate the risk of dangerously low blood sugar only from your heart data.
        </em>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("#### Welcome to HypoPredict Demo")
        st.markdown("""
        This demonstration showcases our **real-time non-invasive hypoglycemia prediction system** 
        using ECG-derived features. The system processes continuous ECG data to forecast 
        hypoglycemia risk over multiple time horizons.
        """)
        
        name = st.text_input("Enter your name", placeholder="Dr. Smith")
        
        if st.button("Start Monitoring Session", type="primary"):
            if name:
                st.session_state.user_name = name
                st.session_state.page = 'load_data'
                st.rerun()
            else:
                st.error("Please enter your name to continue")
                
# filepath: /Users/jenniferdanielonwuchekwa/code/Dati94/hypopredict-frontend/app/app.py
def show_load_data_page():
    st.markdown("""
    <div class="app-header">
        <h1>📊 Load Your Data</h1>
        <p>Connect your ECG data source for real-time monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"### Welcome, {st.session_state.user_name}!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Option 1: Paste Data URL")
        data_url = st.text_input("ECG Data URL", placeholder="https://example.com/ecg-stream")
        if st.button("Connect to Data Source", type="primary"):
            if data_url:
                st.session_state.data_source = data_url
                st.session_state.page = 'forecast'
                st.session_state.monitoring_start = datetime.now()
                st.rerun()
            else:
                st.warning("Please enter a data URL")
    
    with col2:
        st.markdown("#### Option 2: Use Demo Data")
        if st.button("Start Demo Mode", type="secondary"):
            st.session_state.page = 'select_person_day'
            st.rerun()
            
            
def show_select_person_day_page():
    st.markdown("""
    <div class="app-header">
        <h1>👤 Select Person and Day</h1>
        <p>Choose a person and day to load demo data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dropdowns for selecting person and day
    person = st.selectbox("Select person", options=['Person ' + str(i) for i in range(1, 10)])
    day = st.selectbox("Select day", options=['Day ' + str(i) for i in range(1, 7)])
    selection = (person, day)
    
    # Run Prediction button
    if st.button("Run Prediction"):
        # Validate the selected person and day
        if selection not in DATA_OPTIONS:
            st.warning(
                f"ℹ️ Demo data is not available yet for {person}, {day}.\n\n"
                "Please select a supported combination."
            )
            st.stop()

        # Fetch the data URL for the selected person and day
        data_url = DATA_OPTIONS[selection]

        # Call the API to fetch predictions
        with st.spinner("Predicting..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"url": data_url},
                    timeout=120
                )
                response.raise_for_status()  # Raise an error for non-200 responses
            except requests.exceptions.RequestException as e:
                st.error(f"API error: {e}")
                st.stop()

        # Parse the API response
        data = response.json()
        if "predictions" not in data:
            st.error("API response does not contain 'predictions'")
            st.write(data)
            st.stop()

        # Normalize predictions to a list of floats
        raw_preds = data["predictions"]
        if isinstance(raw_preds[0], list):
            predictions = [p[-1] for p in raw_preds]
        else:
            predictions = raw_preds

        # Calculate the maximum risk
        max_risk = max(predictions)
        max_risk_index = predictions.index(max_risk)
        risk_percent = int(max_risk * 100)

        # Display the maximum risk message
        if max_risk < 0.3:
            st.success("🟢 Low hypoglycemia risk detected.")
        elif max_risk < 0.6:
            st.warning("🟡 Moderate hypoglycemia risk — monitor closely.")
        else:
            st.error("🔴 High hypoglycemia risk — intervention recommended!")

        st.metric(
            label="🩸 Max predicted hypoglycemia risk",
            value=f"{risk_percent}%"
        )

        # Plot the predictions using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(predictions))),
            y=predictions,
            mode="lines",
            name="Hypoglycemia Risk",
            line=dict(color="blue", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[max_risk_index],
            y=[max_risk],
            mode="markers",
            name=f"Max Risk: {max_risk:.2f}",
            marker=dict(color="red", size=10)
        ))
        fig.add_hline(y=0.3, line_dash="dash", line_color="green", name="Low Risk Threshold")
        fig.add_hline(y=0.6, line_dash="dash", line_color="orange", name="Moderate Risk Threshold")
        fig.update_layout(
            title="Predicted Hypoglycemia Risk Over Time",
            xaxis_title="Time Step",
            yaxis_title="Hypoglycemia Risk",
            yaxis=dict(range=[0, 1]),
            legend=dict(font=dict(size=10)),
            template="plotly_white"
        )
        st.plotly_chart(fig)

        # Log the response for debugging
        logging.basicConfig(level=logging.INFO)
        logging.info(response.json())

# filepath: /Users/jenniferdanielonwuchekwa/code/Dati94/hypopredict-frontend/app/app.py
def show_forecast_page():
    st.markdown("""
    <div class="app-header">
        <h1>📈 Real-Time Forecast</h1>
        <p>Hypoglycemia Risk Monitoring Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display selected person and day for Demo Data
    if st.session_state.data_source == 'demo':
        person, day = st.session_state.selection
        st.markdown(f"### Demo Data: {person}, {day}")
    
    # Generate simulated ECG features
    features = generate_simulated_ecg_features()
    current_prob = calculate_hypoglycemia_probability(features, datetime.now(), st.session_state.prediction_history)
    st.session_state.prediction_history.append(current_prob)

    # Display risk level
    risk_level, risk_class, risk_text = get_risk_level(current_prob)
    st.markdown(f"<div class='risk-alert {risk_class}'>{risk_text}</div>", unsafe_allow_html=True)

    # Display forecast chart
    forecast_data = pd.DataFrame({
        'time': [datetime.now() - timedelta(minutes=i) for i in range(len(st.session_state.prediction_history))],
        'probability': st.session_state.prediction_history
    })
    fig = create_forecast_chart(forecast_data, datetime.now())
    st.plotly_chart(fig, use_container_width=True)
    
    
    
# =====================================================
# CONFIG
# =====================================================
API_URL = "https://hypopredict-678277177269.europe-west1.run.app/predict_from_url"

# Hidden mapping: what the user selects -> actual data URL
#DATA_OPTIONS = {
#    "Person 8 – Day 3": (
#        "https://drive.google.com/file/d/"
#        "1rGpElJXOn7-gUVIKGGTlnSWoqWfbqNTB/view?usp=share_link"
#    )
#}
DATA_OPTIONS = {
    ("Person 8", "Day 3"): (
        "https://drive.google.com/file/d/"
        "1rGpElJXOn7-gUVIKGGTlnSWoqWfbqNTB/view?usp=share_link"
    ),
    # Later add:
    ("Person 6", "Day 4"): ("https://drive.google.com/file/d/XXXX/view")
}
# =====================================================
# PAGE SETUP
# =====================================================
#st.set_page_config(
#    page_title="Welcome to HypoPredict ",
#    layout="centered"
#)

# filepath: /Users/jenniferdanielonwuchekwa/code/Dati94/hypopredict-frontend/app/app.py
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

if st.session_state.page == 'welcome':
    show_welcome_page()
elif st.session_state.page == 'load_data':
    show_load_data_page()
elif st.session_state.page == 'select_person_day':
    show_select_person_day_page()  # Add this line
elif st.session_state.page == 'forecast':
    show_forecast_page()




#st.caption("_Estimate the risk of hypoglycemia_")
#st.markdown(
#st.title("🧠 HypoPredict")
#    "*We estimate the risk of dangerously low blood sugar using heart (ECG) data and machine learning.*"
#)


# =====================================================
# USER INPUT (NO URL SHOWN)
# =====================================================
#selection = st.selectbox(
#    "Select person and day",
#    options=list(DATA_OPTIONS.keys())
#)
#person = st.selectbox("Select person", options=['Person ' + str(i) for i in range(1, 10)])
#day = st.selectbox("Select day", options=['Day ' + str(i) for i in range(1, 7)])

#selection = (person, day)

# =====================================================
# CACHED FUNCTION
# =====================================================
@st.cache_data
def fetch_predictions(data_url):
    response = requests.post(API_URL, json={"url": data_url}, timeout=120)
    return response.json()["predictions"]
# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("🧪 Developed by the HypoPredict Team")
