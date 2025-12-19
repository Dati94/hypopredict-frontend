from datetime import datetime, timedelta
import streamlit as st
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import logging
import numpy as np
import pandas as pd
from pathlib import Path

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


def create_forecast_chart(series: pd.Series) -> go.Figure:
    """Create the real-time forecast chart from a pandas Series (index=datetime, values in [0,1])."""
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
        x=series.index,
        y=series.values,
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

# =====================================================
# API helpers
# =====================================================
BASE_URL = "https://hypopredictjesus-678277177269.europe-west1.run.app"
PERSON_TO_CODE = {"Person 1": 83, "Person 2": 64}
# Map API codes to pre-rendered Plotly HTML files (relative, no leading slash)
PERSON_HTML = {
    83: "plots/8_final.html",  # Person 1
    64: "plots/6_final.html",  # Person 2
}



def _normalize(s: pd.Series) -> pd.Series:
    """Normalize a pandas Series to [0, 1] using min-max scaling."""
    s = s.astype(float)
    min_v, max_v = s.min(), s.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return s.fillna(0.0)
    return (s - min_v) / (max_v - min_v)

@st.cache_data(ttl=600)
def fetch_predictions(code: int):
    """Fetch fusion and cnn predictions, normalize, and create combined series."""
    url = f"{BASE_URL}/predict_fusion_local_{code}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    # Fusion predictions
    pred_f = pd.Series(payload.get('pred_fusion'))
    pred_f.index = pd.to_datetime(pred_f.index, errors='coerce')
    pred_f = _normalize(pred_f.sort_index()).dropna()

    # CNN/LSTM predictions (drop first element per API note)
    pred_cnn = pd.Series(payload.get('pred_cnn'))[1:]
    pred_cnn.index = pd.to_datetime(pred_cnn.index, errors='coerce')
    pred_cnn = _normalize(pred_cnn.sort_index()).dropna()

    # Merge by nearest minute and create combined prediction
    merged = pd.merge_asof(
        pred_cnn.to_frame(name='lstm'),
        pred_f.to_frame(name='fusion'),
        left_index=True,
        right_index=True,
        direction='nearest',
        tolerance=pd.Timedelta('1min')
    ).dropna()
    pred_combined = 0.4 * merged['lstm'] + 0.6 * merged['fusion']
    pred_combined.name = 'combined'

    return {"fusion": pred_f, "cnn": pred_cnn, "combined": pred_combined}

# =====================================================
# PAGE FUNCTIONS
# =====================================================

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
    """New prediction page with API-driven data and combined forecasts."""
    st.markdown("""
    <div class="app-header">
        <h1>👤 Select Person and Prediction Model</h1>
        <p>Choose a person to load real prediction data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Select person
    person = st.selectbox("Select person", options=list(PERSON_TO_CODE.keys()), index=0)
    
    # Select which model to display
    series_choice = st.radio("Select prediction model", ["Fusion", "CNN", "Combined"], index=0, horizontal=True)
    
    # Run Prediction button
    if st.button("Load and Display Predictions", type="primary"):
        code = PERSON_TO_CODE[person]
        
        # Fetch predictions from API
        with st.spinner("Fetching predictions from API..."):
            try:
                preds = fetch_predictions(code)
            except Exception as e:
                st.error(f"Failed to fetch predictions: {e}")
                st.stop()
        
        # Store in session state for display
        st.session_state.current_predictions = preds
        st.session_state.selected_person = person
        st.session_state.selected_series = series_choice.lower()
        st.session_state.page = 'forecast'
        st.rerun()


def _find_person_plot(code: int):
    """Resolve the HTML path for a person's extra plot, trying common names/locations."""
    app_dir = Path(__file__).parent
    # Preferred mapping (strip any accidental leading slash)
    preferred = str(PERSON_HTML.get(code, "")).lstrip("/")
    # Known alternates from your note (8/6 variants)
    alternates = {
        83: ["8_final.html", "8.html"],
        64: ["6_final.html", "6.html"],
    }.get(code, [])
    # Build candidate names (de-dup)
    names = [n for n in [preferred] + alternates if n]
    seen = set()
    names = [n for n in names if not (n in seen or seen.add(n))]
    # Search in app/ and app/plots/
    folders = [app_dir, app_dir / "plots"]
    for folder in folders:
        for name in names:
            p = folder / name
            if p.exists():
                return p
    return None

def show_forecast_page():
    """Display real-time forecast chart with API data."""
    st.markdown("""
    <div class="app-header">
        <h1>📈 Real-Time Forecast</h1>
        <p>Hypoglycemia Risk Monitoring Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display selected person and model
    if hasattr(st.session_state, 'selected_person'):
        st.markdown(f"### Prediction Data: {st.session_state.selected_person} - {st.session_state.selected_series.title()} Model")
    
    # Retrieve predictions from session state
    if 'current_predictions' in st.session_state and 'selected_series' in st.session_state:
        preds = st.session_state.current_predictions
        key = st.session_state.selected_series
        series_data = preds[key]
        
        # Calculate risk metrics
        current_risk = series_data.iloc[-1] if len(series_data) > 0 else 0.0
        max_risk = series_data.max()
        
        # Display risk level
        risk_level, risk_class, risk_text = get_risk_level(current_risk)

        # Define colors based on risk level
        if risk_level == "LOW":
            bg_color = "#dcfce7"  # Light green
            border_color = "#16a34a"  # Safe green
            text_color = "#15803d"  # Dark green
        elif risk_level == "MEDIUM":
            bg_color = "#fef3c7"  # Light amber
            border_color = "#f59e0b"  # Warning amber
            text_color = "#92400e"  # Dark amber
        else:  # HIGH
            bg_color = "#fee2e2"  # Light red
            border_color = "#dc2626"  # Danger red
            text_color = "#991b1b"  # Dark red

        st.markdown(f"<div style='text-align: center; padding: 16px; background-color: {bg_color}; border-radius: 10px; border: 2px solid {border_color};'><h3 style='color: {text_color};'>{risk_text}</h3></div>", unsafe_allow_html=True)
        
        # Display current and max risk
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Risk", f"{current_risk:.1%}")
        with col2:
            st.metric("Max Risk (24h)", f"{max_risk:.1%}")
        
        # Display forecast chart
        fig = create_forecast_chart(series_data)
        st.plotly_chart(fig, use_container_width=True)

        # Additional per-person pre-rendered Plotly chart
        try:
            person_label = st.session_state.selected_person
            code = PERSON_TO_CODE.get(person_label)
            plot_path = _find_person_plot(code)
            if plot_path and plot_path.exists():
                with open(plot_path, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=600)
                st.caption("Explore the relationship between ECG features and hypoglycemia risk.")
            else:
                st.info("No additional plot found. Place 8_final.html/6_final.html in app/plots or app/.")
        except Exception as e:
            st.info(f"Could not load additional plot: {e}")
        
        # Back button
        if st.button("Back to Person Selection"):
            st.session_state.page = 'select_person_day'
            st.rerun()
    else:
        st.warning("No prediction data available. Please select a person and model first.")
        if st.button("Go Back"):
            st.session_state.page = 'select_person_day'
            st.rerun()

# =====================================================
# CONFIG
# =====================================================
API_URL = "https://hypopredict-678277177269.europe-west1.run.app/predict_from_url"

# =====================================================
# PAGE SETUP
# =====================================================
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

if st.session_state.page == 'welcome':
    show_welcome_page()
elif st.session_state.page == 'load_data':
    show_load_data_page()
elif st.session_state.page == 'select_person_day':
    show_select_person_day_page()
elif st.session_state.page == 'forecast':
    show_forecast_page()

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("🧪 Developed by the HypoPredict Team")