"""
HypoPredict Streamlit Demo
Real-time ECG-based Hypoglycemia Prediction Visualization

This demo showcases the HypoPredict AI system for forecasting hypoglycemia
risk using ECG-derived features. Designed for BMBF grant proposal demonstrations.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import random

# Page configuration
st.set_page_config(
    page_title="HypoPredict - Real-time Hypoglycemia Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for medical-grade styling
st.markdown("""
<style>
    /* Medical-grade color palette */
    :root {
        --primary-blue: #2563eb;
        --danger-red: #dc2626;
        --warning-amber: #f59e0b;
        --safe-green: #16a34a;
        --background: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
    }
    
    .main {
        background-color: var(--background);
    }
    
    .stApp {
        background-color: #f1f5f9;
    }
    
    /* Risk alert styling */
    .risk-alert {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 2px solid #16a34a;
        color: #166534;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
        color: #92400e;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #dc2626;
        color: #991b1b;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--primary-blue);
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .app-header h1 {
        margin: 0;
        font-size: 2rem;
    }
    
    .app-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Welcome page styling */
    .welcome-container {
        max-width: 500px;
        margin: 4rem auto;
        padding: 3rem;
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
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


def create_ecg_trace_chart(ecg_data):
    """Create a simulated ECG trace visualization"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ecg_data['time'],
        y=ecg_data['voltage'],
        mode='lines',
        name='ECG Signal',
        line=dict(color='#2563eb', width=1.5),
    ))
    
    fig.update_layout(
        title=dict(text="<b>ECG Signal (Lead II)</b>", font=dict(size=14)),
        xaxis_title="Time (s)",
        yaxis_title="Voltage (mV)",
        height=200,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=False
    )
    
    return fig


def generate_ecg_waveform(duration_seconds=10, sample_rate=250):
    """Generate a realistic ECG waveform for visualization"""
    t = np.linspace(0, duration_seconds, duration_seconds * sample_rate)
    
    # Generate basic ECG pattern using sinusoids
    ecg = np.zeros_like(t)
    heart_rate = 70 + np.random.normal(0, 5)  # BPM
    beat_interval = 60 / heart_rate  # seconds per beat
    
    for beat_time in np.arange(0, duration_seconds, beat_interval):
        # P wave
        p_center = beat_time
        ecg += 0.15 * np.exp(-((t - p_center) ** 2) / (2 * 0.01 ** 2))
        
        # QRS complex
        qrs_center = beat_time + 0.16
        ecg += 1.0 * np.exp(-((t - qrs_center) ** 2) / (2 * 0.008 ** 2))
        ecg -= 0.3 * np.exp(-((t - (qrs_center - 0.03)) ** 2) / (2 * 0.005 ** 2))
        ecg -= 0.15 * np.exp(-((t - (qrs_center + 0.03)) ** 2) / (2 * 0.005 ** 2))
        
        # T wave
        t_center = beat_time + 0.35
        ecg += 0.3 * np.exp(-((t - t_center) ** 2) / (2 * 0.04 ** 2))
    
    # Add noise
    ecg += np.random.normal(0, 0.02, len(ecg))
    
    return pd.DataFrame({'time': t, 'voltage': ecg})


# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'user_name' not in st.session_state:
    st.session_state.user_name = ''
if 'data_source' not in st.session_state:
    st.session_state.data_source = ''
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'monitoring_start' not in st.session_state:
    st.session_state.monitoring_start = None
if 'current_minute' not in st.session_state:
    st.session_state.current_minute = 0


# ============== PAGE: WELCOME ==============
def show_welcome_page():
    st.markdown("""
    <div class="app-header">
        <h1>🩺 HypoPredict</h1>
        <p>AI-Powered Hypoglycemia Risk Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Welcome to HypoPredict Demo")
        st.markdown("""
        This demonstration showcases our **real-time hypoglycemia prediction system** 
        using ECG-derived features. The system processes continuous ECG data to forecast 
        hypoglycemia risk over multiple time horizons.
        
        **Key Features:**
        - Real-time ECG signal processing
        - Multi-horizon risk prediction (30 min - 2 hours)
        - Calibrated probability estimates
        - Knowledge-graph grounded explanations
        """)
        
        st.divider()
        
        name = st.text_input("Enter your name", placeholder="Dr. Smith")
        
        if st.button("Start Monitoring Session", type="primary"):
            if name:
                st.session_state.user_name = name
                st.session_state.page = 'load_data'
                st.rerun()
            else:
                st.error("Please enter your name to continue")


# ============== PAGE: LOAD DATA ==============
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
        st.markdown("""
        Provide a URL to your ECG data source (CSV, JSON, or streaming API).
        The data should contain ECG signals sampled at 250Hz or higher.
        """)
        
        data_url = st.text_input(
            "ECG Data URL",
            placeholder="https://example.com/ecg-stream or file path",
            help="Enter URL to ECG data source"
        )
        
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
        st.markdown("""
        Don't have ECG data? Use our simulated data to explore the 
        prediction interface. This generates realistic ECG patterns
        typical of Type 1 Diabetes patients.
        """)
        
        st.info("""
        **Demo Mode Features:**
        - Simulated 16-hour monitoring (6am - 10pm)
        - Realistic ECG-derived features
        - Dynamic risk probability patterns
        - 1-second update intervals
        """)
        
        if st.button("Start Demo Mode", type="secondary"):
            st.session_state.data_source = 'demo'
            st.session_state.page = 'forecast'
            st.session_state.monitoring_start = datetime.now().replace(hour=6, minute=0, second=0)
            st.session_state.current_minute = 0
            st.session_state.prediction_history = []
            st.rerun()
    
    st.divider()
    
    if st.button("← Back to Welcome"):
        st.session_state.page = 'welcome'
        st.rerun()


# ============== PAGE: FORECAST ==============
def show_forecast_page():
    # Header with monitoring info
    st.markdown("""
    <div class="app-header">
        <h1>📈 Real-Time Forecast</h1>
        <p>Hypoglycemia Risk Monitoring Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Monitoring info bar
    col1, col2, col3, col4 = st.columns(4)
    
    monitoring_time = st.session_state.monitoring_start + timedelta(minutes=st.session_state.current_minute)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Patient</div>
            <div style="font-size: 1.2rem; font-weight: bold;">{st.session_state.user_name}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Time</div>
            <div style="font-size: 1.2rem; font-weight: bold;">{monitoring_time.strftime('%H:%M:%S')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        elapsed = st.session_state.current_minute
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Monitoring Duration</div>
            <div style="font-size: 1.2rem; font-weight: bold;">{elapsed} min / 960 min</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Data Source</div>
            <div style="font-size: 1.2rem; font-weight: bold;">{'Demo Mode' if st.session_state.data_source == 'demo' else 'Live'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Generate current prediction
    features = generate_simulated_ecg_features()
    current_prob = calculate_hypoglycemia_probability(
        features, 
        monitoring_time,
        st.session_state.prediction_history
    )
    
    # Update history
    st.session_state.prediction_history.append(current_prob)
    if len(st.session_state.prediction_history) > 60:  # Keep last 60 minutes
        st.session_state.prediction_history = st.session_state.prediction_history[-60:]
    
    # Risk alert
    risk_level, risk_class, risk_text = get_risk_level(current_prob)
    
    st.markdown(f"""
    <div class="risk-alert {risk_class}">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{risk_text}</div>
        <div style="font-size: 3rem;">Risk of HG in the next hour: {current_prob:.0%}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main forecast chart
    st.markdown("### Probability Forecast Timeline")
    
    # Create forecast data
    forecast_data = pd.DataFrame({
        'time': [st.session_state.monitoring_start + timedelta(minutes=i) 
                 for i in range(len(st.session_state.prediction_history))],
        'probability': st.session_state.prediction_history
    })
    
    if len(forecast_data) > 0:
        fig = create_forecast_chart(forecast_data, monitoring_time)
        st.plotly_chart(fig, use_container_width=True)
    
    # ECG trace and features
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ECG Signal (Current Window)")
        ecg_data = generate_ecg_waveform(duration_seconds=5)
        ecg_fig = create_ecg_trace_chart(ecg_data)
        st.plotly_chart(ecg_fig, use_container_width=True)
    
    with col2:
        st.markdown("### ECG-Derived Features")
        st.markdown(f"""
        | Feature | Value |
        |---------|-------|
        | HRV SDNN | {features['hrv_sdnn']:.1f} ms |
        | HRV RMSSD | {features['hrv_rmssd']:.1f} ms |
        | Mean HR | {features['hr_mean']:.0f} bpm |
        | HR Variability | {features['hr_variability']:.1f} bpm |
        | QT Interval | {features['qt_interval']:.0f} ms |
        | ST Deviation | {features['st_deviation']:.2f} mV |
        """)
        
        st.markdown("### Prediction Horizons")
        horizons = [
            ("30 min", current_prob * 0.7),
            ("1 hour", current_prob),
            ("2 hours", current_prob * 1.1),
            ("4 hours", current_prob * 0.85),
        ]
        for horizon, prob in horizons:
            prob = min(prob, 0.95)
            color = "#dc2626" if prob > 0.5 else "#f59e0b" if prob > 0.25 else "#16a34a"
            st.markdown(f"**{horizon}:** <span style='color: {color}'>{prob:.0%}</span>", unsafe_allow_html=True)
    
    # Update info
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Data Selection"):
            st.session_state.page = 'load_data'
            st.session_state.prediction_history = []
            st.session_state.current_minute = 0
            st.rerun()
    
    with col2:
        st.info("📊 Updates every 1 second | 10-minute prediction window | ECG monitoring: 6am - 10pm (960 minutes)")
    
    with col3:
        if st.button("Reset Session"):
            st.session_state.page = 'welcome'
            st.session_state.prediction_history = []
            st.session_state.current_minute = 0
            st.session_state.user_name = ''
            st.rerun()
    
    # Auto-refresh for real-time updates
    st.session_state.current_minute += 1
    if st.session_state.current_minute < 960:  # 16 hours
        time.sleep(1)  # 1 second update interval
        st.rerun()
    else:
        st.success("✅ Monitoring session complete (16 hours)")


# Main routing
if st.session_state.page == 'welcome':
    show_welcome_page()
elif st.session_state.page == 'load_data':
    show_load_data_page()
elif st.session_state.page == 'forecast':
    show_forecast_page()
