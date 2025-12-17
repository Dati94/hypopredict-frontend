import streamlit as st
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import logging

# =====================================================
# CONFIG
# =====================================================
API_URL = "https://hypopredict-678277177269.europe-west1.run.app/predict_from_url"

DATA_OPTIONS = {
    ('Person 8', 'Day 3'): (
        "https://drive.google.com/file/d/"
        "1rGpElJXOn7-gUVIKGGTlnSWoqWfbqNTB/view?usp=share_link"
    )
    # Later add:
    # ('Person 6', 'Day 4'): "https://drive.google.com/file/d/XXXX/view"
}

# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(
    page_title="Welcome to HypoPredict",
    layout="centered"
)

st.markdown("""
# Welcome to HypoPredict
Hypoglycemia risk prediction – we can estimate the risk of dangerously low blood sugar using heart data.
""")

# =====================================================
# USER INPUT
# =====================================================
selection = st.selectbox(
    "Select person and day",
    options=list(DATA_OPTIONS.keys()),
    index=0  # Default to the first option
)

# =====================================================
# RUN PREDICTION
# =====================================================
if st.button("Run prediction"):
    data_url = DATA_OPTIONS[selection]

    with st.spinner("Predicting..."):
        try:
            response = requests.post(API_URL, json={"url": data_url}, timeout=120)
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {e}")
            st.stop()

        if response.status_code != 200:
            st.error(f"API error: {response.status_code}")
            st.text(response.text)
            st.stop()

        # =================================================
        # PARSE RESPONSE
        # =================================================
        data = response.json()

        if "predictions" not in data:
            st.error("API response does not contain 'predictions'")
            st.write(data)
            st.stop()

        raw_preds = data["predictions"]

        # Normalize predictions to List[float]
        if isinstance(raw_preds[0], list):
            predictions = [p[-1] for p in raw_preds]
        else:
            predictions = raw_preds

        if not predictions:
            st.error("No predictions returned by the API.")
            st.stop()

        # =================================================
        # MESSAGE
        # =================================================
        max_risk = max(predictions)
        max_risk_index = predictions.index(max_risk)  # Find the index of the maximum risk

        if max_risk < 0.3:
            st.success("Low hypoglycemia risk detected.")
        elif max_risk < 0.6:
            st.warning("Moderate hypoglycemia risk detected.")
        else:
            st.error("High hypoglycemia risk detected!")

        st.write(f"Maximum risk detected: {max_risk:.2f}")

        # =================================================
        # PLOT (MATPLOTLIB)
        # =================================================
        fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size for better readability
        ax.plot(predictions, color="blue", linewidth=2, label="Hypoglycemia Risk")
        ax.scatter(max_risk_index, max_risk, color="red", label=f"Max Risk: {max_risk:.2f}")
        ax.axhline(0.3, color="green", linestyle="--", label="Low Risk Threshold")
        ax.axhline(0.6, color="orange", linestyle="--", label="Moderate Risk Threshold")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Hypoglycemia Risk", fontsize=12)
        ax.set_title("Predicted Hypoglycemia Risk Over Time", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="upper right", fontsize=10)
        st.pyplot(fig)

        # =================================================
        # PLOT (PLOTLY)
        # =================================================
        fig = go.Figure()

        # Add predictions line
        fig.add_trace(go.Scatter(
            x=list(range(len(predictions))),
            y=predictions,
            mode="lines",
            name="Hypoglycemia Risk",
            line=dict(color="blue", width=2)
        ))

        # Highlight maximum risk point
        fig.add_trace(go.Scatter(
            x=[max_risk_index],
            y=[max_risk],
            mode="markers",
            name=f"Max Risk: {max_risk:.2f}",
            marker=dict(color="red", size=10)
        ))

        # Add risk thresholds
        fig.add_hline(y=0.3, line_dash="dash", line_color="green", name="Low Risk Threshold")
        fig.add_hline(y=0.6, line_dash="dash", line_color="orange", name="Moderate Risk Threshold")

        # Customize layout
        fig.update_layout(
            title="Predicted Hypoglycemia Risk Over Time",
            xaxis_title="Time Step",
            yaxis_title="Hypoglycemia Risk",
            yaxis=dict(range=[0, 1]),
            legend=dict(font=dict(size=10)),
            template="plotly_white"
        )

        # Display the plot
        st.plotly_chart(fig)

        # Logging for debugging
        logging.basicConfig(level=logging.INFO)
        logging.info(response.json())

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
st.markdown("Developed by HypoPredict Team!")