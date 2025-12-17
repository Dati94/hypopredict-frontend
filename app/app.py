import streamlit as st
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import logging
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
    )
    # Later add:
    # ("Person 6", "Day 4"): "https://drive.google.com/file/d/XXXX/view"
}
# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(
    page_title="Welcome to HypoPredict ",
    layout="centered"
)

st.title("Welcome to HypoPredict")
st.write("Hypoglycemia risk prediction – we can estimate the risk of dangerously low blood blood sugar only by heart data ")

# =====================================================
# USER INPUT (NO URL SHOWN)
# =====================================================
#selection = st.selectbox(
#    "Select person and day",
#    options=list(DATA_OPTIONS.keys())
#)
person = st.selectbox("Select person", options=['Person ' + str(i) for i in range(1, 10)])
day = st.selectbox("Select day", options=['Day ' + str(i) for i in range(1, 7)])

selection = (person, day)
# =====================================================
# RUN PREDICTION
# =====================================================
if st.button("Run prediction"):
    if selection not in DATA_OPTIONS:
        st.warning(
            f"No demo data available for {person}, {day}.\n\n"
            "Please select a supported combination."
        )
        st.stop()

    data_url = DATA_OPTIONS[selection]

    with st.spinner("Predicting..."):
        response = requests.post(
            API_URL,
            json={"url": data_url},
            timeout=120
        )

    if response.status_code != 200:
        st.error("API error")
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
    # Handles:
    # - [0.1, 0.2, 0.3]
    # - [[0.1], [0.2], ...]
    # - [[0.9, 0.1], [0.8, 0.2], ...]
    if isinstance(raw_preds[0], list):
        predictions = [p[-1] for p in raw_preds]
    else:
        predictions = raw_preds

    max_risk = max(predictions)
    risk_percent = int(max_risk * 100)

    st.metric(
        label="Max predicted hypoglycemia risk",
        value=f"{risk_percent}%"
    )


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

        #st.write(f"Maximum risk detected: {max_risk:.2f}")

    # =================================================
    # PLOT(MATPLOTLIB)
    # =================================================

    #plt.style.use("dark_background")
    #fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size for better readability
    #ax.plot(predictions, color="blue", linewidth=2, label="Hypoglycemia Risk")
    #ax.scatter(max_risk_index, max_risk, color="red", label=f"Max Risk: {max_risk:.2f}")
    #ax.axhline(0.3, color="green", linestyle="--", label="Low Risk Threshold")
    #ax.axhline(0.6, color="orange", linestyle="--", label="Moderate Risk Threshold")
    #ax.set_ylim(0, 1)
    #ax.set_xlabel("Time Step", fontsize=12)
    #ax.set_ylabel("Hypoglycemia Risk", fontsize=12)
    #ax.set_title("Predicted Hypoglycemia Risk Over Time", fontsize=14)
    #ax.grid(True, linestyle="--", alpha=0.7)
    #ax.legend(loc="upper right", fontsize=10)
    #st.pyplot(fig)
#
    #st.pyplot(fig)

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
st.markdown("Developed by HypoPredict Team")
