import streamlit as st
import requests
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
API_BASE_URL = "https://hypopredict-678277177269.europe-west1.run.app"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"   # adjust if endpoint name differs

st.set_page_config(
    page_title="HypoPredict",
    layout="centered"
)

# =========================
# SIDEBAR (REFERENCES)
# =========================
st.sidebar.markdown("## 🔍 References")

st.sidebar.markdown(
    "[📘 API Documentation](https://hypopredict-678277177269.europe-west1.run.app/docs#/default/predict_from_url_predict_from_url_post)"
)

st.sidebar.markdown(
    "[📊 Example Prediction Output](https://drive.google.com/file/d/1rGpElJXOn7-gUVIKGGTlnSWoqWfbqNTB/view)"
)

# =========================
# HEADER
# =========================
st.title("HypoPredict")
st.subheader("Hypoglycemia Risk Forecasting from Wearable Data")

st.write(
    """
    HypoPredict estimates the **risk of hypoglycemia** using wearable sensor data.
    Select a **person** and **day** to visualize the predicted risk over time.
    """
)

# =========================
# USER INPUT
# =========================
col1, col2 = st.columns(2)

with col1:
    person_id = st.selectbox(
        "Select Person",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        index=7
    )

with col2:
    day_id = st.selectbox(
        "Select Day",
        options=[1, 2, 3, 4, 5, 6, 7],
        index=2
    )

# =========================
# FETCH PREDICTION
# =========================
if st.button("Get Prediction"):

    with st.spinner("Requesting prediction from API..."):
        response = requests.get(
            PREDICT_ENDPOINT,
            params={
                "person_id": person_id,
                "day_id": day_id
            }
        )

    if response.status_code != 200:
        st.error("Failed to retrieve prediction from the API.")
    else:
        data = response.json()

        # =========================
        # EXPECTED API RESPONSE
        # =========================
        # Example expected structure:
        # {
        #   "time": [0, 10, 20, 30, 40, 50],
        #   "risk": [0.12, 0.25, 0.41, 0.63, 0.78, 0.85]
        # }

        time = data.get("time", [])
        risk = data.get("risk", [])

        if not time or not risk:
            st.error("API response format is invalid.")
        else:
            max_risk = max(risk)

            # =========================
            # RISK MESSAGE
            # =========================
            if max_risk < 0.3:
                st.success("🟢 Low hypoglycemia risk detected.")
            elif max_risk < 0.6:
                st.warning("🟡 Moderate hypoglycemia risk detected.")
            else:
                st.error("🔴 High hypoglycemia risk detected!")

            st.metric(
                label="Maximum Risk Probability",
                value=f"{max_risk * 100:.1f}%"
            )

            # =========================
            # PLOT
            # =========================
            fig, ax = plt.subplots(figsize=(8, 4))

            ax.plot(time, risk, marker="o", linewidth=2)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Time (minutes)")
            ax.set_ylabel("Hypoglycemia Risk")
            ax.set_title(f"Person {person_id} – Day {day_id}")

            ax.axhline(0.3, linestyle="--", color="green", alpha=0.4)
            ax.axhline(0.6, linestyle="--", color="orange", alpha=0.4)

            st.pyplot(fig)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "This application demonstrates an **MVP** for hypoglycemia risk prediction. "
    "Predictions are generated via a machine learning model served through a FastAPI backend."
)
