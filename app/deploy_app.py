import streamlit as st
import requests
import matplotlib.pyplot as plt

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
            f"No demo data available for Person {person}, Day {day}.\n\n"
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

    # =================================================
    # PLOT
    # =================================================
    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    ax.plot(predictions, linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Hypoglycemia Risk")
    ax.set_title("Predicted Hypoglycemia Risk")
    st.pyplot(fig)

    # =================================================
    # MESSAGE
    # =================================================
    max_risk = max(predictions)

    if max_risk < 0.3:
        st.success("Low hypoglycemia risk detected.")
    elif max_risk < 0.6:
        st.warning("Moderate hypoglycemia risk detected.")
    else:
        st.error("High hypoglycemia risk detected!")
