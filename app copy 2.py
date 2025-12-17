import streamlit as st
import requests
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
API_URL = "https://hypopredict-678277177269.europe-west1.run.app/predict_from_url"

TEST_DATA_URL = (
    "https://drive.google.com/file/d/"
    "1rGpElJXOn7-gUVIKGGTlnSWoqWfbqNTB/view?usp=share_link"
)

st.set_page_config(page_title="HypoPredict Demo", layout="centered")

# ----------------------------
# UI
# ----------------------------
st.title("HypoPredict – Demo Website")
st.write("Test frontend for hypoglycemia prediction API")

st.markdown("### Test ECG data")
st.code(TEST_DATA_URL)

if st.button("Run prediction"):
    with st.spinner("Calling prediction API..."):
        response = requests.post(
            API_URL,
            json={"url": TEST_DATA_URL},
            timeout=120
        )

    if response.status_code != 200:
        st.error(f"API Error: {response.status_code}")
        st.text(response.text)
        st.stop()

    data = response.json()

    # ----------------------------
    # ⚠️ ADJUST THIS IF NEEDED
    # ----------------------------
    # Change 'predictions' if your API uses a different key
    predictions = data["predictions"]

    # ----------------------------
    # PLOT
    # ----------------------------
    fig, ax = plt.subplots()
    ax.plot(predictions, color="red", linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Hypoglycemia Risk")
    ax.set_title("Predicted Hypoglycemia Risk")

    st.pyplot(fig)

    # ----------------------------
    # MESSAGE
    # ----------------------------
    max_risk = max(predictions)

    if max_risk < 0.3:
        st.success("Low hypoglycemia risk detected.")
    elif max_risk < 0.6:
        st.warning("Moderate hypoglycemia risk detected.")
    else:
        st.error("High hypoglycemia risk detected!")
