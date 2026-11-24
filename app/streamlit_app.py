
import requests
import streamlit as st
import numpy as np
import cv2
import base64
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="RetinoGrade AI",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ RetinoGrade AI")
st.subheader("Diabetic Retinopathy Detection")
st.write("Upload a retinal fundus image to assess its quality and detect diabetic retinopathy.")

st.warning(
    "Research prototype for decision support only. "
    "Results must be reviewed by a qualified eye-care professional."
)

uploaded_file = st.file_uploader(
    "Upload a retinal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded retinal image", use_container_width=True)

    with col2:
        if st.button("Analyze Image", type="primary", use_container_width=True):

            with st.spinner("Analyzing retinal image..."):

                try:
                    response = requests.post(
                        API_URL,
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type
                            )
                        },
                        timeout=180
                    )

                    response.raise_for_status()
                    result = response.json()

                    st.session_state["result"] = result

                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to the API: {e}")

if "result" in st.session_state:

    result = st.session_state["result"]
    quality = result["quality"]

    st.divider()
    st.header("Analysis Results")

    if quality["class"] == "Good":
        st.success(f"✅ {quality['message']}")

    elif quality["class"] == "Usable":
        st.warning(f"⚠️ {quality['message']}")

    else:
        st.error(f"❌ {quality['message']}")

    st.metric("Image Quality", quality["class"])

    if quality["class"] != "Reject" and result["dr"] is not None:

        dr = result["dr"]

        st.subheader("Diabetic Retinopathy Prediction")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Predicted Grade", dr["predicted_grade"])

        with c2:
            st.metric("Confidence", f"{dr['confidence']:.2f}%")

        st.subheader("Class Probabilities")

        for grade, probability in dr["probabilities"].items():
            st.write(f"**{grade}** — {probability:.2f}%")
            st.progress(probability / 100)

        st.divider()
        st.subheader("Grad-CAM Visualization")

        gradcam_data = result["gradcam"]["image"]
        image_data = base64.b64decode(gradcam_data.split(",", 1)[1])
        heatmap_image = Image.open(io.BytesIO(image_data))

        st.image(
            heatmap_image,
            caption="Grad-CAM: Regions influencing the model's prediction",
            use_container_width=True
        )

    else:
        st.info("No DR grade or Grad-CAM is generated because the image quality was rejected.")
