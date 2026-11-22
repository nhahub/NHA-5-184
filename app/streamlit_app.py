"""
RetinoGrade demo app: upload a fundus photo -> quality check -> grade, probabilities, flags, heatmap.

    streamlit run app/streamlit_app.py      (listens on localhost only: .streamlit/config.toml)
    RG_MODEL           path of the v2 weights        (default models/B_full_s42_384_mix_model.keras)
    RG_QUALITY_MODEL   path of the quality weights   (default models/quality_b0_224.keras)

CPU only. Decision support for screening research, not a diagnostic device.
"""
import os
import sys

import numpy as np
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import retinograde_inference as R  # noqa: E402
import review  # noqa: E402
from explain import gradcam as G  # noqa: E402
from quality import quality as Q  # noqa: E402

st.set_page_config(page_title="RetinoGrade", page_icon="👁", layout="wide")

MAX_MB = 25
GRADE_COLORS = ["#2e7d32", "#9e9d24", "#ef6c00", "#d84315", "#b71c1c"]


# ------------------------------------------------------------------ model (loaded once per server)
@st.cache_resource(show_spinner="Loading the models (a few seconds, once)...")
def load_pipeline():
    from pipeline import RetinoPipeline
    # RG_APP_TEST_NO_GATE is only for the headless tests: it switches the quality gate off
    return RetinoPipeline(model_path=None, quality_model_path=None, use_gate=not os.environ.get("RG_APP_TEST_NO_GATE"))


@st.cache_data(show_spinner=False, max_entries=32)
def decode(data: bytes) -> np.ndarray:
    """bytes -> RGB uint8, or ValueError with a message for the user."""
    return R.load_rgb(data)


@st.cache_data(show_spinner=False, max_entries=64)
def analyse(data: bytes) -> dict:
    return load_pipeline().predict(decode(data))


@st.cache_data(show_spinner=False, max_entries=64)
def heatmap(data: bytes, target: str) -> tuple:
    """(prepared image 0..255, heatmap) for the chosen target, at the model's size."""
    pipe = load_pipeline()
    x = pipe.grader.prepare(decode(data))
    h, _, used = G.gradcam(pipe.grader.model, x, target)
    return x, h, used


# ------------------------------------------------------------------ page
st.title("RetinoGrade - diabetic retinopathy grading")
st.caption("Upload one retinal (fundus) photo. " + R.DISCLAIMER)

with st.sidebar:
    st.header("About")
    st.write("One model grades the photo on the 0-4 scale. Two flags mark eyes that should be referred. "
             "A heatmap shows where the model looked.")
    st.markdown("**Grades:** 0 No DR · 1 Mild · 2 Moderate · 3 Severe · 4 Proliferative")
    st.warning("This is a research prototype for decision support. It must not be used to diagnose or treat patients. "
               "Every result needs a qualified eye-care professional.")
    st.caption(f"Model {R.MODEL_NAME} · review rule: {review.STATUS}")

def uploaded_bytes():
    """Bytes of the uploaded photo. With RG_APP_TEST=1 the tests place them in st.session_state["test_bytes"]
    (Streamlit's test harness cannot press the upload button)."""
    if os.environ.get("RG_APP_TEST") and "test_bytes" in st.session_state:
        return st.session_state["test_bytes"]
    up = st.file_uploader("Fundus photo (PNG or JPG)", type=["png", "jpg", "jpeg"])
    return None if up is None else up.getvalue()


data = uploaded_bytes()
if data is None:
    st.info("Waiting for a photo.")
    st.stop()
if len(data) > MAX_MB * 1024 * 1024:
    st.error(f"The file is larger than {MAX_MB} MB.")
    st.stop()
try:
    img = decode(data)
except ValueError as e:
    st.error(f"This file could not be read as an image. {e}")
    st.stop()
if min(img.shape[:2]) < Q.MIN_SIDE:
    st.error(f"The image is too small ({img.shape[1]}×{img.shape[0]} px). Please upload the original photo.")
    st.stop()

try:
    with st.spinner("Analysing the photo..."):
        result = analyse(data)
except FileNotFoundError as e:
    st.error(f"A model file is missing. {e}")
    st.stop()

quality = result["quality"]
left, right = st.columns([1, 1])
with left:
    st.image(img, caption=f"Uploaded photo ({img.shape[1]}×{img.shape[0]})", use_container_width=True)

with right:
    if quality is None:
        st.warning("The quality model is not available. Image quality was NOT checked.")
    elif quality["action"] == "retake":
        st.error(f"❌ {quality['message']}")
        st.markdown("No grade is shown for a photo that cannot be judged reliably.")
        with st.expander("Quality details"):
            st.json({k: v for k, v in quality.items() if k != "message"})
        st.stop()
    elif quality["action"] == "warn":
        st.warning(f"⚠️ {quality['message']}")
    else:
        st.success(f"✅ {quality['message']}")

    g = result["grade"]
    a = review.assess(result["probabilities"], quality["label"] if quality else None)
    st.markdown(f"<h2 style='color:{GRADE_COLORS[g]};margin-bottom:0'>Grade {g} - {result['grade_name']}</h2>", unsafe_allow_html=True)
    st.metric("Calibrated confidence", f"{a['confidence_calibrated']:.0%}", help="Temperature-scaled probability of the predicted grade (see docs/components/calibration_review.md).")
    if a["needs_review"]:
        st.error("👤 **Needs human review**\n\n" + "\n".join(f"- {r}" for r in a["reasons"]))
    c1, c2 = st.columns(2)
    c1.metric("Referable DR (grade ≥ 2)", "YES" if result["referable"] else "no", help=f"P(grade ≥ 2) = {result['p_referable']:.2f}, threshold {R.REFERABLE_THRESHOLD:.2f}")
    c2.metric("Sight-threatening DR (grade ≥ 3)", "YES" if result["sight_threatening"] else "no",
              help=f"P(grade ≥ 3) = {result['p_sight_threatening']:.3f}, threshold {R.SIGHT_THREATENING_THRESHOLD:.3f}")
    st.caption("The flags use their own thresholds, so they can be on although the predicted grade is lower.")

st.subheader("Class probabilities")
st.bar_chart({"probability": {n: result["probabilities"][n] for n in R.CLASS_NAMES}}, horizontal=True)

st.subheader("Where did the model look? (Grad-CAM)")
choice = st.radio("Explain", ["Predicted grade", "Referral decision (grade ≥ 2)"], horizontal=True)
with st.spinner("Computing the heatmap..."):
    x, h, used = heatmap(data, "pred" if choice == "Predicted grade" else "referable")
h1, h2 = st.columns(2)
h1.image(np.clip(x, 0, 255).astype(np.uint8), caption="Prepared image the model sees", use_container_width=True)
h2.image(G.overlay(x, h), caption=f"Heatmap: {choice.lower()}", use_container_width=True)
st.caption("The heatmap shows regions that increased the chosen score. Colours are scaled separately for each map: red is the "
           "strongest region of this map, not how serious it is, so red appears even when the answer is \"no\". "
           "It is coarse (12x12 cells) and does not outline lesions.")

with st.expander("All output values"):
    st.json({k: v for k, v in result.items()})

st.divider()
st.markdown(f"⚕️ **{R.DISCLAIMER}**")
