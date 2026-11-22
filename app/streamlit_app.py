"""
RetinoGrade demo app: upload a fundus photo -> quality check -> grade, probabilities, flags, heatmap.

    streamlit run app/streamlit_app.py      (listens on localhost only: .streamlit/config.toml)
    RG_MODEL           path of the v2 weights        (default models/B_full_s42_384_mix_model.keras)
    RG_QUALITY_MODEL   path of the quality weights   (default models/quality_b0_224.keras)
    RG_ALLOW_NETWORK   set to 1 only for a deliberate deployment: other devices may then open the page

CPU only. Decision support for screening research, not a diagnostic device.
"""
import io
import logging
import os
import sys

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import retinograde_inference as R  # noqa: E402
import review  # noqa: E402
from explain import gradcam as G  # noqa: E402
from quality import quality as Q  # noqa: E402

st.set_page_config(page_title="RetinoGrade", page_icon="👁", layout="wide")

MAX_MB = 25
MAX_MEGAPIXELS = 60                                     # fundus cameras give about 1-20; a tiny file can hide a huge image
SHOWN_SIDE = 1000                                       # the uploaded photo is shrunk to this for display only
GRADE_COLORS = ["#388e3c", "#8d8b1e", "#ef6c00", "#e64a19", "#d32f2f"]      # readable on the dark and the light theme
CACHE = {"show_spinner": False, "ttl": 600}             # small caches that expire: photos should not pile up in memory


# ------------------------------------------------------------------ model (loaded once per server)
@st.cache_resource(show_spinner="Loading the models (a few seconds, once)...")
def load_pipeline():
    from pipeline import RetinoPipeline
    # RG_APP_TEST_NO_GATE is only for the headless tests: it switches the quality gate off
    return RetinoPipeline(model_path=None, quality_model_path=None, use_gate=not os.environ.get("RG_APP_TEST_NO_GATE"))


@st.cache_data(max_entries=2, **CACHE)
def decode(data: bytes) -> np.ndarray:
    """bytes -> RGB uint8, or ValueError with a message for the user."""
    try:
        w, h = Image.open(io.BytesIO(data)).size        # reads only the header
    except Image.DecompressionBombError:
        raise ValueError(f"It has too many pixels; the limit is {MAX_MEGAPIXELS} megapixels.") from None
    except Exception:
        raise ValueError("The file is not a valid image") from None
    if w * h > MAX_MEGAPIXELS * 1e6:
        raise ValueError(f"It has too many pixels ({w}×{h}); the limit is {MAX_MEGAPIXELS} megapixels.")
    return R.load_rgb(data)


def percent(p: float) -> str:
    """Never "100%": the model is never certain."""
    return ">99%" if p >= 0.995 else f"{p:.0%}"


def probability_chart(probs: dict) -> alt.Chart:
    """Bars in clinical order (by default charts sort the labels alphabetically) on a fixed 0-100% axis."""
    d = pd.DataFrame({"grade": [f"{k} {n}" for k, n in enumerate(R.CLASS_NAMES)], "probability": [probs[n] for n in R.CLASS_NAMES]})
    return alt.Chart(d).mark_bar().encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
        y=alt.Y("grade:N", sort=None, title=None, axis=alt.Axis(labelLimit=200)),
        tooltip=["grade", alt.Tooltip("probability:Q", format=".1%")]).properties(height=200)


def shrink(img: np.ndarray) -> np.ndarray:
    f = SHOWN_SIDE / max(img.shape[:2])
    return img if f >= 1 else cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


@st.cache_data(max_entries=8, **CACHE)
def analyse(data: bytes) -> dict:
    return load_pipeline().predict(decode(data))


@st.cache_data(max_entries=8, **CACHE)
def heatmap(data: bytes, target: str) -> tuple:
    """(prepared image 0..255, heatmap) for the chosen target, at the model's size."""
    pipe = load_pipeline()
    x = pipe.grader.prepare(decode(data))
    h, _, used = G.gradcam(pipe.grader.model, x, target)
    return x, h, used


# ------------------------------------------------------------------ page
st.title("RetinoGrade - diabetic retinopathy grading")
# .streamlit/config.toml is read only when the app is started from the project folder; without it Streamlit listens to the network
if st.get_option("server.address") not in ("localhost", "127.0.0.1", "::1") and not os.environ.get("RG_ALLOW_NETWORK"):
    st.error("Stopped: other devices on the network could open this page. Start the app from the project folder, "
             "or add --server.address localhost to the command.")
    st.stop()
st.caption("Upload one retinal (fundus) photo. " + R.DISCLAIMER)

with st.sidebar:
    st.header("About")
    st.write("One model grades the photo on the 0-4 scale. Two flags mark eyes that should be referred. "
             "A heatmap shows where the model looked.")
    st.markdown("**Grades:** 0 No DR · 1 Mild · 2 Moderate · 3 Severe · 4 Proliferative")
    st.warning("This is a research prototype for decision support. It must not be used to diagnose or treat patients. "
               "Every result needs a qualified eye-care professional.")


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
except Exception:
    logging.exception("analysis failed")                # details go to the terminal, not to the page
    st.error("The photo could not be analysed. Please try another photo; the details are in the terminal.")
    st.stop()

quality = result["quality"]
left, right = st.columns([1, 1])
with left:
    st.image(shrink(img), caption=f"Uploaded photo ({img.shape[1]}×{img.shape[0]})", use_container_width=True)

with right:
    if quality is None:
        st.warning("The quality model is not available. Image quality was NOT checked.")
    elif quality["action"] == "retake":
        st.error(f"❌ {quality['message']}")
        st.markdown("No grade is shown for a photo that cannot be judged reliably.")
        if quality["probabilities"]:
            st.caption("Quality check: " + " · ".join(f"{n} {v:.0%}" for n, v in quality["probabilities"].items()))
        st.stop()
    elif quality["action"] == "warn":
        st.warning(f"⚠️ {quality['message']}")
    else:
        st.success(f"✅ {quality['message']}")

    g = result["grade"]
    a = review.assess(result["probabilities"], quality["label"] if quality else None)
    st.markdown(f"<h2 style='color:{GRADE_COLORS[g]};margin-bottom:0'>Grade {g} - {result['grade_name']}</h2>", unsafe_allow_html=True)
    st.metric("Calibrated confidence", percent(a["confidence_calibrated"]), help="Temperature-scaled probability of the predicted grade (see docs/components/calibration_review.md).")
    if a["needs_review"]:
        st.error("👤 **Needs human review**\n\n" + "\n".join(f"- {r}" for r in a["reasons"]))
    c1, c2 = st.columns(2)
    c1.metric("Referable DR (grade ≥ 2)", "YES" if result["referable"] else "no", help=f"P(grade ≥ 2) = {result['p_referable']:.2f}, threshold {R.REFERABLE_THRESHOLD:.2f}")
    c2.metric("Sight-threatening DR (grade ≥ 3)", "YES" if result["sight_threatening"] else "no",
              help=f"P(grade ≥ 3) = {result['p_sight_threatening']:.3f}, threshold {R.SIGHT_THREATENING_THRESHOLD:.3f}")
    st.caption("The flags use their own thresholds, so they can be on although the predicted grade is lower.")

st.subheader("Class probabilities")
st.altair_chart(probability_chart(result["probabilities"]), use_container_width=True)
st.caption("The bars are the model's raw probabilities. The calibrated confidence above is adjusted, so it can differ from the longest bar.")

st.subheader("Where did the model look? (Grad-CAM)")
choice = st.radio("Explain", ["Referral decision (grade ≥ 2)", "Predicted grade"], horizontal=True,
                  help="The referral map is the default: on IDRiD it pointed at real lesions more often than the predicted-grade map.")
with st.spinner("Computing the heatmap..."):
    x, h, used = heatmap(data, "pred" if choice == "Predicted grade" else "referable")
h1, h2 = st.columns(2)
h1.image(np.clip(x, 0, 255).astype(np.uint8), caption="Prepared image the model sees", use_container_width=True)
h2.image(G.overlay(x, h), caption=f"Heatmap: {choice.lower()}", use_container_width=True)
if h.max() == 0:
    st.info("No region increased this score, so the heatmap is empty.")
st.caption("The heatmap shows regions that increased the chosen score. Colours are scaled separately for each map: red is the "
           "strongest region of this map, not how serious it is, so red appears even when the answer is \"no\". "
           "It is coarse (12x12 cells) and does not outline lesions.")

st.divider()
st.markdown(f"⚕️ **{R.DISCLAIMER}**")
