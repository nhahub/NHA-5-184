"""Streamlit app smoke tests. Need Streamlit + TensorFlow + the v2 model file (the `.venv-tf` environment);
skipped otherwise. No real photos: synthetic fundus-like images and garbage bytes."""
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from test_inference_v2 import fundus  # noqa: E402

pytest.importorskip("tensorflow")
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import retinograde_inference as R  # noqa: E402

MODEL = os.environ.get(R.MODEL_ENV) or os.path.join(ROOT, R.DEFAULT_MODEL_PATH)
pytestmark = pytest.mark.skipif(not os.path.isfile(MODEL), reason="real v2 model file not available")
APP = os.path.join(ROOT, "app", "streamlit_app.py")


def run(data=None, gate=False):
    """gate=False switches the quality gate off, so the tests do not depend on the gate's opinion of a drawing."""
    os.environ["RG_APP_TEST"] = "1"
    os.environ[R.MODEL_ENV] = MODEL
    os.environ["RG_APP_TEST_NO_GATE"] = "" if gate else "1"
    from streamlit import config
    if not config.get_option("server.address"):                # pytest was started outside the project folder
        config.set_option("server.address", "localhost")
    if run.gate != gate:                                       # the pipeline is cached: reload it when the setting changes
        import streamlit as st
        st.cache_resource.clear()
        st.cache_data.clear()
        run.gate = gate
    at = AppTest.from_file(APP, default_timeout=300)
    if data is not None:
        at.session_state["test_bytes"] = data
    at.run()
    return at


run.gate = None


def png(img):
    return cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()


def text(at):
    return " ".join([e.value for k in ("info", "error", "warning", "success", "markdown", "title", "caption", "subheader")
                     for e in getattr(at, k)])


def test_page_waits_for_a_photo():
    at = run()
    assert not at.exception and any("Waiting" in i.value for i in at.info)


def test_photo_gives_grade_flags_probabilities_and_disclaimer():
    at = run(png(fundus(h=1200, w=1600)))
    assert not at.exception
    t = text(at)
    assert "Grade" in t and R.DISCLAIMER in t
    assert len(at.metric) == 3                                # confidence, referable, sight-threatening
    labels = {m.label for m in at.metric}
    assert {"Calibrated confidence", "Referable DR (grade ≥ 2)", "Sight-threatening DR (grade ≥ 3)"} == labels
    assert "heatmap" in t.lower() or "Grad-CAM" in t
    assert "Image quality" in t or "quality" in t.lower()


def test_heatmap_toggle_does_not_crash():
    at = run(png(fundus(h=900, w=1200, seed=3)))
    assert at.radio[0].value == "Referral decision (grade ≥ 2)"      # the more reliable map is the default
    at.radio[0].set_value("Predicted grade").run()
    assert not at.exception


@pytest.mark.parametrize("data,expected", [
    (b"this is not an image", "could not be read"),
    (b"", "could not be read"),
    (png(np.zeros((8000, 8000, 3), np.uint8)), "too many pixels"),          # 64 megapixels in a tiny file
    (png(np.zeros((16000, 16000, 3), np.uint8)), "too many pixels"),        # Pillow refuses this one by itself
    (png(np.full((40, 500, 3), 80, np.uint8)), "too small"),
])
def test_bad_files_are_explained_not_crashed(data, expected):
    at = run(data)
    assert not at.exception and any(expected in e.value for e in at.error)


def test_huge_file_is_refused(monkeypatch):
    at = run(b"\0" * (26 * 1024 * 1024))
    assert not at.exception and any("larger than" in e.value for e in at.error)


def test_rejected_photo_shows_no_grade():
    """With the real quality gate on, a drawing that the gate rejects must give the retake message and NO grade."""
    from quality import quality as Q
    gate_path = os.environ.get(Q.MODEL_ENV) or os.path.join(ROOT, Q.DEFAULT_MODEL_PATH)
    if not os.path.isfile(gate_path):
        pytest.skip("quality model not available")
    os.environ[Q.MODEL_ENV] = gate_path
    at = run(png(fundus(h=1200, w=1600)), gate=True)
    assert not at.exception
    assert any("too low to grade" in e.value for e in at.error)
    import re
    assert not at.metric and re.search(r"Grade d -", text(at)) is None    # the sidebar only lists the grade names


def test_page_shows_no_technical_json_and_no_internal_status():
    at = run(png(fundus(h=1200, w=1600)))
    assert not at.json and not at.expander
    t = text(at)
    assert "review rule:" not in t and "not approved" not in t


def test_probability_bars_are_in_clinical_order_on_a_fixed_axis():
    import json
    at = run(png(fundus(h=1200, w=1600)))
    specs = [json.loads(e.proto.spec) for e in at.main if getattr(e, "type", "") == "arrow_vega_lite_chart"]
    assert len(specs) == 1
    enc = specs[0]["encoding"]
    assert enc["y"]["sort"] is None and enc["x"]["scale"]["domain"] == [0, 1]


def test_confidence_is_never_shown_as_100_percent(monkeypatch):
    import review
    real = review.assess
    monkeypatch.setattr(review, "assess", lambda *a, **k: {**real(*a, **k), "confidence_calibrated": 0.9997})
    at = run(png(fundus(h=1200, w=1600)))
    assert [m.value for m in at.metric if m.label == "Calibrated confidence"] == [">99%"]


def test_a_crash_in_the_analysis_is_explained(monkeypatch):
    import streamlit as st
    from pipeline import RetinoPipeline
    st.cache_data.clear()

    def boom(self, source):
        raise RuntimeError("secret path C:/x")
    monkeypatch.setattr(RetinoPipeline, "predict", boom)
    at = run(png(fundus(h=700, w=900, seed=11)))
    assert not at.exception and any("could not be analysed" in e.value for e in at.error)
    assert "secret path" not in text(at)


def test_page_refuses_to_run_open_to_the_network(monkeypatch):
    from streamlit import config
    old = config.get_option("server.address")
    try:
        config.set_option("server.address", "0.0.0.0")
        at = run(png(fundus(h=700, w=900, seed=12)))
        assert any("Stopped" in e.value for e in at.error) and not at.metric
        monkeypatch.setenv("RG_ALLOW_NETWORK", "1")                        # a deliberate deployment
        assert len(run(png(fundus(h=700, w=900, seed=12))).metric) == 3
    finally:
        config.set_option("server.address", old)
