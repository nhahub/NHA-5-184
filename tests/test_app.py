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


def run(data=None, **env):
    os.environ["RG_APP_TEST"] = "1"
    os.environ[R.MODEL_ENV] = MODEL
    for k, v in env.items():
        os.environ[k] = v
    at = AppTest.from_file(APP, default_timeout=300)
    if data is not None:
        at.session_state["test_bytes"] = data
    at.run()
    return at


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
    at.radio[0].set_value("Referral decision (grade ≥ 2)").run()
    assert not at.exception


@pytest.mark.parametrize("data,expected", [
    (b"this is not an image", "could not be read"),
    (png(np.full((40, 500, 3), 80, np.uint8)), "too small"),
])
def test_bad_files_are_explained_not_crashed(data, expected):
    at = run(data)
    assert not at.exception and any(expected in e.value for e in at.error)


def test_huge_file_is_refused(monkeypatch):
    at = run(b"\0" * (26 * 1024 * 1024))
    assert not at.exception and any("larger than" in e.value for e in at.error)
