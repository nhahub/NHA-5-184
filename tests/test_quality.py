"""Quality gate: preparation, decision and metrics (no TensorFlow)."""
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from evaluation import quality_metrics as QM  # noqa: E402
from quality import quality as Q  # noqa: E402
from test_inference_v2 import fundus  # noqa: E402


def test_prepare_quality_shape_and_crop():
    x = Q.prepare_quality(fundus(h=900, w=1300))
    assert x.shape == (224, 224, 3) and x.dtype == np.uint8
    # the black border around the round retina is cropped: content touches the frame on at least two sides
    g = x.max(2)
    assert (g[:, 0] > 20).any() or (g[0] > 20).any()


def test_dark_photo_is_not_blown_up():
    """A nearly black photo must stay dark (the border crop would otherwise enlarge a tiny bright speck)."""
    img = np.full((600, 800, 3), 3, np.uint8)
    img[300:304, 400:404] = 200
    x = Q.prepare_quality(img)
    assert x.mean() < 8 and x.shape == (224, 224, 3)


def test_prepare_quality_does_not_equalise_contrast():
    """No CLAHE: a darker version of the same photo stays darker by the same factor."""
    a = Q.prepare_quality(fundus(seed=4)).astype(float)
    b = Q.prepare_quality((fundus(seed=4) * 0.5).astype(np.uint8)).astype(float)
    m = a.max(2) > 40
    assert b[m].mean() == pytest.approx(0.5 * a[m].mean(), rel=0.06)


def test_result_actions():
    r = Q.quality_result([0.1, 0.2, 0.7])
    assert r["label"] == "Reject" and r["action"] == "retake" and r["confidence"] == pytest.approx(0.7)
    assert Q.quality_result([0.1, 0.6, 0.3])["action"] == "warn" and Q.quality_result([0.9, 0.05, 0.05])["action"] == "ok"
    assert set(r["probabilities"]) == set(Q.QUALITY_NAMES)
    with pytest.raises(ValueError):
        Q.quality_result([0.5, 0.5])


def test_metrics_and_tradeoff():
    y = np.array([0] * 6 + [1] * 3 + [2] * 3)
    p = np.eye(3)[y] * 0.8 + 0.1
    p[0] = [0.1, 0.1, 0.8]                            # one Good photo called Reject
    p[11] = [0.5, 0.2, 0.3]                           # one Reject called Good
    rep = QM.report(y, p)
    assert rep["accuracy"] == pytest.approx(10 / 12) and rep["good_called_reject"] == pytest.approx(1 / 6)
    assert rep["reject_called_good"] == pytest.approx(1 / 3) and rep["reject_sensitivity"] == pytest.approx(2 / 3)
    assert rep["per_class"]["Good"]["recall"] == pytest.approx(5 / 6)
    t = {r["threshold"]: r for r in QM.reject_tradeoff(y, p)}
    assert t[0.3]["reject_stopped"] == 1.0 and t[0.5]["reject_stopped"] == pytest.approx(2 / 3)
    assert t[0.2]["good_refused"] == pytest.approx(1 / 6)


def test_augment_keeps_brightness_statistics():
    sys.path.insert(0, os.path.join(ROOT, "src", "training"))
    import train_quality as TQ
    rng = np.random.default_rng(0)
    x = rng.integers(0, 255, (8, 32, 32, 3), dtype=np.uint8)
    a = TQ.augment(x, np.random.default_rng(1))
    np.testing.assert_array_equal(np.sort(x, axis=None), np.sort(a, axis=None))   # only pixels moved
    assert a.shape == x.shape and a.dtype == np.uint8


def test_default_model_paths_are_configurable(monkeypatch, tmp_path):
    import quality.quality as q2
    with pytest.raises(FileNotFoundError, match="RG_QUALITY_MODEL"):
        q2.QualityGate(str(tmp_path / "missing.keras"))


def test_min_side_rejects_tiny_files(monkeypatch):
    """pipeline.predict refuses images whose shorter side is below MIN_SIDE before calling the model."""
    import types
    import pipeline
    p = object.__new__(pipeline.RetinoPipeline)
    p.grader = types.SimpleNamespace(predict=lambda img: {"grade": 0})
    p.gate = types.SimpleNamespace(predict=lambda img: pytest.fail("model must not be called"))
    r = p.predict(np.zeros((50, 300, 3), np.uint8))
    assert r["quality"]["action"] == "retake" and "too small" in r["quality"]["message"]
    p.gate = None
    assert p.predict(np.zeros((50, 300, 3), np.uint8))["quality"] is None
