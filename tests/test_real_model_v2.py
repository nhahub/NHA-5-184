"""Checks that need TensorFlow 2.21 AND the real frozen v2 model file. Skipped when either is missing.

    RG_MODEL=models/B_full_s42_384_mix_model.keras python -m pytest tests/test_real_model_v2.py
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
import retinograde_inference as R  # noqa: E402
from test_inference_v2 import V1_KEYS, fundus  # noqa: E402

pytest.importorskip("tensorflow")
MODEL = os.environ.get(R.MODEL_ENV) or os.path.join(ROOT, R.DEFAULT_MODEL_PATH)
pytestmark = pytest.mark.skipif(not os.path.isfile(MODEL), reason="real v2 model file not available")


@pytest.fixture(scope="module")
def rg():
    return R.RetinoGrade(MODEL)


def test_predict_contract_on_the_real_model(rg):
    r = rg.predict(fundus())
    assert V1_KEYS <= set(r) and r["model_name"] == "B384-mix-v2"
    assert sum(r["probabilities"].values()) == pytest.approx(1, abs=1e-3)
    assert r["grade_name"] == R.CLASS_NAMES[r["grade"]]


def test_compiled_graph_equals_the_plain_model_call(rg):
    x = np.stack([rg.prepare(fundus(seed=s)) for s in range(3)])
    plain = np.asarray(rg.model(x, training=False), np.float64)
    np.testing.assert_allclose(rg.probabilities(x), plain, atol=1e-5)


def test_batch_and_single_predictions_agree(rg):
    imgs = [fundus(seed=s) for s in range(3)]
    batch = rg.predict_batch(imgs)
    for img, b in zip(imgs, batch):
        assert rg.predict(img)["p_referable"] == pytest.approx(b["p_referable"], abs=2e-2)


def test_wrong_size_model_is_refused(tmp_path):
    import tensorflow as tf
    m = tf.keras.Sequential([tf.keras.Input((224, 224, 3)), tf.keras.layers.GlobalAveragePooling2D(),
                             tf.keras.layers.Dense(5, activation="softmax")])
    p = str(tmp_path / "small.keras")
    m.save(p)
    with pytest.raises(ValueError, match="384px"):
        R.RetinoGrade(p)
