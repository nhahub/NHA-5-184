"""Calibration and review rule (pure functions, no TensorFlow)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import review as V  # noqa: E402
from evaluation import calibration as K  # noqa: E402
from retinograde_inference import CLASS_NAMES, REFERABLE_THRESHOLD  # noqa: E402


def rand_probs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(5) * 0.6, n)
    y = np.array([rng.choice(5, p=pp) for pp in p])
    return p, y


def test_temperature_scaling_keeps_order_and_sums_to_one():
    p, _ = rand_probs()
    q = V.calibrate(p, 1.7)
    np.testing.assert_allclose(q.sum(1), 1)
    np.testing.assert_array_equal(q.argmax(1), p.argmax(1))
    assert (q.max(1) <= p.max(1) + 1e-9).all()                 # T > 1 softens
    np.testing.assert_allclose(V.calibrate(p, 1.0), p, atol=1e-6)


def test_fit_temperature_recovers_a_known_temperature():
    p, _ = rand_probs(4000, 1)
    rng = np.random.default_rng(2)
    true = K.apply_temperature(p, 0.5)                          # sharper truth: best T must be near 0.5 when reading p
    y = np.array([rng.choice(5, p=t) for t in true])
    assert K.fit_temperature(p, y) == pytest.approx(0.5, abs=0.06)
    assert K.fit_temperature(true, y) == pytest.approx(1.0, abs=0.12)


def test_ece_of_a_perfectly_calibrated_model_is_small_and_of_an_overconfident_one_is_large():
    p, y = rand_probs(6000, 3)
    assert K.ece(p, y) < 0.04
    assert K.ece(K.apply_temperature(p, 0.4), y) > 0.1


def test_crossfit_only_uses_other_folds():
    p, y = rand_probs(600, 4)
    fold = np.arange(600) % 5
    out, Ts = K.crossfit_temperature(p, y, fold)
    assert len(Ts) == 5
    np.testing.assert_allclose(out[fold == 2], K.apply_temperature(p[fold == 2], Ts[2]))


def test_alarm_row_counts():
    y = np.array([0, 0, 2, 3, 4])
    r = K.alarm_row(y, np.array([0.1, 0.6, 0.2, 0.7, 0.9]), 0.5, 3)
    assert (r["sensitivity"], r["missed"], r["positives"]) == (1.0, 0, 2)
    assert r["ppv"] == pytest.approx(2 / 3) and r["false_alarms_per_1000"] == 200.0


def test_review_reasons():
    sure = dict(zip(CLASS_NAMES, [0.97, 0.01, 0.01, 0.005, 0.005]))
    assert not V.assess(sure)["needs_review"]
    unsure = V.assess([0.3, 0.2, 0.2, 0.2, 0.1])
    assert unsure["needs_review"] and any("unsure" in r for r in unsure["reasons"])
    border = V.assess([0.6, 0.06, 0.34, 0.0, 0.0])             # P(>=2) = 0.34, close to the threshold
    assert any("borderline" in r for r in border["reasons"])
    assert V.assess(sure, "Usable")["needs_review"] and not V.assess(sure, "Good")["needs_review"]
    assert V.assess(sure, "Reject")["needs_review"] is False   # Reject never reaches this point (retake)


def test_calibrated_confidence_uses_the_predicted_grade_and_the_configured_temperature(monkeypatch):
    p = [0.5, 0.1, 0.1, 0.2, 0.1]
    monkeypatch.setenv("RG_TEMPERATURE", "2.0")
    a = V.assess(p)
    assert a["confidence_raw"] == 0.5
    assert a["confidence_calibrated"] == pytest.approx(float(V.calibrate(p, 2.0)[0]))


@pytest.mark.parametrize("delta,flagged", [(0.0, True), (0.19, True), (-0.19, True), (0.21, False), (-0.21, False)])
def test_review_margin_is_symmetric_around_the_frozen_threshold(delta, flagged):
    pr = REFERABLE_THRESHOLD + delta                          # probability of grade >= 2
    p = np.array([1 - pr - 0.01, 0.01, pr, 0.0, 0.0])
    p = p / p.sum()
    reasons = V.assess(p)["reasons"]
    assert any("borderline" in r for r in reasons) is flagged
