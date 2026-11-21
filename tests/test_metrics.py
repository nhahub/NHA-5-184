"""Fast local tests (synthetic data, no TensorFlow, no images). Run: pytest tests/ -q"""
import os
import sys

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluation import metrics as M  # noqa: E402


@pytest.fixture
def rand_preds():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 500)
    noise = rng.integers(-1, 2, 500) * (rng.random(500) < 0.4)
    p = np.clip(y + noise, 0, 4)
    return y, p


def test_qwk_matches_sklearn(rand_preds):
    y, p = rand_preds
    assert M.qwk(y, p) == pytest.approx(cohen_kappa_score(y, p, weights="quadratic"), abs=1e-12)


def test_qwk_matches_sklearn_missing_classes():
    y = np.array([0, 0, 2, 2, 4])
    p = np.array([0, 2, 2, 2, 4])
    assert M.qwk(y, p) == pytest.approx(cohen_kappa_score(y, p, weights="quadratic"), abs=1e-12)


def test_qwk_perfect_and_constant():
    y = np.array([0, 1, 2, 3, 4])
    assert M.qwk(y, y) == pytest.approx(1.0)
    assert M.qwk(np.zeros(5), np.zeros(5)) == 1.0


def test_confusion_matches_sklearn(rand_preds):
    y, p = rand_preds
    np.testing.assert_array_equal(M.confusion(y, p), confusion_matrix(y, p, labels=range(5)))


def test_referable_metrics_hand_example():
    y = np.array([0, 1, 2, 3, 4, 2])
    p = np.array([0, 2, 2, 1, 4, 3])
    sens, spec = M.referable_metrics(y, p, 2)       # true ref: idx 2,3,4,5 -> caught 2,4,5
    assert sens == pytest.approx(3 / 4)
    assert spec == pytest.approx(1 / 2)             # true non-ref: idx 0,1 -> idx1 flagged


def test_cascade_predict_matches_v1_rule():
    p1 = np.array([0.1, 0.35, 0.9])
    p2 = np.array([[0.1, 0.7, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])
    np.testing.assert_array_equal(M.cascade_predict(p1, p2, 0.3), [0, 1, 4])
    np.testing.assert_array_equal(M.cascade_predict(p1, p2, 0.5), [0, 0, 4])


def test_cascade_probs5_sums_to_one():
    rng = np.random.default_rng(1)
    p1 = rng.random(20)
    p2 = rng.dirichlet(np.ones(4), 20)
    np.testing.assert_allclose(M.cascade_probs5(p1, p2).sum(1), 1.0)


def test_thresholds_roundtrip_and_fit():
    s = np.array([-0.2, 0.6, 1.49, 1.51, 2.7, 3.9])
    np.testing.assert_array_equal(M.apply_thresholds(s), [0, 1, 1, 2, 3, 4])
    rng = np.random.default_rng(2)
    y = rng.integers(0, 5, 400)
    score = y * 0.8 + rng.normal(0, 0.3, 400)       # biased scale -> default cuts are suboptimal
    cuts = M.fit_thresholds(y, score)
    assert np.all(np.diff(cuts) > 0)
    assert M.qwk(y, M.apply_thresholds(score, cuts)) >= M.qwk(y, M.apply_thresholds(score))


def test_auc_matches_sklearn():
    rng = np.random.default_rng(3)
    yb = rng.integers(0, 2, 300)
    s = yb + rng.normal(0, 1, 300)
    assert M.binary_auc(yb, s) == pytest.approx(roc_auc_score(yb, s))
    assert np.isnan(M.binary_auc(np.ones(5), np.arange(5)))


def test_bootstrap_ci_contains_point_and_is_reproducible(rand_preds):
    y, p = rand_preds
    ci1 = M.bootstrap_ci(y, p, n_boot=200, seed=7)
    ci2 = M.bootstrap_ci(y, p, n_boot=200, seed=7)
    assert ci1 == ci2
    q = M.qwk(y, p)
    assert ci1["qwk"][0] <= q <= ci1["qwk"][1]
    assert ci1["qwk"][1] - ci1["qwk"][0] < 0.15


def test_bootstrap_coverage_sanity():
    """~95% of CIs from independent samples should contain the population QWK."""
    rng = np.random.default_rng(11)
    N = 200_000
    y_pop = rng.integers(0, 5, N)
    p_pop = np.clip(y_pop + rng.integers(-1, 2, N) * (rng.random(N) < 0.5), 0, 4)
    true_q = M.qwk(y_pop, p_pop)
    hits = 0
    trials = 40
    for t in range(trials):
        idx = rng.integers(0, N, 300)
        ci = M.bootstrap_ci(y_pop[idx], p_pop[idx], n_boot=200, seed=t, keys=("qwk",))["qwk"]
        hits += ci[0] <= true_q <= ci[1]
    assert hits / trials >= 0.8


def test_full_report_and_mean_std(rand_preds):
    y, p = rand_preds
    rep = M.full_report(y, p, score_dr=p, score_ref=p, n_boot=50)
    for k in ("qwk", "referable_sens", "any_dr_auc", "per_class", "confusion", "ci95"):
        assert k in rep
    agg = M.mean_std([{"qwk": 0.8, "n": 1}, {"qwk": 0.9, "n": 1}])
    assert agg["qwk"][0] == pytest.approx(0.85)
    assert agg["qwk"][2] == 2
    assert "n" not in agg
