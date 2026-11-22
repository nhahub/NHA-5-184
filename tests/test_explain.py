"""Grad-CAM vs lesion-mask metrics (synthetic data, no TensorFlow)."""
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
import retinograde_inference as R  # noqa: E402
from explain import gradcam as G  # noqa: E402
from explain import lesion_overlap as L  # noqa: E402
from test_inference_v2 import fundus  # noqa: E402

S = 384


@pytest.fixture()
def scene():
    retina = np.zeros((S, S), np.uint8)
    cv2.circle(retina, (192, 192), 170, 1, -1)
    lesion = np.zeros((S, S), np.uint8)
    cv2.circle(lesion, (260, 150), 20, 1, -1)
    return retina.astype(bool), lesion.astype(bool)


def test_perfect_and_opposite_heatmaps(scene):
    retina, lesion = scene
    good = L.metrics(L.gaussian_map((S, S), (150, 260), 15), lesion, retina, tol=0)
    bad = L.metrics(L.gaussian_map((S, S), (250, 100), 15), lesion, retina, tol=0)
    assert good["pointing"] == 1 and bad["pointing"] == 0
    assert good["energy"] > 0.5 > 0.05 > bad["energy"]
    assert good["recall@5"] > 0.95 and bad["recall@20"] < 0.2
    assert good["enrichment@5"] > 5


def test_random_maps_match_their_analytic_expectation(scene):
    retina, lesion = scene
    rng = np.random.default_rng(0)
    ms = [L.metrics(rng.random((S, S)), lesion, retina, tol=16) for _ in range(200)]
    e = L.random_expectation(lesion, retina, tol=16)
    for k in ("energy", "precision@10", "recall@10", "enrichment@20"):
        assert np.mean([m[k] for m in ms]) == pytest.approx(e[k], rel=0.05), k
    assert np.mean([m["pointing"] for m in ms]) == pytest.approx(e["pointing"], abs=0.06)


def test_top_k_region_is_exact_and_stays_in_the_retina(scene):
    retina, _ = scene
    heat = np.zeros((S, S))                     # all ties
    for k in L.KS:
        reg = L.top_k_region(heat, retina, k)
        assert reg.sum() == round(retina.sum() * k / 100) and not reg[~retina].any()


def test_heat_outside_the_retina_does_not_count(scene):
    retina, lesion = scene
    heat = L.gaussian_map((S, S), (150, 260), 15)
    out = heat.copy()
    out[~retina] = 100
    assert L.metrics(out, lesion, retina) == L.metrics(heat, lesion, retina)


def test_empty_heatmap_scores_zero(scene):
    retina, lesion = scene
    m = L.metrics(np.zeros((S, S)), lesion, retina)
    assert m["pointing"] == 0 and m["energy"] == 0


def test_optic_disc_baseline_peaks_on_the_disc(scene):
    retina, od = scene
    b = L.baseline_map("optic_disc", retina, od)
    assert od[np.unravel_index(np.argmax(b), b.shape)]
    c = L.baseline_map("centre", retina)
    assert np.hypot(*(np.array(np.unravel_index(np.argmax(c), c.shape)) - 192)) < 3


def test_tiny_lesions_survive_only_with_keep_small():
    img = fundus(h=2848, w=4288, seed=1)
    _, geom = R.preprocess_rgb(img, return_geometry=True)
    mask = np.zeros(img.shape[:2], np.uint8)
    rng = np.random.default_rng(0)
    pts = [(int(rng.integers(1800, 2500)), int(rng.integers(1100, 1700))) for _ in range(40)]
    for x, y in pts:
        mask[y:y + 3, x:x + 3] = 1              # 3x3 px lesion in a 4288 px photo
    near, keep = R.transform_mask(mask, geom), R.transform_mask(mask, geom, keep_small=True)
    assert set(np.unique(keep)) <= {0, 1}
    assert keep.sum() >= 40 > near.sum()
    big = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(big, (2144, 1424), 300, 1, -1)   # for large masks both modes agree except the border
    a, b = R.transform_mask(big, geom) > 0, R.transform_mask(big, geom, keep_small=True) > 0
    assert (a & ~b).sum() == 0 and (b & ~a).sum() < 0.25 * a.sum()


def test_bootstrap_is_paired_and_ignores_nan():
    a, b = np.array([1.0, 2, 3, np.nan]), np.array([0.5, 1.5, 2.5, 1])
    d, lo, hi, n = L.bootstrap_mean_diff(a, b)
    assert (d, lo, hi, n) == (0.5, 0.5, 0.5, 3)


def test_retina_mask_and_overlay_on_a_prepared_image():
    x = R.to_model_input(R.preprocess_rgb(fundus(seed=2)))
    m = L.retina_mask(x)
    assert 0.4 < m.mean() < 0.95 and not m[0, 0]
    o = G.overlay(x, np.where(m, 1.0, 0.0))
    assert o.shape == x.shape and o[0, 0].max() <= 12


@pytest.mark.parametrize("gain", [1.0, 0.25])                    # normal and very dark photo
def test_heat_is_not_painted_on_background_noise(scene, gain):
    """CLAHE lifts the noise of the black background above 12; the first version painted heat speckles there."""
    retina, _ = scene
    x = np.random.default_rng(0).integers(0, 20, (S, S, 3)).astype(np.float32)
    x[retina] = np.array([200.0, 110.0, 40.0]) * gain
    m = L.retina_mask(x)
    assert m[~retina].mean() < 0.02 and m[retina].mean() > 0.97
    outside = ~L.dilate(retina, 4)
    o = G.overlay(x, np.ones((S, S), np.float32))
    assert (o[outside] == x[outside].astype(np.uint8)).all()
