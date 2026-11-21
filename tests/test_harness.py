"""Local tests for the TF-free parts of the training harness / data module (synthetic data)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "training"))
from data import aptos_v2 as D  # noqa: E402
import train as T  # noqa: E402


def test_balanced_weights_equalize_class_mass():
    y = np.array([0] * 80 + [1] * 15 + [2] * 5)
    w = T.balanced_sample_weights(y, 1.0, 3)
    mass = [w[y == c].sum() for c in range(3)]
    np.testing.assert_allclose(mass, mass[0], rtol=1e-5)
    np.testing.assert_allclose(T.balanced_sample_weights(y, 0.0, 3), 1.0)


def test_summarize_predictions_all_candidates():
    y = np.array([0, 2, 4])
    a = T.summarize_predictions("A", y, {"p1": np.array([0.1, 0.8, 0.9]),
                                         "p2": np.array([[.7, .1, .1, .1], [.1, .7, .1, .1], [.1, .1, .1, .7]])})
    np.testing.assert_array_equal(a["pred"], [0, 2, 4])
    np.testing.assert_allclose(a["probs5"].sum(1), 1)
    b = T.summarize_predictions("B", y, {"probs5": np.eye(5)[[0, 2, 4]]})
    np.testing.assert_array_equal(b["pred"], [0, 2, 4])
    c = T.summarize_predictions("C", y, {"score": np.array([0.2, 2.1, 3.7])})
    np.testing.assert_array_equal(c["pred"], [0, 2, 4])
    with pytest.raises(ValueError):
        T.summarize_predictions("Z", y, {})


def test_run_id_and_quick_subset():
    cfg = {**T.DEFAULTS, "candidate": "C", "seed": 7, "quick_test": True}
    assert T.run_id(cfg, 3) == "C_f3_s7_224_qt"
    s = T.quick_subset(np.arange(1000), 64, 0)
    assert len(s) == 64 and len(set(s)) == 64 and np.all(np.diff(s) > 0)


def test_augment_keeps_background_dark_and_range():
    rng = np.random.default_rng(0)
    img = np.full((2, 32, 32, 3), 3.0, np.float32)          # CLAHE-lifted background level
    img[:, 8:24, 8:24] = 120.0                               # "retina"
    out = D.augment_batch(img, rng)
    assert out.shape == img.shape
    assert out.min() >= 0 and out.max() <= 255
    assert (out <= D.BACKGROUND_MAX).sum() == (img <= D.BACKGROUND_MAX).sum()   # dihedral ops keep the count
    assert out[out <= D.BACKGROUND_MAX].max() == 0.0          # background not brightened


def test_augment_is_seeded():
    img = np.random.default_rng(1).uniform(20, 200, (4, 16, 16, 3)).astype(np.float32)
    a = D.augment_batch(img, np.random.default_rng(5))
    b = D.augment_batch(img, np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)


class _FakeModel:
    def predict_on_batch(self, x):
        return x.reshape(len(x), -1)[:, :1].astype(np.float64)   # first pixel value = row id


def test_predict_memmap_preserves_order(tmp_path):
    X = np.zeros((20, 4, 4, 3), np.uint8)
    X[:, 0, 0, 0] = np.arange(20)
    np.save(tmp_path / "x.npy", X)
    Xm = np.load(tmp_path / "x.npy", mmap_mode="r")
    idx = np.array([17, 3, 9, 0, 12, 5, 19])
    out = D.predict_memmap(_FakeModel(), Xm, idx, batch_size=3)
    np.testing.assert_array_equal(out.ravel(), idx)


def _meta(rows):
    return pd.DataFrame(rows, columns=["id_code", "grade", "dup_group", "split", "fold"])


def test_check_splits_detects_leaks():
    ok = _meta([("a", 0, 1, "train", 0), ("b", 1, 1, "train", 0), ("c", 2, 2, "test", -1),
                ("d", 2, 2, "dropped", -1)])
    D.check_splits(ok)
    tr, va = D.fold_indices(ok, 0)
    assert len(tr) == 0 and list(va) == [0, 1]
    assert list(D.test_indices(ok)) == [2]
    with pytest.raises(AssertionError):
        D.check_splits(_meta([("a", 0, 1, "train", 0), ("b", 0, 1, "test", -1)]))
    with pytest.raises(AssertionError):
        D.check_splits(_meta([("a", 0, 1, "train", 0), ("b", 0, 1, "train", 1)]))


class _ShapeRecorder(_FakeModel):
    def __init__(self):
        self.shapes = set()

    def predict_on_batch(self, x):
        self.shapes.add(x.shape)
        return super().predict_on_batch(x)


def test_predict_memmap_fixed_batch_shape(tmp_path):
    X = np.zeros((10, 4, 4, 3), np.uint8)
    X[..., 0] = np.arange(10)[:, None, None]          # id in every pixel -> invariant to TTA flips
    np.save(tmp_path / "x.npy", X)
    m = _ShapeRecorder()
    out = D.predict_memmap(m, np.load(tmp_path / "x.npy", mmap_mode="r"), np.arange(10)[::-1], batch_size=4, tta=True)
    np.testing.assert_array_equal(out.ravel(), np.arange(10)[::-1])
    assert m.shapes == {(4, 4, 4, 3)}


def test_predict_run_id_matches_training_names():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "training"))
    import predict as P
    assert P.model_run_id("B", 3, 7, 224) == "B_f3_s7_224"
    with pytest.raises(AssertionError):            # test split refuses to run before the freeze
        P.main({"split": "test", "i_confirm_frozen": False})


def test_fraction_subset_nested_and_stratified():
    idx = np.arange(1000)
    y = np.repeat([0, 1, 2, 3, 4], [500, 100, 250, 50, 100])
    s25, s50 = T.fraction_subset(idx, y, 0.25, 7), T.fraction_subset(idx, y, 0.50, 7)
    assert set(s25) <= set(s50)                                   # nested
    np.testing.assert_array_equal(np.bincount(y[s25], minlength=5), [125, 25, 62, 12, 25])   # stratified
    assert len(T.fraction_subset(idx, y, 1.0, 7)) == 1000
    assert T.run_id({**T.DEFAULTS, "candidate": "B", "train_fraction": 0.5}, 1) == "B_f1_s42_224_p50"


def test_stacked_arrays_reads_across_sources(tmp_path):
    a = np.zeros((3, 2, 2, 3), np.uint8); a[:, 0, 0, 0] = [0, 1, 2]
    b = np.zeros((4, 2, 2, 3), np.uint8); b[:, 0, 0, 0] = [10, 11, 12, 13]
    np.save(tmp_path / "b.npy", b)
    S = D.StackedArrays(a, np.load(tmp_path / "b.npy", mmap_mode="r"))
    assert S.shape == (7, 2, 2, 3) and len(S) == 7
    np.testing.assert_array_equal(S[np.array([0, 2, 3, 6])][:, 0, 0, 0], [0, 2, 10, 13])
    assert S[4][0, 0, 0] == 11
    # predict_memmap works on it unchanged (unsorted request order is restored)
    out = D.predict_memmap(_FakeModel(), S, np.array([6, 0, 4, 2]), batch_size=3)
    np.testing.assert_array_equal(out.ravel(), [13, 0, 11, 2])


def test_mix_run_id():
    assert T.run_id({**T.DEFAULTS, "candidate": "B", "img_size": 384, "extra_data": "eyepacs"}, 0) == "B_f0_s42_384_mix"


def test_gradcam_overlay_and_spearman_without_tf():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from explain import gradcam as G
    img = np.zeros((32, 32, 3), np.float32); img[8:24, 8:24] = 120
    heat = np.zeros((32, 32), np.float32); heat[10:14, 10:14] = 1.0
    out = G.overlay(img, heat)
    assert out.shape == (32, 32, 3) and out.dtype == np.uint8
    assert out[0, 0].sum() == 0                                  # background stays black
    assert not np.array_equal(out[12, 12], img[12, 12].astype(np.uint8))   # hot spot is coloured
    a = np.arange(100.0)
    assert abs(G.spearman(a, a) - 1) < 1e-9 and abs(G.spearman(a, -a) + 1) < 1e-9


def test_full_pool_run_id_and_indices():
    assert T.run_id({**T.DEFAULTS, "candidate": "B", "img_size": 384}, -1) == "B_full_s42_384"
    m = _meta([("a", 0, 1, "train", 0), ("b", 1, 2, "train", 3), ("c", 2, 3, "test", -1)])
    tr, va = D.fold_indices(m, -1)
    assert list(tr) == [0, 1] and len(va) == 0                 # all training rows, never the test row


def test_data_roots_env_override(tmp_path, monkeypatch):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "marker.txt").write_text("x")
    monkeypatch.setenv("RG_DATA_ROOTS", str(tmp_path))
    assert D.data_roots() == (str(tmp_path),)
    assert D.find_file("marker.txt").endswith("marker.txt")
    monkeypatch.delenv("RG_DATA_ROOTS")
    assert D.data_roots() == D.KAGGLE_ROOTS
