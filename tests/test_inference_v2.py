"""Inference contract of model v2 (no TensorFlow, synthetic images)."""
import json
import os
import re
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import retinograde_inference as R  # noqa: E402
from data import aptos_v2 as D  # noqa: E402

V1_KEYS = {"model_version", "grade", "grade_name", "stage1_dr_probability", "stage2_grade_probabilities",
           "confidence", "referable", "sight_threatening", "threshold", "disclaimer"}


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def fundus(h=700, w=900, seed=0, white_frame=False):
    """Synthetic fundus-like RGB photo: textured disc on black, plus a bright 'optic disc' and dark spots."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[:h, :w]
    disc = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < (0.45 * h) ** 2
    img[disc] = (170, 90, 40)
    img = np.clip(img.astype(int) + rng.integers(-12, 13, img.shape) * disc[..., None], 0, 255).astype(np.uint8)
    cv2.circle(img, (w // 2 + 130, h // 2), 45, (250, 220, 150), -1)
    for _ in range(25):
        cv2.circle(img, (int(rng.integers(w // 2 - 200, w // 2 + 200)), int(rng.integers(h // 2 - 200, h // 2 + 200))),
                   int(rng.integers(3, 9)), (110, 20, 20), -1)
    if white_frame:
        img = cv2.copyMakeBorder(img, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img


# ------------------------------------------------------------------ configuration
def test_constants_match_the_frozen_config():
    cfg = json.loads(read("docs", "frozen_config.json"))
    assert R.REFERABLE_THRESHOLD == cfg["thresholds"]["referable"]
    assert R.SIGHT_THREATENING_THRESHOLD == cfg["thresholds"]["sight_threatening"]
    assert R.MODEL_NAME == cfg["name"]
    assert os.path.basename(R.DEFAULT_MODEL_PATH) == cfg["primary"] + "_model.keras"
    assert "384x384" in cfg["model"] and R.IMG_SIZE == 384


def test_image_size_is_one_setting():
    img = fundus()
    out, geom = R.preprocess_rgb(img, return_geometry=True)
    assert out.shape == (R.IMG_SIZE, R.IMG_SIZE, 3) and out.dtype == np.float32
    assert R.preprocess_image(img).shape == out.shape
    assert R.transform_mask(np.zeros(img.shape[:2], np.uint8), geom).shape == (R.IMG_SIZE, R.IMG_SIZE)
    code = re.sub(r'""".*?"""', "", read("src", "retinograde_inference.py"), flags=re.S)
    lines = [l.split("#")[0] for l in code.split("\n") if not l.startswith(("DEFAULT_MODEL_PATH =", "MODEL_NAME ="))]
    assert len(re.findall(r"\b384\b", "\n".join(lines))) == 1


# ------------------------------------------------------------------ preprocessing parity with training
class _Capture:
    def predict_on_batch(self, x):
        self.x = np.array(x)
        return np.zeros((len(x), 5))


@pytest.mark.parametrize("kw", [dict(), dict(h=1200, w=1600, seed=1), dict(h=500, w=500, seed=2),
                                dict(seed=3, white_frame=True)])
def test_inference_input_is_bit_exact_with_training(tmp_path, kw):
    path = str(tmp_path / "eye.png")
    cv2.imwrite(path, cv2.cvtColor(fundus(**kw), cv2.COLOR_RGB2BGR))
    # training: rg_data_v2 / rg_eyepacs_data store this uint8 array, aptos_v2 feeds it to the model as float32
    stored = np.round(R.preprocess_rgb(R.load_rgb(path), 384) * 255).astype(np.uint8)
    np.save(tmp_path / "x.npy", stored[None])
    cap = _Capture()
    D.predict_memmap(cap, np.load(tmp_path / "x.npy"), [0], batch_size=1)
    # inference: exactly the code path of RetinoGrade.prepare
    model_input = R.to_model_input(R.preprocess_rgb(R.load_rgb(path), R.IMG_SIZE))
    assert model_input.dtype == np.float32
    np.testing.assert_array_equal(cap.x[0], model_input)
    with open(path, "rb") as f:
        raw = f.read()
    for src in (raw, R.load_rgb(path)):                                    # bytes and arrays give the same input
        np.testing.assert_array_equal(R.to_model_input(R.preprocess_rgb(R.load_rgb(src))), model_input)


@pytest.mark.parametrize("kernel", ["rg_data_v2.py", "rg_eyepacs_data.py"])
def test_training_kernels_still_use_the_same_expression(kernel):
    assert "np.round(preprocess_rgb(load_rgb(path), IMG) * 255).astype(np.uint8)" in read("kaggle", "kernels", kernel)


def test_transform_mask_stays_aligned_with_the_image():
    img = fundus(h=900, w=1300, seed=5)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mask, (1300 // 2 + 130, 450), 45, 1, -1)                    # the bright optic disc drawn by fundus()
    out, geom = R.preprocess_rgb(img, return_geometry=True)
    m = R.transform_mask(mask, geom).astype(bool)
    assert 50 < m.sum() < 5000
    bright = out.mean(2)
    assert bright[m].mean() > bright[~m & (bright > 0.1)].mean() + 0.2     # the mask sits on the bright disc


# ------------------------------------------------------------------ output contract
def test_every_v1_key_is_kept_and_legacy_keys_are_derived():
    r = R.v2_result(np.array([0.10, 0.20, 0.40, 0.20, 0.10]))
    assert V1_KEYS <= set(r) and r["model_version"] == "v2"
    assert r["grade"] == 2 and r["grade_name"] == "Moderate" and r["confidence"] == pytest.approx(0.40)
    assert r["stage1_dr_probability"] == pytest.approx(0.90)
    cond = r["stage2_grade_probabilities"]
    assert list(cond) == R.CLASS_NAMES[1:] and sum(cond.values()) == pytest.approx(1)
    assert cond["Moderate"] == pytest.approx(0.40 / 0.90)
    assert r["p_referable"] == pytest.approx(0.70) and r["p_sight_threatening"] == pytest.approx(0.30)
    assert r["threshold"] == r["thresholds"]["referable"] == R.REFERABLE_THRESHOLD
    json.dumps(r)                                                          # plain JSON types only


def test_flags_use_the_frozen_thresholds_not_the_grade():
    eps = 1e-6
    t = R.REFERABLE_THRESHOLD
    assert R.v2_result([1 - t - eps, 0, t + eps, 0, 0])["referable"] is True      # grade 0, flag on
    assert R.v2_result([1 - t + eps, 0, t - eps, 0, 0])["referable"] is False
    t = R.SIGHT_THREATENING_THRESHOLD
    assert R.v2_result([1 - t - eps, 0, 0, t + eps, 0])["sight_threatening"] is True
    assert R.v2_result([1 - t + eps, 0, 0, 0, t - eps])["sight_threatening"] is False


def test_degenerate_probabilities():
    r = R.v2_result([1, 0, 0, 0, 0])
    assert r["grade"] == 0 and sum(r["stage2_grade_probabilities"].values()) == 0 and not r["referable"]
    with pytest.raises(ValueError):
        R.v2_result([0.5, 0.5])


def test_v1_decision_is_unchanged():
    r = R.v1_result(0.8, np.array([0.1, 0.6, 0.2, 0.1]))
    assert set(r) == V1_KEYS and r["grade"] == 2 and r["referable"] and not r["sight_threatening"] and r["threshold"] == 0.3
    assert R.v1_result(0.29, np.array([0, 0, 0, 1.0]))["grade"] == 0


def test_five_grades_only():
    assert len(R.CLASS_NAMES) == 5
    for root, _, files in os.walk(os.path.join(ROOT, "src")):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    assert "screening_level" not in fh.read(), f


# ------------------------------------------------------------------ loading options
def test_model_path_resolution(tmp_path, monkeypatch):
    f = tmp_path / "m.keras"
    f.write_bytes(b"x")
    monkeypatch.setenv(R.MODEL_ENV, str(f))
    assert R.resolve_model_path() == str(f)
    with pytest.raises(FileNotFoundError, match="RG_MODEL"):
        R.resolve_model_path(str(tmp_path / "missing.keras"))


def test_v1_must_be_asked_for_explicitly():
    with pytest.raises(TypeError):
        R.RetinoGrade("stage1.keras", "stage2.keras")                      # old positional call is refused
    with pytest.raises(ValueError, match="version='v1'"):
        R.RetinoGrade(stage1_path="a", stage2_path="b")
    with pytest.raises(ValueError, match="needs stage1_path"):
        R.RetinoGrade(version="v1")
    with pytest.raises(ValueError, match="Unknown version"):
        R.RetinoGrade(version="v3")


@pytest.mark.parametrize("bad", [b"not an image", np.zeros((10, 10)), np.zeros((10, 10, 3), np.float32)])
def test_bad_inputs_raise_value_error(bad):
    with pytest.raises(ValueError):
        R.load_rgb(bad)


def test_file_names_with_arabic_letters_can_be_read(tmp_path):
    """cv2.imread cannot open such names on Windows."""
    import cv2
    img = fundus(h=300, w=400)
    f = tmp_path / "صورة عين.png"
    f.write_bytes(cv2.imencode(".png", img[..., ::-1])[1].tobytes())
    np.testing.assert_array_equal(R.load_rgb(str(f)), img)
    np.testing.assert_array_equal(R.load_rgb(f), img)


def test_empty_input_is_a_value_error_and_streams_are_rewound(tmp_path):
    import io

    import cv2
    with pytest.raises(ValueError):
        R.load_rgb(b"")
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    for bad in (str(empty), str(tmp_path / "missing.png")):
        with pytest.raises(ValueError, match="Cannot read the image"):
            R.load_rgb(bad)
    img = fundus(h=300, w=400)
    stream = io.BytesIO(cv2.imencode(".png", img[..., ::-1])[1].tobytes())
    stream.read()                                                          # someone read it before
    np.testing.assert_array_equal(R.load_rgb(stream), img)


def test_default_model_path_works_from_any_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = R.default_path(R.DEFAULT_MODEL_PATH)
    assert os.path.isabs(p) and p.replace("\\", "/").endswith(R.DEFAULT_MODEL_PATH)
    (tmp_path / "models").mkdir()
    (tmp_path / R.DEFAULT_MODEL_PATH).write_bytes(b"x")
    assert R.default_path(R.DEFAULT_MODEL_PATH) == R.DEFAULT_MODEL_PATH    # a file in the current folder comes first
