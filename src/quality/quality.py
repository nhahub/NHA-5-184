"""
Image-quality gate: is a fundus photo good enough to be graded? (Good / Usable / Reject, EyeQ labels)

    gate = QualityGate()                       # weights: argument, else $RG_QUALITY_MODEL, else DEFAULT_MODEL_PATH
    q = gate.predict("eye.png")                # {"label": "Reject", "action": "retake", "probabilities": {...}, ...}

Behaviour:
    Good   -> action "ok"
    Usable -> action "warn"     grade it, but show a warning and ask for human review
    Reject -> action "retake"   ask for a new photo; do not show a grade as a result

The gate looks at the photo WITHOUT the contrast enhancement (CLAHE) that the grading model gets:
CLAHE would hide the darkness and haze the gate has to find.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

QUALITY_NAMES = ["Good", "Usable", "Reject"]
QUALITY_IMG_SIZE = 224
DEFAULT_MODEL_PATH = "models/quality_b0_224.keras"
MODEL_ENV = "RG_QUALITY_MODEL"
MIN_SIDE = 64                    # smaller files cannot be judged (nor graded)
ACTIONS = {"Good": "ok", "Usable": "warn", "Reject": "retake"}
MESSAGES = {"Good": "Image quality is good.",
            "Usable": "Image quality is limited. The result needs a human review.",
            "Reject": "Image quality is too low to grade. Please take a new photo."}


def prepare_quality(img_rgb: np.ndarray, size: int = QUALITY_IMG_SIZE) -> np.ndarray:
    """RGB uint8 photo -> (size, size, 3) uint8: black border cropped, padded to a square, resized. No CLAHE.
    Very dark photos (hardly any pixel above the background) are NOT cropped, so they stay visibly dark."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs:
        x, y, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea))
        if w * h >= 0.2 * gray.size:
            img_rgb = img_rgb[y:y + h, x:x + w]
    h, w = img_rgb.shape[:2]
    s = max(h, w)
    t, l = (s - h) // 2, (s - w) // 2
    sq = cv2.copyMakeBorder(img_rgb, t, s - h - t, l, s - w - l, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA if s > size else cv2.INTER_LINEAR)


def quality_result(probs3) -> dict:
    """3 class probabilities -> additive `quality` field (pure function, no TensorFlow)."""
    p = np.asarray(probs3, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError(f"Expected 3 quality probabilities, got shape {p.shape}")
    label = QUALITY_NAMES[int(np.argmax(p))]
    return {"label": label, "action": ACTIONS[label], "message": MESSAGES[label], "confidence": float(p.max()),
            "probabilities": {n: float(v) for n, v in zip(QUALITY_NAMES, p)}}


class QualityGate:
    def __init__(self, model_path=None):
        import tensorflow as tf
        from model_io import load_model
        path = str(model_path or os.environ.get(MODEL_ENV) or DEFAULT_MODEL_PATH)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Quality model not found: {path}. Pass model_path or set {MODEL_ENV} (see models/README.md).")
        self.model_path, self.model = path, load_model(path)
        size, n_out = int(self.model.input_shape[1]), int(self.model.output_shape[-1])
        if (size, n_out) != (QUALITY_IMG_SIZE, 3):
            raise ValueError(f"{path} expects {size}px and has {n_out} outputs; the gate needs {QUALITY_IMG_SIZE}px and 3")
        self._forward = tf.function(lambda x: self.model(x, training=False),
                                    input_signature=[tf.TensorSpec([None, size, size, 3], tf.float32)])

    def probabilities(self, batch_uint8) -> np.ndarray:
        return np.asarray(self._forward(np.asarray(batch_uint8, np.float32)), dtype=np.float64)

    def predict(self, source) -> dict:
        from retinograde_inference import load_rgb
        return quality_result(self.probabilities(prepare_quality(load_rgb(source))[None])[0])
