"""
RetinoGrade AI - inference module (model v2)
============================================
Single source of truth for the app, Grad-CAM and the tests. Do not change preprocess_rgb: the models
were trained on exactly its output.

    from retinograde_inference import RetinoGrade
    model = RetinoGrade()                        # weights: argument, else $RG_MODEL, else DEFAULT_MODEL_PATH
    r = model.predict("eye.png")                 # path | bytes | uploaded file | numpy RGB uint8
    print(r["grade_name"], r["confidence"], r["referable"])

    python src/retinograde_inference.py eye1.png eye2.png [--model path.keras]

Model v2 = one EfficientNetB0 with a 5-class softmax at 384x384 (docs/frozen_config.json).
Every output key of v1 is kept. v2 has no Stage 1 / Stage 2, so the legacy keys are derived:
    stage1_dr_probability       = 1 - P(grade 0)
    stage2_grade_probabilities  = P(grade k | grade >= 1) = P(k) / (1 - P(0)),  k = 1..4
    confidence                  = P(predicted grade), the top-class probability
    threshold                   = threshold on P(grade >= 2) behind "referable"
                                  (v1 had no such threshold; there this key was the Stage 1 cut-off 0.3)
Keys added by v2: probabilities, p_referable, p_sight_threatening, thresholds, model_name.

Decision rules (frozen before the one-shot test):
    grade             = argmax of the 5 probabilities
    referable         = P(grade >= 2) >= REFERABLE_THRESHOLD
    sight_threatening = P(grade >= 3) >= SIGHT_THREATENING_THRESHOLD
The two flags use their own thresholds, so a flag can be on although the predicted grade is lower.

v1 (the old cascade at 224px) stays loadable for comparison only:
    RetinoGrade(version="v1", stage1_path=..., stage2_path=...)
"""
import os
import json
import argparse
import numpy as np
import cv2

# --- configuration (one place) ---
IMG_SIZE = 384                                   # the ONLY image-size setting of v2
MODEL_VERSION = "v2"
MODEL_NAME = "B384-mix-v2"
DEFAULT_MODEL_PATH = "models/B_full_s42_384_mix_model.keras"
MODEL_ENV = "RG_MODEL"                           # environment variable with the weights path
REFERABLE_THRESHOLD = 0.3269                     # = docs/frozen_config.json (a test checks it)
SIGHT_THREATENING_THRESHOLD = 0.0277             # = docs/frozen_config.json
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
DISCLAIMER = "Decision-support tool for initial screening only; it does not replace diagnosis by an ophthalmologist."

# --- v1 (comparison only) ---
V1_IMG_SIZE = 224
V1_THRESHOLD = 0.3          # threshold بتاع Stage 1 (اتختار من جدول Cell 8)


# ----------------------------------------------------------------------
# تحميل الصورة
# ----------------------------------------------------------------------
def load_rgb(source):
    """source = مسار ملف | bytes | ملف مرفوع (Streamlit UploadedFile) | numpy RGB uint8"""
    if isinstance(source, (str, os.PathLike)):
        img = cv2.imread(str(source))
        if img is None:
            raise ValueError(f"Cannot read the image: {source}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(source, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("The file is not a valid image")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if hasattr(source, "read"):
        return load_rgb(source.read())
    arr = np.asarray(source)
    if arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype == np.uint8:
        return arr
    raise ValueError("Unsupported image type (use a file path, bytes, or a numpy RGB uint8 array)")


# ----------------------------------------------------------------------
# المعالجة النهائية (نفس اللي اتدرّبت بيه الموديلات)
# ----------------------------------------------------------------------
def preprocess_rgb(img, target_size=IMG_SIZE, return_geometry=False):
    """img: numpy RGB uint8 -> float32 (target_size,target_size,3) بين 0 و1.
    return_geometry=True بيرجّع كمان معلومات الـ Crop والـ Padding (محتاجينها لتحويل masks بتاعة IDRiD)."""
    geom = {"orig_shape": tuple(img.shape[:2]), "crop1": None, "crop2": None, "pad": (0, 0, 0, 0)}

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # المرحلة الأولى: شيل الخلفية السودا الخارجية
    _, thresh_outer = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_outer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        img = img[y:y + h, x:x + w]
        geom["crop1"] = (x, y, w, h)

    original_area_after_stage1 = img.shape[0] * img.shape[1]

    # المرحلة الثانية: شيل أي إطار أبيض (بشرط أمان: متشيلش أكتر من 30% من المساحة)
    gray2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask_retina = cv2.inRange(gray2, 15, 230)
    contours2, _ = cv2.findContours(mask_retina, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours2:
        largest_contour2 = max(contours2, key=cv2.contourArea)
        x2, y2, w2, h2 = cv2.boundingRect(largest_contour2)
        if w2 * h2 >= 0.7 * original_area_after_stage1:
            img = img[y2:y2 + h2, x2:x2 + w2]
            geom["crop2"] = (x2, y2, w2, h2)

    # Padding لمربع قبل الـ Resize (يمنع التمطيط)
    h, w = img.shape[:2]
    size = max(h, w)
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left
    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    geom["pad"] = (pad_top, pad_bottom, pad_left, pad_right)

    img = cv2.resize(img, (target_size, target_size))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img = img.astype(np.float32) / 255.0
    return (img, geom) if return_geometry else img


def preprocess_image(source, target_size=IMG_SIZE):
    """اختصار: مسار/bytes/numpy -> float32 (target_size,target_size,3) بين 0 و1"""
    return preprocess_rgb(load_rgb(source), target_size)


def transform_mask(mask, geom, target_size=IMG_SIZE):
    """بيطبّق نفس Crop/Padding/Resize على mask (مثلاً من IDRiD) عشان تنطبق على الصورة المعالجة.
    mask لازم يبقى بنفس مقاس الصورة الأصلية اللي اتحسبت منها geom."""
    m = np.asarray(mask)
    if m.dtype == bool:
        m = m.astype(np.uint8)
    if tuple(m.shape[:2]) != tuple(geom["orig_shape"]):
        raise ValueError(f"Mask size {m.shape[:2]} differs from the original image size {geom['orig_shape']}")
    if geom["crop1"]:
        x, y, w, h = geom["crop1"]
        m = m[y:y + h, x:x + w]
    if geom["crop2"]:
        x2, y2, w2, h2 = geom["crop2"]
        m = m[y2:y2 + h2, x2:x2 + w2]
    t, b, l, r = geom["pad"]
    m = cv2.copyMakeBorder(m, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(m, (target_size, target_size), interpolation=cv2.INTER_NEAREST)


# ----------------------------------------------------------------------
# Model input and decisions
# ----------------------------------------------------------------------
def to_model_input(img01):
    """preprocess_rgb output (float 0..1) -> exactly the array the model was trained on (0..255, whole numbers).
    Training stored round(img * 255) as uint8; rounding here makes inference bit-exact with it."""
    return np.round(np.asarray(img01) * 255.0).astype(np.float32)


def v2_result(probs5):
    """5 class probabilities -> output dictionary (pure function, no TensorFlow)."""
    p = np.asarray(probs5, dtype=np.float64)
    if p.shape != (5,):
        raise ValueError(f"Expected 5 class probabilities, got shape {p.shape}")
    grade = int(np.argmax(p))
    p_dr = float(p[1:].sum())
    cond = p[1:] / p_dr if p_dr > 0 else np.zeros(4)      # P(k | DR); all 0 if the model gives DR no mass at all
    p_ref, p_st = float(p[2:].sum()), float(p[3:].sum())
    return {
        "model_version": MODEL_VERSION,
        "grade": grade,
        "grade_name": CLASS_NAMES[grade],
        "stage1_dr_probability": float(1.0 - p[0]),
        "stage2_grade_probabilities": {CLASS_NAMES[i + 1]: float(cond[i]) for i in range(4)},
        "confidence": float(p[grade]),
        "referable": bool(p_ref >= REFERABLE_THRESHOLD),
        "sight_threatening": bool(p_st >= SIGHT_THREATENING_THRESHOLD),
        "threshold": REFERABLE_THRESHOLD,
        "disclaimer": DISCLAIMER,
        # added by v2
        "model_name": MODEL_NAME,
        "probabilities": {CLASS_NAMES[i]: float(p[i]) for i in range(5)},
        "p_referable": p_ref,
        "p_sight_threatening": p_st,
        "thresholds": {"referable": REFERABLE_THRESHOLD, "sight_threatening": SIGHT_THREATENING_THRESHOLD},
    }


def v1_result(p1, p2, threshold=V1_THRESHOLD):
    """v1 cascade decision (kept for comparison only)."""
    has_dr = p1 >= threshold
    grade = int(np.argmax(p2)) + 1 if has_dr else 0
    # confidence: لو في اعتلال = احتمال الدرجة دي من Stage 2 (مشروط بوجود اعتلال) | لو سليم = 1 - p1
    confidence = float(p2[grade - 1]) if has_dr else float(1.0 - p1)
    return {
        "model_version": "v1",
        "grade": grade,
        "grade_name": CLASS_NAMES[grade],
        "stage1_dr_probability": float(p1),
        "stage2_grade_probabilities": {CLASS_NAMES[i + 1]: float(p2[i]) for i in range(4)},
        "confidence": confidence,
        "referable": grade >= 2,               # Moderate أو أعلى
        "sight_threatening": grade >= 3,       # Severe أو أعلى
        "threshold": float(threshold),
        "disclaimer": DISCLAIMER,
    }


def resolve_model_path(model_path=None):
    """Explicit argument > environment variable RG_MODEL > DEFAULT_MODEL_PATH."""
    path = str(model_path or os.environ.get(MODEL_ENV) or DEFAULT_MODEL_PATH)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}. Pass model_path or set {MODEL_ENV} (see models/README.md).")
    return path


class RetinoGrade:
    """Load once, then predict() / predict_batch(). TensorFlow is imported only here."""

    def __init__(self, model_path=None, *, version="v2", stage1_path=None, stage2_path=None, threshold=V1_THRESHOLD):
        from model_io import load_model
        self.version = version
        if version == "v2":
            if stage1_path or stage2_path:
                raise ValueError("stage1_path/stage2_path are v1 options: pass version='v1' to load the old cascade")
            import tensorflow as tf
            self.img_size = IMG_SIZE
            self.model_path = resolve_model_path(model_path)
            self.model = load_model(self.model_path)
            size, n_out = int(self.model.input_shape[1]), int(self.model.output_shape[-1])
            if (size, n_out) != (IMG_SIZE, 5):
                raise ValueError(f"{self.model_path} expects {size}px and has {n_out} outputs; v2 needs {IMG_SIZE}px and 5")
            # one compiled graph: same numbers as calling the model, much less Python overhead per image
            self._forward = tf.function(lambda x: self.model(x, training=False),
                                        input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32)])
        elif version == "v1":
            if not (stage1_path and stage2_path):
                raise ValueError("version='v1' needs stage1_path and stage2_path")
            self.img_size = V1_IMG_SIZE
            self.stage1, self.stage2 = load_model(stage1_path), load_model(stage2_path)
            self.threshold = float(threshold)
        else:
            raise ValueError(f"Unknown version {version!r} (use 'v2' or 'v1')")

    def prepare(self, source):
        """Image source -> (S, S, 3) float32 array in 0..255, exactly what the model sees."""
        img = preprocess_rgb(load_rgb(source), self.img_size)
        return to_model_input(img) if self.version == "v2" else img * 255.0

    def probabilities(self, batch_0_255):
        """(N, S, S, 3) prepared images -> (N, 5) probabilities (v2 only)."""
        if self.version != "v2":
            raise ValueError("probabilities() exists only for v2")
        return np.asarray(self._forward(np.asarray(batch_0_255, np.float32)), dtype=np.float64)

    def predict_batch(self, sources, batch_size=16):
        results = []
        for i in range(0, len(sources), batch_size):
            x = np.stack([self.prepare(s) for s in sources[i:i + batch_size]])
            if self.version == "v2":
                results.extend(v2_result(p) for p in self.probabilities(x))
            else:
                p1 = np.asarray(self.stage1(x, training=False)).ravel()
                p2 = np.asarray(self.stage2(x, training=False))
                results.extend(v1_result(float(a), b, self.threshold) for a, b in zip(p1, p2))
        return results

    def predict(self, source):
        return self.predict_batch([source])[0]


def main():
    ap = argparse.ArgumentParser(description="RetinoGrade AI - diabetic retinopathy grading from fundus images")
    ap.add_argument("images", nargs="+", help="Image file paths")
    ap.add_argument("--model", help=f"v2 weights (default: ${MODEL_ENV} or {DEFAULT_MODEL_PATH})")
    ap.add_argument("--version", choices=["v2", "v1"], default="v2", help="v1 = old cascade, comparison only")
    ap.add_argument("--stage1", default=os.environ.get("RG_STAGE1"), help="v1 only")
    ap.add_argument("--stage2", default=os.environ.get("RG_STAGE2"), help="v1 only")
    args = ap.parse_args()

    if args.version == "v2":
        model = RetinoGrade(args.model)
    else:
        model = RetinoGrade(version="v1", stage1_path=args.stage1, stage2_path=args.stage2)
    out = {path: res for path, res in zip(args.images, model.predict_batch(args.images))}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
