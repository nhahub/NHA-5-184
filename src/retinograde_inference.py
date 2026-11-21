"""
RetinoGrade AI - Inference Module (النسخة v1)
==============================================
المصدر الوحيد للحقيقة: كل الفريق (Streamlit / Azure / Grad-CAM / الاختبارات) بيستخدم الملف ده.
متعدّلوش في المعالجة (preprocess_rgb) لأنها لازم تفضل مطابقة 100% للي اتدرّبت بيه الموديلات.

الاستخدام:
    from retinograde_inference import RetinoGrade
    model = RetinoGrade("models/stage1_v1.keras", "models/best_severity_model.keras")
    result = model.predict("eye.png")            # مسار | bytes (رفع Streamlit) | numpy RGB uint8
    print(result["grade_name"], result["confidence"])

من الـ Terminal:
    python retinograde_inference.py eye1.png eye2.png --stage1 models/stage1_v1.keras --stage2 models/best_severity_model.keras

لما يطلع Stage 2 أحسن (v2): غيّروا مسار الملف بس، شكل الـ output ثابت.
"""
import os
import json
import argparse
import numpy as np
import cv2

MODEL_VERSION = "v1"
IMG_SIZE = 224
DEFAULT_THRESHOLD = 0.3          # threshold بتاع Stage 1 (اتختار من جدول Cell 8)
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
DISCLAIMER = "أداة مساعدة للفرز الأولي وليست بديلاً عن تشخيص طبيب العيون."


# ----------------------------------------------------------------------
# تحميل الصورة
# ----------------------------------------------------------------------
def load_rgb(source):
    """source = مسار ملف | bytes | ملف مرفوع (Streamlit UploadedFile) | numpy RGB uint8"""
    if isinstance(source, (str, os.PathLike)):
        img = cv2.imread(str(source))
        if img is None:
            raise ValueError(f"مش قادر أقرا الصورة: {source}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(source, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("الملف مش صورة صالحة")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if hasattr(source, "read"):
        return load_rgb(source.read())
    arr = np.asarray(source)
    if arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype == np.uint8:
        return arr
    raise ValueError("نوع الصورة مش مدعوم (استخدموا مسار / bytes / numpy RGB uint8)")


# ----------------------------------------------------------------------
# المعالجة النهائية (نفس اللي اتدرّبت بيه الموديلات)
# ----------------------------------------------------------------------
def preprocess_rgb(img, target_size=IMG_SIZE, return_geometry=False):
    """img: numpy RGB uint8 -> float32 (224,224,3) بين 0 و1.
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
    """اختصار: مسار/bytes/numpy -> float32 (224,224,3) بين 0 و1"""
    return preprocess_rgb(load_rgb(source), target_size)


def transform_mask(mask, geom, target_size=IMG_SIZE):
    """بيطبّق نفس Crop/Padding/Resize على mask (مثلاً من IDRiD) عشان تنطبق على الصورة المعالجة.
    mask لازم يبقى بنفس مقاس الصورة الأصلية اللي اتحسبت منها geom."""
    m = np.asarray(mask)
    if m.dtype == bool:
        m = m.astype(np.uint8)
    if tuple(m.shape[:2]) != tuple(geom["orig_shape"]):
        raise ValueError(f"مقاس الـ mask {m.shape[:2]} مختلف عن الصورة الأصلية {geom['orig_shape']}")
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
# الـ Cascade: Stage 1 (في اعتلال؟) ثم Stage 2 (الدرجة 1-4)
# ----------------------------------------------------------------------
class RetinoGrade:
    def __init__(self, stage1_path, stage2_path, threshold=DEFAULT_THRESHOLD):
        import tensorflow as tf                        # import هنا عشان باقي الملف يشتغل من غير TensorFlow
        self.stage1 = tf.keras.models.load_model(stage1_path)
        self.stage2 = tf.keras.models.load_model(stage2_path)
        self.threshold = float(threshold)

    @staticmethod
    def _to_batch(sources):
        imgs = np.stack([preprocess_image(s) for s in sources])
        return imgs * 255.0            # الموديلات متدرّبة على 0-255 (EfficientNet بيعمل normalization جواه)

    def _decide(self, p1, p2):
        has_dr = p1 >= self.threshold
        grade = int(np.argmax(p2)) + 1 if has_dr else 0
        # confidence: لو في اعتلال = احتمال الدرجة دي من Stage 2 (مشروط بوجود اعتلال) | لو سليم = 1 - p1
        confidence = float(p2[grade - 1]) if has_dr else float(1.0 - p1)
        return {
            "model_version": MODEL_VERSION,
            "grade": grade,
            "grade_name": CLASS_NAMES[grade],
            "stage1_dr_probability": float(p1),
            "stage2_grade_probabilities": {CLASS_NAMES[i + 1]: float(p2[i]) for i in range(4)},
            "confidence": confidence,
            "referable": grade >= 2,               # Moderate أو أعلى
            "sight_threatening": grade >= 3,       # Severe أو أعلى
            "threshold": self.threshold,
            "disclaimer": DISCLAIMER,
        }

    def predict_batch(self, sources, batch_size=16):
        results = []
        for i in range(0, len(sources), batch_size):
            x = self._to_batch(sources[i:i + batch_size])
            p1 = self.stage1.predict(x, verbose=0).ravel()
            p2 = self.stage2.predict(x, verbose=0)
            results.extend(self._decide(float(a), b) for a, b in zip(p1, p2))
        return results

    def predict(self, source):
        return self.predict_batch([source])[0]


def main():
    ap = argparse.ArgumentParser(description="RetinoGrade AI - تصنيف صور قاع العين")
    ap.add_argument("images", nargs="+", help="مسارات الصور")
    ap.add_argument("--stage1", default=os.environ.get("RG_STAGE1", "models/stage1_v1.keras"))
    ap.add_argument("--stage2", default=os.environ.get("RG_STAGE2", "models/best_severity_model.keras"))
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args()

    model = RetinoGrade(args.stage1, args.stage2, args.threshold)
    out = {path: res for path, res in zip(args.images, model.predict_batch(args.images))}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
