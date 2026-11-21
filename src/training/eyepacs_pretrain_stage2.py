# =====================================================================
# RetinoGrade AI - Pretraining لـ Stage 2 على EyePACS (الصور المصابة فقط: Grade 1-4)
# TensorFlow / Keras + EfficientNetB0
# بعده: بتعملوا Fine-tuning على APTOS بـ stage2_severity_cascade.py (PRETRAINED_STAGE2)
# شغّلوه في Notebook فيه GPU + Internet، وضيفوا Input: داتا eyepacs_224 المجهزة + داتا APTOS المعالجة
# كل "Cell" تحت تتنسخ في خلية منفصلة
# =====================================================================


# ============================== Cell 1 ===============================
import os, shutil, random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D,
                                     BatchNormalization, Dropout, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from sklearn.metrics import (cohen_kappa_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

_DARK = {
    'figure.facecolor': 'black', 'axes.facecolor': 'black', 'savefig.facecolor': 'black',
    'text.color': 'white', 'axes.labelcolor': 'white', 'axes.edgecolor': 'white',
    'axes.titlecolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white',
}
plt.rcParams.update(_DARK)

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)

SEARCH_ROOTS = ["/kaggle/working", "/kaggle/input"]
APTOS_DIR = "/kaggle/input/datasets/omarashour206/aptos2019-preprocessed-dataset"   # داتا APTOS المعالجة (للتقييم Zero-shot)
OUT_MODEL = "/kaggle/working/eyepacs_stage2_pretrained.keras"
BATCH_SIZE = 32
WEIGHTS = "imagenet"       # لازم Internet On
QUICK_TEST = False         # True = اختبار دخان سريع
CLASS_WEIGHT_POWER = 0.5   # 1.0 = balanced كامل | 0.5 = أخف | 0.0 = من غير أوزان
EPOCHS_A = 1 if QUICK_TEST else 4      # المرحلة أ: الـ EfficientNet متجمّد
EPOCHS_B = 1 if QUICK_TEST else 20     # المرحلة ب: Fine-tuning لكل الطبقات (ماعدا BatchNorm)
LR_A, LR_B = 1e-3, 5e-5
CLASS_NAMES = ["Mild", "Moderate", "Severe", "Proliferative"]

print("TF:", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))


# ============================== Cell 2 ===============================
# تحميل داتا EyePACS المجهزة (من eyepacs_prepare.py)
def find_dir_with(filename, roots=SEARCH_ROOTS):
    for root in roots:
        for dirpath, _, files in os.walk(root):
            if filename in files:
                return dirpath
    return None

EYEPACS_DIR = find_dir_with("meta.csv")
assert EYEPACS_DIR, "مش لاقي meta.csv: ضيفوا داتا eyepacs_224 كـ Input (Add Input)"
print("EyePACS dir:", EYEPACS_DIR)

X = np.load(os.path.join(EYEPACS_DIR, "X.npy"), mmap_mode="r")     # uint8 (0-255)
meta = pd.read_csv(os.path.join(EYEPACS_DIR, "meta.csv"))
grade = meta["grade"].values.astype(np.int32)
split = meta["split"].values
ok = meta["ok"].values.astype(bool)

sel = (grade > 0) & ok                       # Stage 2 = الصور المصابة بس
tr_idx = np.where(sel & (split == 0))[0]
va_idx = np.where(sel & (split == 1))[0]
te_idx = np.where(sel & (split == 2))[0]
if QUICK_TEST:
    rng = np.random.RandomState(SEED)
    tr_idx = np.sort(rng.choice(tr_idx, min(64, len(tr_idx)), replace=False))
    va_idx = np.sort(rng.choice(va_idx, min(32, len(va_idx)), replace=False))
    te_idx = np.sort(rng.choice(te_idx, min(32, len(te_idx)), replace=False))

y_tr, y_va, y_te = grade[tr_idx] - 1, grade[va_idx] - 1, grade[te_idx] - 1
print("train:", len(tr_idx), np.bincount(y_tr, minlength=4))
print("val:  ", len(va_idx), np.bincount(y_va, minlength=4))
print("test: ", len(te_idx), np.bincount(y_te, minlength=4))

counts = np.bincount(y_tr, minlength=4)
w = (len(y_tr) / (4.0 * np.maximum(counts, 1))) ** CLASS_WEIGHT_POWER
class_weights = {i: float(w[i]) for i in range(4)}
print("Class weights:", class_weights)


# ============================== Cell 3 ===============================
# Pipeline التحميل + Augmentation وقت التدريب (flip / rotate90 / تغيير خفيف في السطوع والتباين)
_aug_rng = np.random.RandomState(SEED)

def augment_batch(imgs):
    out = np.empty_like(imgs)
    for j in range(len(imgs)):
        im = imgs[j]
        if _aug_rng.rand() < 0.5: im = im[:, ::-1]
        if _aug_rng.rand() < 0.5: im = im[::-1, :]
        im = np.rot90(im, _aug_rng.randint(4))
        im = im * _aug_rng.uniform(0.9, 1.1) + _aug_rng.uniform(-10, 10)
        out[j] = np.clip(im, 0, 255)
    return out

def make_dataset(Xarr, indices, labels, training=False, augment=False, scale=1.0):
    """scale=1.0 لو الداتا 0-255 (EyePACS uint8) | scale=255.0 لو الداتا 0-1 (APTOS float)"""
    ds = tf.data.Dataset.from_tensor_slices((indices.astype(np.int64), labels.astype(np.int32)))
    if training:
        ds = ds.shuffle(len(indices), seed=SEED, reshuffle_each_iteration=True)

    def load_batch(idx, lab):
        def _load(i):
            imgs = np.asarray(Xarr[i], dtype=np.float32) * scale
            return augment_batch(imgs) if augment else imgs
        images = tf.numpy_function(_load, [idx], tf.float32)
        images.set_shape([None, 224, 224, 3])
        return images, lab

    ds = ds.batch(BATCH_SIZE).map(load_batch, num_parallel_calls=1)
    return ds.prefetch(1)

train_ds = make_dataset(X, tr_idx, y_tr, training=True, augment=True)
val_ds = make_dataset(X, va_idx, y_va)
test_ds = make_dataset(X, te_idx, y_te)


# ============================== Cell 4 ===============================
class QWKCallback(Callback):
    def __init__(self, val_ds, y_true):
        super().__init__()
        self.val_ds, self.y_true, self.scores = val_ds, y_true, []

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.val_ds, verbose=0).argmax(axis=1)
        qwk = cohen_kappa_score(self.y_true, preds, weights="quadratic")
        self.scores.append(qwk)
        if logs is not None:
            logs["val_qwk"] = qwk
        print(f" - val_qwk: {qwk:.4f}")


def build_stage2(n_classes=4):
    base = EfficientNetB0(weights=WEIGHTS, include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    inputs = Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(n_classes, activation="softmax")(x)
    return Model(inputs, outputs), base

model, base = build_stage2()
model.compile(optimizer=tf.keras.optimizers.Adam(LR_A),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# ============================== Cell 5 ===============================
# المرحلة أ: تدريب الطبقة الأخيرة بس
PATH_A = OUT_MODEL.replace(".keras", "_phaseA.keras")
qwk_a = QWKCallback(val_ds, y_va)
model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS_A, class_weight=class_weights,
    callbacks=[qwk_a,
               ModelCheckpoint(PATH_A, monitor="val_qwk", mode="max", save_best_only=True, verbose=1),
               EarlyStopping(monitor="val_qwk", mode="max", patience=3, restore_best_weights=True)],
    verbose=1,
)
print("أحسن QWK في المرحلة أ:", max(qwk_a.scores))


# ============================== Cell 6 ===============================
# المرحلة ب: Fine-tuning لكل الطبقات (الـ BatchNorm متجمّد)
base.trainable = True
for layer in base.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False
print("عدد الطبقات القابلة للتدريب:", sum(l.trainable for l in base.layers))

model.compile(optimizer=tf.keras.optimizers.Adam(LR_B),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
qwk_b = QWKCallback(val_ds, y_va)
model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS_B, class_weight=class_weights,
    callbacks=[qwk_b,
               ModelCheckpoint(OUT_MODEL, monitor="val_qwk", mode="max", save_best_only=True, verbose=1),
               EarlyStopping(monitor="val_qwk", mode="max", patience=5, restore_best_weights=True)],
    verbose=1,
)
best_a, best_b = max(qwk_a.scores), max(qwk_b.scores)
print(f"QWK مرحلة أ: {best_a:.4f} | QWK مرحلة ب: {best_b:.4f}")
if best_a > best_b:
    shutil.copy(PATH_A, OUT_MODEL)
    print("المرحلة أ كانت أحسن -> اتنسخت على OUT_MODEL")
print("حجم الموديل:", round(os.path.getsize(OUT_MODEL) / 1024**2, 1), "MB")
print("⚠️ نزّلوا", OUT_MODEL, "من Output وارفعوه كـ Dataset (اسم مقترح: eyepacs-stage2-pretrained)")


# ============================== Cell 7 ===============================
# التقييم: (1) على EyePACS test (مرضى ماشافهمش الموديل) (2) Zero-shot على APTOS من غير أي تدريب عليها
pre = tf.keras.models.load_model(OUT_MODEL)

def evaluate(name, ds, y_true):
    preds = pre.predict(ds, verbose=0).argmax(axis=1)
    print(f"\n===== {name} =====")
    print(classification_report(y_true, preds, labels=[0, 1, 2, 3],
                                target_names=CLASS_NAMES, zero_division=0))
    print("QWK:", round(cohen_kappa_score(y_true, preds, weights="quadratic"), 4),
          "| Acc:", round(float(np.mean(preds == y_true)), 4),
          "| Within-1:", round(float(np.mean(np.abs(preds - y_true) <= 1)), 4))
    with plt.rc_context(_DARK):
        ConfusionMatrixDisplay.from_predictions(
            y_true, preds, labels=[0, 1, 2, 3], display_labels=CLASS_NAMES,
            cmap="Blues", colorbar=False)
        plt.title(name)
        plt.xticks(rotation=25)
        plt.show()

evaluate("EyePACS test (Stage 2)", test_ds, y_te)

aptos_x, aptos_y = os.path.join(APTOS_DIR, "X_val.npy"), os.path.join(APTOS_DIR, "y_val.npy")
if os.path.exists(aptos_x) and os.path.exists(aptos_y):
    Xa = np.load(aptos_x, mmap_mode="r")                 # 0-1 float
    ya = np.load(aptos_y).astype(np.int32)
    ia = np.where(ya > 0)[0]
    aptos_ds = make_dataset(Xa, ia, ya[ia] - 1, scale=255.0)
    evaluate("APTOS val - Zero-shot (الموديل ماتدربش على APTOS)", aptos_ds, ya[ia] - 1)
else:
    print("داتا APTOS المعالجة مش موجودة في APTOS_DIR - اتخطينا التقييم Zero-shot")
