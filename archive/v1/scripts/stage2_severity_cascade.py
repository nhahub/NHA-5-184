# =====================================================================
# RetinoGrade AI - Stage 2: DR Severity (Grade 1-4) + Cascade Evaluation
# TensorFlow / Keras + EfficientNetB0
# كل "Cell" تحت تتنسخ في خلية منفصلة في Kaggle Notebook وتتشغل بالترتيب
# =====================================================================


# ============================== Cell 1 ===============================
# الإعدادات
import os, shutil, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D,
                                     BatchNormalization, Dropout, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from sklearn.metrics import (cohen_kappa_score, classification_report,
                             ConfusionMatrixDisplay)

# ثيم غامق كامل: خلفية سودا + كتابة بيضا (لازم الاتنين مع بعض عشان الأرقام وأسماء الفئات تبان)
plt.rcParams.update({
    'figure.facecolor': 'black', 'axes.facecolor': 'black', 'savefig.facecolor': 'black',
    'text.color': 'white', 'axes.labelcolor': 'white', 'axes.edgecolor': 'white',
    'axes.titlecolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white',
})

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)

DATA_DIR    = "/kaggle/input/datasets/omarashour206/aptos2019-preprocessed-dataset"
STAGE1_FILENAMES = ["stage1_v1.keras", "best_binary_dr_model.keras"]   # Cell 8 بيدوّر بالترتيب ده (النسخة الثابتة الأول)
SEARCH_ROOTS = ["/kaggle/working", "/kaggle/input"]  # الأماكن اللي بندوّر فيها على الموديل
CLASS_WEIGHT_POWER = 1.0   # 1.0 = balanced كامل (اللي اتجرب) | 0.5 = أخف | 0.0 = من غير أوزان

# --- Fine-tuning من موديل EyePACS المتدرّب (ناتج eyepacs_pretrain_stage2.py) ---
PRETRAINED_STAGE2 = None   # حطوا مسار الموديل، مثال: "/kaggle/input/datasets/<user>/eyepacs-stage2-pretrained/eyepacs_stage2_pretrained.keras"
AUGMENT = False            # True = Augmentation (flip/rotate) وقت التدريب (فعّلوه مع الـ Pretrained)
FINETUNE_LAYERS = 30       # عدد آخر الطبقات اللي بتتفك في المرحلة ب (مع الـ Pretrained حطوا 100000 = كلها)
LR_B = 1e-5                # معدل تعلم المرحلة ب (مع الـ Pretrained: 2e-5)

# كل تجربة بتتحفظ باسم ملف مختلف، فالموديلات القديمة مش بتتمسح
if PRETRAINED_STAGE2:
    STAGE2_PATH = "/kaggle/working/best_severity_model_eyepacs_ft.keras"
elif CLASS_WEIGHT_POWER == 1.0:
    STAGE2_PATH = "/kaggle/working/best_severity_model.keras"
else:
    STAGE2_PATH = f"/kaggle/working/best_severity_model_pow{CLASS_WEIGHT_POWER}.keras"
BATCH_SIZE  = 32
WEIGHTS     = "imagenet"   # لازم Internet يبقى On في Session options
QUICK_TEST  = False        # True = اختبار دخان سريع (عينة صغيرة + Epoch واحد)

print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))


# ============================== Cell 2 ===============================
# تحميل الداتا وتجهيز داتا Stage 2 (الصور المصابة فقط: Grade 1-4)
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"), mmap_mode="r")
y_train_full = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(np.int32)
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"), mmap_mode="r")
y_val_full = np.load(os.path.join(DATA_DIR, "y_val.npy")).astype(np.int32)

tr_idx = np.where(y_train_full > 0)[0]
va_idx = np.where(y_val_full > 0)[0]
if QUICK_TEST:
    rng = np.random.RandomState(SEED)
    tr_idx = np.sort(rng.choice(tr_idx, 64, replace=False))
    va_idx = np.sort(rng.choice(va_idx, 32, replace=False))

y_tr = y_train_full[tr_idx] - 1     # 1..4  ->  0..3
y_va = y_val_full[va_idx] - 1

CLASS_NAMES = ["Mild", "Moderate", "Severe", "Proliferative"]
print("Stage-2 train:", len(tr_idx), "| per class:", np.bincount(y_tr, minlength=4))
print("Stage-2 val:  ", len(va_idx), "| per class:", np.bincount(y_va, minlength=4))


# ============================== Cell 3 ===============================
# Pipeline التحميل (نفس فكرة Stage 1: الصور 0-1 بتتضرب في 255 لأن EfficientNet عايز 0-255)
_aug_rng = np.random.RandomState(SEED)

def augment_batch(imgs):
    """Augmentation لكل صورة لوحدها: flip / rotate90 / تغيير خفيف في السطوع والتباين"""
    out = np.empty_like(imgs)
    for j in range(len(imgs)):
        im = imgs[j]
        if _aug_rng.rand() < 0.5: im = im[:, ::-1]
        if _aug_rng.rand() < 0.5: im = im[::-1, :]
        im = np.rot90(im, _aug_rng.randint(4))
        im = im * _aug_rng.uniform(0.9, 1.1) + _aug_rng.uniform(-10, 10)
        out[j] = np.clip(im, 0, 255)
    return out


def make_dataset(X, indices, labels, training=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((indices.astype(np.int64), labels))
    if training:
        ds = ds.shuffle(len(indices), seed=SEED, reshuffle_each_iteration=True)

    def load_batch(idx, lab):
        def _load(i):
            imgs = np.asarray(X[i], dtype=np.float32) * 255.0
            return augment_batch(imgs) if augment else imgs
        images = tf.numpy_function(_load, [idx], tf.float32)
        images.set_shape([None, 224, 224, 3])
        return images, lab

    ds = ds.batch(BATCH_SIZE).map(load_batch, num_parallel_calls=1)
    return ds.prefetch(1)

# Class weights (balanced) محسوبة من الـ labels اللي الموديل هيتدرب عليها فعلاً
counts = np.bincount(y_tr, minlength=4)
w = (len(y_tr) / (4.0 * np.maximum(counts, 1))) ** CLASS_WEIGHT_POWER
class_weights = {i: float(w[i]) for i in range(4)}
print("Class weights:", class_weights)

train_ds = make_dataset(X_train, tr_idx, y_tr.astype(np.int32), training=True, augment=AUGMENT)
val_ds   = make_dataset(X_val,   va_idx, y_va.astype(np.int32))


# ============================== Cell 4 ===============================
# Callback بيحسب QWK آخر كل Epoch (عشان نختار أحسن موديل بالمقياس الحقيقي مش بالـ loss بس)
class QWKCallback(Callback):
    def __init__(self, val_ds, y_true):
        super().__init__()
        self.val_ds, self.y_true, self.scores = val_ds, y_true, []

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.val_ds, verbose=0).argmax(axis=1)
        qwk = cohen_kappa_score(self.y_true, preds, weights="quadratic")
        self.scores.append(qwk)
        if logs is not None:
            logs["val_qwk"] = qwk          # عشان ModelCheckpoint / EarlyStopping يشوفوه
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

if PRETRAINED_STAGE2:
    model = tf.keras.models.load_model(PRETRAINED_STAGE2)
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))   # الـ EfficientNet جوه الموديل
    print("اتحمّل موديل EyePACS المتدرّب:", PRETRAINED_STAGE2)
else:
    model, base = build_stage2()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()


# ============================== Cell 5 ===============================
# المرحلة أ: تدريب الطبقة الأخيرة بس (الـ EfficientNet متجمّد)
PATH_A = STAGE2_PATH.replace(".keras", "_phaseA.keras")
qwk_a = QWKCallback(val_ds, y_va)

if PRETRAINED_STAGE2:
    print("موديل Pretrained: بنعدّي المرحلة أ ونروح للـ Fine-tuning (Cell 6) على طول")
else:
    history_a = model.fit(
        train_ds, validation_data=val_ds,
        epochs=1 if QUICK_TEST else 10,
        class_weight=class_weights,
        callbacks=[
            qwk_a,   # لازم يبقى الأول في القايمة
            ModelCheckpoint(PATH_A, monitor="val_qwk", mode="max", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_qwk", mode="max", patience=4,
                          restore_best_weights=True, verbose=1),
        ],
        verbose=1,
    )
    print("أحسن QWK في المرحلة أ:", max(qwk_a.scores))


# ============================== Cell 6 ===============================
# المرحلة ب: Fine-tuning لآخر FINETUNE_LAYERS طبقة بمعدل تعلم صغير LR_B (الـ BatchNorm يفضل متجمّد)
base.trainable = True
for layer in base.layers[:-FINETUNE_LAYERS]:
    layer.trainable = False
for layer in base.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False
print("عدد الطبقات القابلة للتدريب:", sum(l.trainable for l in base.layers))

model.compile(optimizer=tf.keras.optimizers.Adam(LR_B),   # لازم compile تاني بعد تغيير trainable
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

qwk_b = QWKCallback(val_ds, y_va)
history_b = model.fit(
    train_ds, validation_data=val_ds,
    epochs=1 if QUICK_TEST else 20,
    class_weight=class_weights,
    callbacks=[
        qwk_b,
        ModelCheckpoint(STAGE2_PATH, monitor="val_qwk", mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_qwk", mode="max", patience=6,
                      restore_best_weights=True, verbose=1),
    ],
    verbose=1,
)

# STAGE2_PATH يفضل دايماً فيه الأحسن بين المرحلتين
best_a = max(qwk_a.scores) if qwk_a.scores else -1.0     # -1 = المرحلة أ اتعدّت (Pretrained)
best_b = max(qwk_b.scores)
print(f"QWK مرحلة أ: {best_a:.4f} | QWK مرحلة ب: {best_b:.4f}")
if best_a > best_b:
    shutil.copy(PATH_A, STAGE2_PATH)
    print("المرحلة أ كانت أحسن -> اتنسخت على STAGE2_PATH")
print("حجم موديل Stage 2 المحفوظ:", round(os.path.getsize(STAGE2_PATH) / 1024**2, 1), "MB")
print("⚠️ ملفات /kaggle/working بتضيع لما الـ Session تقفل: نزّلوا الملف من Output (يمين) وارفعوه كـ Dataset")


# ============================== Cell 7 ===============================
# تقييم Stage 2 لوحده (على الصور المصابة بس)
stage2 = tf.keras.models.load_model(STAGE2_PATH)
preds2 = stage2.predict(val_ds, verbose=0).argmax(axis=1)

print(classification_report(y_va, preds2, labels=[0, 1, 2, 3],
                            target_names=CLASS_NAMES, zero_division=0))
print("QWK (Stage 2 لوحده):", round(cohen_kappa_score(y_va, preds2, weights="quadratic"), 4))
print("Accuracy:", round(float(np.mean(preds2 == y_va)), 4))
print("Within-1-grade accuracy:", round(float(np.mean(np.abs(preds2 - y_va) <= 1)), 4))

ConfusionMatrixDisplay.from_predictions(
    y_va, preds2, labels=[0, 1, 2, 3], display_labels=CLASS_NAMES, cmap="Blues")
plt.title("Stage 2 - Confusion Matrix (Grade 1-4)")
plt.show()


# ============================== Cell 8 ===============================
# التقييم الحقيقي: Cascade كامل (Stage 1 ثم Stage 2) على كل الـ Validation، بمقياس QWK من 0 لـ 4
def find_model(filenames, roots=SEARCH_ROOTS):
    for filename in filenames:                     # بالترتيب: stage1_v1 الأول وبعدين الاسم القديم
        for root in roots:
            for dirpath, _, files in os.walk(root):
                if filename in files:
                    return os.path.join(dirpath, filename)
    return None


def evaluate_cascade(stage1_path):
    stage1 = tf.keras.models.load_model(stage1_path)
    stage2 = tf.keras.models.load_model(STAGE2_PATH)   # بنحمّله هنا عشان Cell 8 يشتغل لوحده من غير Cell 7

    all_idx = np.arange(len(y_val_full))
    all_ds = make_dataset(X_val, all_idx, y_val_full)       # الـ labels مش بتتستخدم في predict

    p1 = stage1.predict(all_ds, verbose=0).ravel()          # احتمال "في اعتلال"
    p2 = stage2.predict(all_ds, verbose=0)                  # Stage 2 على كل الصور (بنستخدمه بس لو Stage 1 قال مريض)
    grade_if_dr = p2.argmax(axis=1) + 1                     # 1..4

    print("threshold | QWK    | Acc   | DR recall | سليم اتصنّف مريض")
    for thr in [0.5, 0.4, 0.3, 0.2]:
        final = np.where(p1 >= thr, grade_if_dr, 0)
        qwk = cohen_kappa_score(y_val_full, final, weights="quadratic")
        recall = float(np.mean(p1[y_val_full > 0] >= thr))
        flagged = float(np.mean(p1[y_val_full == 0] >= thr))
        acc = float(np.mean(final == y_val_full))
        print(f"   {thr:.1f}    | {qwk:.4f} | {acc:.3f} |   {recall:.3f}   |   {flagged:.3f}")

    # إيه الـ Grades اللي Stage 1 فوّتها (عند threshold = 0.5)؟
    missed = (y_val_full > 0) & (p1 < 0.5)
    print("\nحالات مصابة Stage 1 فوّتها [Grade1, Grade2, Grade3, Grade4]:",
          np.bincount(y_val_full[missed], minlength=5)[1:])

    final = np.where(p1 >= 0.5, grade_if_dr, 0)
    ConfusionMatrixDisplay.from_predictions(
        y_val_full, final, labels=[0, 1, 2, 3, 4], cmap="Blues")
    plt.title("Cascade (threshold=0.5) - Confusion Matrix (Grade 0-4)")
    plt.show()
    return p1, p2


P1 = P2 = None
STAGE1_PATH = find_model(STAGE1_FILENAMES)
if STAGE1_PATH is None:
    print(f"⚠️ مش لاقي أي ملف من {STAGE1_FILENAMES} في {SEARCH_ROOTS}")
    print("ده طبيعي لو Stage 1 اتدرّب في Notebook تاني (كل Notebook ليه /kaggle/working لوحده).")
    print("الحل: ارفعوا الملف كـ Dataset وضيفوه Input لـ Notebook ده (Add Input) وشغّلوا Cell 8 تاني.")
else:
    print("لقيت موديل Stage 1 في:", STAGE1_PATH)
    frozen = "/kaggle/working/stage1_v1.keras"
    if STAGE1_PATH.startswith("/kaggle/working") and not os.path.exists(frozen):
        shutil.copy(STAGE1_PATH, frozen)           # نسخة ثابتة: نزّلوها وارفعوها كـ Dataset ومتدربوهاش تاني
        print("اتعملت نسخة ثابتة:", frozen)
    P1, P2 = evaluate_cascade(STAGE1_PATH)


# ============================== Cell 9 ===============================
# تجربة من غير إعادة تدريب: تصحيح الـ prior في توقعات Stage 2 + مقاييس Referable / Sight-threatening
# الـ balanced weights بتدرّب الموديل كأن الفئات متساوية، لكن في الحقيقة Moderate هي الأكتر.
# الـ prior الطبيعي لـ Grade 1-4 في APTOS الكامل: 370 / 999 / 193 / 295
NATURAL_PRIOR = np.array([370, 999, 193, 295], dtype=np.float64)
NATURAL_PRIOR /= NATURAL_PRIOR.sum()
THR = 0.3   # threshold بتاع Stage 1 (اتختار من جدول Cell 8: Recall أعلى بتكلفة صغيرة)


def referable_metrics(y_true, y_pred, thr=2):
    """thr=2: Referable DR (Grade 2 فما فوق) | thr=3: Sight-threatening (Grade 3 فما فوق)"""
    t, p = y_true >= thr, y_pred >= thr
    sens = (t & p).sum() / max(t.sum(), 1)
    spec = (~t & ~p).sum() / max((~t).sum(), 1)
    return sens, spec


def prior_sweep():
    print("alpha | S2 QWK | S2 Acc | Cascade QWK | Referable Sens/Spec | Sight-threat. Sens/Spec | Grade4 recall")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        adj = P2 * (NATURAL_PRIOR ** alpha)              # alpha=0 يعني من غير أي تصحيح
        pred_dr = adj[va_idx].argmax(axis=1)             # Stage 2 لوحده (الصور المصابة بس)
        qwk_s2 = cohen_kappa_score(y_va, pred_dr, weights="quadratic")
        acc_s2 = float(np.mean(pred_dr == y_va))

        final = np.where(P1 >= THR, adj.argmax(axis=1) + 1, 0)   # Cascade كامل
        qwk_c = cohen_kappa_score(y_val_full, final, weights="quadratic")
        s2, p2_ = referable_metrics(y_val_full, final, 2)
        s3, p3_ = referable_metrics(y_val_full, final, 3)
        is4 = y_val_full == 4
        rec4 = float(np.mean(final[is4] == 4)) if is4.any() else float("nan")
        print(f" {alpha:.2f} | {qwk_s2:.4f} | {acc_s2:.3f}  |   {qwk_c:.4f}    |   {s2:.3f} / {p2_:.3f}   |    {s3:.3f} / {p3_:.3f}    |    {rec4:.3f}")


def show_missed_cases(min_grade=2):
    """الحالات (Grade 2+) اللي Stage 1 فوّتها عند THR: لو p1 قريبة من THR يبقى threshold أقل كان هيمسكها"""
    missed = np.where((y_val_full >= min_grade) & (P1 < THR))[0]
    print(f"\nحالات Grade {min_grade}+ اللي Stage 1 فوّتها عند THR={THR}: {len(missed)}")
    for i in missed:
        print(f"   index={i}  grade={y_val_full[i]}  p1={P1[i]:.3f}")


if P1 is None:
    print("شغّلوا Cell 8 الأول (محتاجين P1 و P2)")
else:
    prior_sweep()
    show_missed_cases()


# ============================== Cell 10 ==============================
# Confusion Matrix لـ Stage 2 لوحده (على الصور المصابة بس) لكذا قيمة alpha + Recall/Precision لكل درجة
from sklearn.metrics import confusion_matrix

ALPHAS_TO_SHOW = [0.0, 0.5, 0.75]
_DARK = {
    'figure.facecolor': 'black', 'axes.facecolor': 'black', 'savefig.facecolor': 'black',
    'text.color': 'white', 'axes.labelcolor': 'white', 'axes.edgecolor': 'white',
    'axes.titlecolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white',
}


def show_stage2_alone():
    with plt.rc_context(_DARK):
        fig, axes = plt.subplots(1, len(ALPHAS_TO_SHOW), figsize=(6 * len(ALPHAS_TO_SHOW), 5.5))
        axes = np.atleast_1d(axes)
        for ax, alpha in zip(axes, ALPHAS_TO_SHOW):
            pred = (P2 * (NATURAL_PRIOR ** alpha))[va_idx].argmax(axis=1)
            cm = confusion_matrix(y_va, pred, labels=[0, 1, 2, 3])
            qwk = cohen_kappa_score(y_va, pred, weights="quadratic")
            ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
                ax=ax, cmap="Blues", colorbar=False, values_format="d")
            ax.set_title(f"Stage 2 alone | alpha={alpha} | QWK={qwk:.3f}")
            ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        plt.show()

    for alpha in ALPHAS_TO_SHOW:
        pred = (P2 * (NATURAL_PRIOR ** alpha))[va_idx].argmax(axis=1)
        cm = confusion_matrix(y_va, pred, labels=[0, 1, 2, 3])
        recall = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
        precision = cm.diagonal() / np.maximum(cm.sum(axis=0), 1)
        print(f"\nalpha = {alpha}")
        for name, r, p in zip(CLASS_NAMES, recall, precision):
            print(f"   {name:<14} recall={r:.2f}  precision={p:.2f}")


if P2 is None:
    print("شغّلوا Cell 8 الأول (محتاجين P2)")
else:
    show_stage2_alone()


# ============================== Cell 11 ==============================
# مقارنة كل موديلات Stage 2 المحفوظة (مثلاً weights 1.0 مقابل 0.5) على نفس P1 ونفس الـ THR
import fnmatch
from sklearn.metrics import confusion_matrix


def find_models(pattern, roots=SEARCH_ROOTS):
    found = []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if fnmatch.fnmatch(f, pattern) and "_phaseA" not in f:
                    found.append(os.path.join(dirpath, f))
    return sorted(found)


def report_stage2(label, p2):
    """بيطبع نفس المقاييس لأي مصفوفة احتمالات p2 (شكلها: عدد صور الـ Validation x 4)"""
    pred_dr = p2[va_idx].argmax(axis=1)
    cm = confusion_matrix(y_va, pred_dr, labels=[0, 1, 2, 3])
    rec = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    final = np.where(P1 >= THR, p2.argmax(axis=1) + 1, 0)
    qwk_c = cohen_kappa_score(y_val_full, final, weights="quadratic")
    s2, sp2 = referable_metrics(y_val_full, final, 2)
    s3, sp3 = referable_metrics(y_val_full, final, 3)
    print(f"\n{label}")
    print(f"   Stage 2 لوحده : QWK={cohen_kappa_score(y_va, pred_dr, weights='quadratic'):.4f}  Acc={np.mean(pred_dr == y_va):.3f}")
    print(f"   Cascade       : QWK={qwk_c:.4f}  Acc={np.mean(final == y_val_full):.3f}")
    print(f"   Referable     : Sens={s2:.3f}  Spec={sp2:.3f}   | Sight-threatening: Sens={s3:.3f}  Spec={sp3:.3f}")
    print("   Recall لكل درجة: " + "  ".join(f"{n}={r:.2f}" for n, r in zip(CLASS_NAMES, rec)))


def predict_all_stage2():
    """بيرجّع {اسم الملف: احتمالات Stage 2 على كل الـ Validation} لكل موديل محفوظ"""
    paths = find_models("best_severity_model*.keras")
    all_ds = make_dataset(X_val, np.arange(len(y_val_full)), y_val_full)
    return {os.path.basename(p): tf.keras.models.load_model(p).predict(all_ds, verbose=0) for p in paths}


def compare_stage2_models():
    probs = predict_all_stage2()
    if not probs:
        print("مفيش موديلات Stage 2 محفوظة")
        return None
    print(f"Stage 1 threshold = {THR} | عدد موديلات Stage 2 اللي اتلقت: {len(probs)}")
    for name, p2 in probs.items():
        report_stage2(name, p2)
    return probs


STAGE2_PROBS = None
if P1 is None:
    print("شغّلوا Cell 8 وCell 9 الأول")
else:
    STAGE2_PROBS = compare_stage2_models()


# ============================== Cell 12 ==============================
# Ensemble: متوسط احتمالات كل موديلات Stage 2 المحفوظة (من غير أي تدريب جديد)
ENSEMBLE_NAMES = None   # None = كل الموديلات | أو قايمة أسماء، مثال: ["best_severity_model.keras", "best_severity_model_eyepacs_ft.keras"]
chosen = {k: v for k, v in (STAGE2_PROBS or {}).items() if ENSEMBLE_NAMES is None or k in ENSEMBLE_NAMES}
if len(chosen) < 2:
    print("محتاجين Cell 11 يكون اشتغل وفيه موديلين Stage 2 على الأقل (راجعوا ENSEMBLE_NAMES)")
else:
    ens = np.mean(list(chosen.values()), axis=0)
    report_stage2(f"Ensemble ({len(chosen)} موديلات: {', '.join(chosen)})", ens)


# ============================== Cell 13 ==============================
# شكل الصور اللي Stage 1 فوّتها (Grade 2+ عند THR) - للـ Error Analysis في التقرير
_DARK = {
    'figure.facecolor': 'black', 'axes.facecolor': 'black', 'savefig.facecolor': 'black',
    'text.color': 'white', 'axes.labelcolor': 'white', 'axes.edgecolor': 'white',
    'axes.titlecolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white',
}


def show_missed_images(min_grade=2, max_images=6):
    ids = np.where((y_val_full >= min_grade) & (P1 < THR))[0][:max_images]
    if len(ids) == 0:
        print("مفيش حالات فايتة")
        return
    with plt.rc_context(_DARK):
        fig, axes = plt.subplots(1, len(ids), figsize=(4 * len(ids), 4.5))
        axes = np.atleast_1d(axes)
        for ax, i in zip(axes, ids):
            ax.imshow(np.asarray(X_val[i]))
            ax.set_title(f"idx={i} | grade={y_val_full[i]} | p1={P1[i]:.3f}", fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        plt.show()


if P1 is None:
    print("شغّلوا Cell 8 الأول")
else:
    show_missed_images()


# ============================== Cell 14 ==============================
# حفظ احتمالات الـ Validation (بتخلي باقي الفريق يشتغل على Confidence Analysis / MLflow / الاختبارات من غير GPU)
def save_val_probs(path="/kaggle/working/val_probs_v1.npz"):
    if P1 is None:
        print("شغّلوا Cell 8 الأول")
        return
    np.savez_compressed(path, p1=P1, p2=P2, y_true=y_val_full, threshold=THR,
                        stage2_model=os.path.basename(STAGE2_PATH))
    print("اتحفظ:", path, "| الموديل:", os.path.basename(STAGE2_PATH))


save_val_probs()
