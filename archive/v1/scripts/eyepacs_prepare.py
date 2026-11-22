# =====================================================================
# RetinoGrade AI - تجهيز داتا EyePACS (35k) مرة واحدة
#   - Split على مستوى المريض (عين يمين وشمال لنفس المريض في نفس الـ split)
#   - نفس دالة المعالجة النهائية (Crop + Padding + CLAHE) بس بتحفظ uint8 224x224
# شغّلوه في Notebook منفصل (CPU كفاية - مش محتاج GPU)
# كل "Cell" تحت تتنسخ في خلية منفصلة
# =====================================================================


# ============================== Cell 1 ===============================
import os, random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
random.seed(SEED); np.random.seed(SEED)

BASE_DIRS = [
    '/kaggle/input/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized/colored_images/colored_images/',
    '/kaggle/input/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized/colored_images/',
]
OUT_DIR = '/kaggle/working/eyepacs_224'
SIZE = 224
LIMIT = None        # للاختبار السريع: حطوا رقم زي 400 (بيشتغل على عينة عشوائية بس)
N_WORKERS = os.cpu_count() or 2
GRADE_OF = {'No_DR': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Proliferate_DR': 4}

plt.rcParams.update({
    'figure.facecolor': 'black', 'axes.facecolor': 'black', 'savefig.facecolor': 'black',
    'text.color': 'white', 'axes.labelcolor': 'white', 'axes.edgecolor': 'white',
    'axes.titlecolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white',
})
print("CPU workers:", N_WORKERS)


# ============================== Cell 2 ===============================
# فهرسة كل الصور (المسار + الدرجة + رقم المريض)
base_dir = None
for d in BASE_DIRS:
    if os.path.isdir(d) and any(c in os.listdir(d) for c in GRADE_OF):
        base_dir = d
        break
assert base_dir is not None, "EyePACS folder not found - check BASE_DIRS"
print("Base dir:", base_dir)

rows = []
for cls, grade in GRADE_OF.items():
    cls_dir = os.path.join(base_dir, cls)
    if not os.path.isdir(cls_dir):
        continue
    for f in sorted(os.listdir(cls_dir)):
        if f.lower().endswith(('.png', '.jpeg', '.jpg', '.tif')):
            stem = os.path.splitext(f)[0]            # مثال: 10003_right
            rows.append((os.path.join(cls_dir, f), grade, stem.split('_')[0], stem))

meta = pd.DataFrame(rows, columns=['path', 'grade', 'patient', 'stem'])
if LIMIT:
    meta = meta.sample(LIMIT, random_state=SEED).reset_index(drop=True)

print("Images:", len(meta), "| patients:", meta['patient'].nunique())
print("Example names:", meta['stem'].head(3).tolist())
print("Share of numeric patient IDs (should be close to 1.0):", round(float(meta['patient'].str.isdigit().mean()), 3))
print(meta['grade'].value_counts().sort_index())


# ============================== Cell 3 ===============================
# Split على مستوى المريض: 80% train / 10% val / 10% test (Stratified بالدرجة)
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
split = np.zeros(len(meta), dtype=np.int8)           # 0=train 1=val 2=test
for fold, (_, te_idx) in enumerate(sgkf.split(meta, meta['grade'], groups=meta['patient'])):
    if fold == 0:
        split[te_idx] = 2
    elif fold == 1:
        split[te_idx] = 1
meta['split'] = split

# تأكيد: مفيش مريض في أكتر من split
for a, b in [(0, 1), (0, 2), (1, 2)]:
    overlap = set(meta.patient[meta.split == a]) & set(meta.patient[meta.split == b])
    assert not overlap, f"Patient leakage between splits {a} and {b}: {len(overlap)}"
print("✅ No patient leakage between splits")
print(pd.crosstab(meta['split'].map({0: 'train', 1: 'val', 2: 'test'}), meta['grade']))


# ============================== Cell 4 ===============================
# دالة المعالجة النهائية (نفس اللي استخدمتوها في APTOS)
def preprocess_image(img_path, target_size=224):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # المرحلة الأولى: شيل الخلفية السودا الخارجية
    _, thresh_outer = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_outer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        img = img[y:y+h, x:x+w]

    original_area_after_stage1 = img.shape[0] * img.shape[1]

    # المرحلة الثانية: شيل أي إطار أبيض (بشرط أمان: متشيلش أكتر من 30% من المساحة)
    gray2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask_retina = cv2.inRange(gray2, 15, 230)
    contours2, _ = cv2.findContours(mask_retina, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours2:
        largest_contour2 = max(contours2, key=cv2.contourArea)
        x2, y2, w2, h2 = cv2.boundingRect(largest_contour2)
        if w2 * h2 >= 0.7 * original_area_after_stage1:
            img = img[y2:y2+h2, x2:x2+w2]

    # Padding لمربع قبل الـ Resize (يمنع التمطيط)
    h, w = img.shape[:2]
    size = max(h, w)
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left
    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])

    img = cv2.resize(img, (target_size, target_size))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img = img.astype(np.float32) / 255.0
    return img


def to_uint8(path):
    """بتعالج صورة وترجّعها uint8 (0-255). لو فشلت بترجّع صورة سودا وعلامة False"""
    try:
        img = preprocess_image(path, SIZE)
        return (img * 255.0 + 0.5).astype(np.uint8), True
    except Exception:
        return np.zeros((SIZE, SIZE, 3), np.uint8), False


# ============================== Cell 5 ===============================
# المعالجة الفعلية (متوازية على كل الـ CPUs). ياخد وقت حسب عدد الصور، سيبوه يشتغل
os.makedirs(OUT_DIR, exist_ok=True)
X = np.lib.format.open_memmap(os.path.join(OUT_DIR, 'X.npy'), mode='w+',
                              dtype=np.uint8, shape=(len(meta), SIZE, SIZE, 3))
ok = np.zeros(len(meta), dtype=bool)

with mp.get_context('fork').Pool(N_WORKERS) as pool:
    for i, (arr, good) in enumerate(pool.imap(to_uint8, meta['path'].tolist(), chunksize=32)):
        X[i] = arr
        ok[i] = good
        if i % 2000 == 0:
            print(f"Processed {i} / {len(meta)}")
X.flush()
meta['ok'] = ok

meta.to_csv(os.path.join(OUT_DIR, 'meta.csv'), index=False)
print(f"✅ Done: {int(ok.sum())} valid images out of {len(meta)} | failed: {int((~ok).sum())}")
print("X.npy size:", round(os.path.getsize(os.path.join(OUT_DIR, 'X.npy')) / 1024**3, 2), "GB")


# ============================== Cell 6 ===============================
# تأكد بصرياً: 10 صور عشوائية بعد المعالجة
Xr = np.load(os.path.join(OUT_DIR, 'X.npy'), mmap_mode='r')
ids = np.random.RandomState(SEED).choice(len(meta), min(10, len(meta)), replace=False)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, i in zip(axes.ravel(), ids):
    ax.imshow(np.asarray(Xr[i]))
    ax.set_title(f"grade={meta.grade[i]} | split={meta.split[i]}", fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.show()

print("\nNext: Save Version, or download the eyepacs_224 folder from Output and upload it as a Dataset (suggested name: eyepacs-224-uint8)")
