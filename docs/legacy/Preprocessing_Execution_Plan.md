# خطة تنفيذ تظبيط الداتا - مرتبطة بمهام Notion (T01 → T03)
## مشروع RetinoGrade AI

هنا كل خطوات تظبيط الداتا، بس دلوقتي مقسّمة ومرتبة بالظبط تحت كل Task ID زي ما هي في Notion، عشان كل عضو يعرف بالظبط هو بيشتغل على أنهي جزء وإمتى يقدر يحط "Done" على المهمة بتاعته.

⚠️ **تغيير مهم عن الدليل الأول:** حساب **Class Weights اتشال من هنا بالكامل** ونقل لـ Task T08 (Imbalance Handling) اللي هتيجي بعدين في M2. هنا هنركز بس على تنضيف الصور والـ Augmentation الفيزيائي.

---

## T01 — Dataset Collection, EDA and Quality
**المسؤول:** Omar Khaled | **Depends on:** Independent

### الخطوة 0: الإعدادات الأساسية
```python
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

BASE_DIR = '/kaggle/input/competitions/aptos2019-blindness-detection/'
TRAIN_IMG_DIR = BASE_DIR + 'train_images/'
TRAIN_CSV = BASE_DIR + 'train.csv'

df = pd.read_csv(TRAIN_CSV)
print(f"عدد الصور الكلي: {len(df)}")
df.head()
```
**إيه المفروض يحصل:** `عدد الصور الكلي: 3662`.

### الخطوة 1.1: فحص الصور التالفة أو المفقودة
```python
broken_images = []
missing_images = []

for id_code in df['id_code']:
    path = TRAIN_IMG_DIR + id_code + '.png'
    if not os.path.exists(path):
        missing_images.append(id_code)
        continue
    try:
        img = Image.open(path)
        img.verify()
    except Exception:
        broken_images.append(id_code)

print(f"صور مفقودة: {len(missing_images)}")
print(f"صور تالفة: {len(broken_images)}")

# لو طلع رقم غير صفر، احذفوهم:
df = df[~df['id_code'].isin(broken_images + missing_images)].reset_index(drop=True)
```

### الخطوة 1.2: فحص الصور المكررة
```python
duplicate_count = df['id_code'].duplicated().sum()
print(f"عدد الصور المكررة: {duplicate_count}")
```

### الخطوة 1.3: فحص أحجام الصور
```python
import random
sample_ids = random.sample(list(df['id_code']), 15)

for id_code in sample_ids:
    img = Image.open(TRAIN_IMG_DIR + id_code + '.png')
    print(f"{id_code}: {img.size}")
```
**إيه المفروض يحصل:** أحجام مختلفة تماماً - ده طبيعي، ومعناه Resize هيبقى إجباري في T03.

### الخطوة 1.4: فحص الإضاءة
```python
brightness_values = []
for id_code in sample_ids:
    img = np.array(Image.open(TRAIN_IMG_DIR + id_code + '.png').convert('L'))
    brightness_values.append(img.mean())

plt.figure(figsize=(8,4))
plt.bar(range(len(sample_ids)), brightness_values)
plt.axhline(y=np.mean(brightness_values), color='r', linestyle='--', label='المتوسط العام')
plt.ylabel('متوسط الإضاءة (0-255)')
plt.title('تفاوت الإضاءة بين الصور')
plt.legend()
plt.show()
```

### الخطوة 1.5: تأكيد وتوثيق توزيع الفئات (Class Imbalance)
```python
print(df['diagnosis'].value_counts().sort_index())

imbalance_ratio = df['diagnosis'].value_counts().max() / df['diagnosis'].value_counts().min()
print(f"نسبة عدم التوازن: {imbalance_ratio:.1f}x")
```

### الخطوة 1.6: التوثيق (في خلية Markdown داخل الـ Notebook)
اكتبوا بأسلوبكوا:
- عدد الصور التالفة/المفقودة/المكررة اللي لقيتوها.
- ملاحظة إن الأحجام مش موحدة.
- ملاحظة تفاوت الإضاءة.
- نسبة عدم التوازن بين الفئات.

### ✅ Done when (T01):
"Class distribution, missing/invalid images checked and documented" → يتحقق لما الخطوات 1.1 لـ 1.6 تخلص وتتوثق.

---

## T02 — Dataset Splitting
**المسؤول:** Omar Ashour | **Depends on:** T01

```python
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['diagnosis']
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"Train: {len(train_df)} صورة")
print(f"Validation: {len(val_df)} صورة")

print("\nتوزيع الفئات في Train:")
print(train_df['diagnosis'].value_counts(normalize=True).sort_index())
print("\nتوزيع الفئات في Validation:")
print(val_df['diagnosis'].value_counts(normalize=True).sort_index())
```
**إيه المفروض يحصل:** النسب المئوية للفئات في Train وValidation متقاربة جداً من بعض.

⚠️ **قاعدة ذهبية:** `random_state=42` ثابت لكل الفريق - محدّش يغيّره.

### ✅ Done when (T02):
"Train, Validation, Test split and proportional class distribution confirmed" → يتحقق لما الأرقام فوق تتطبع وتتأكد النسب متقاربة.

---

## T03 — Image Preprocessing & Augmentation
**المسؤولين:** Omar Khaled, Omar Ashour, Ahmed Samy | **Depends on:** T02

### الخطوة 3.1: بناء دالة المعالجة
```python
def preprocess_image(img_path, target_size=224):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        img = img[y:y+h, x:x+w]

    img = cv2.resize(img, (target_size, target_size))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img = img.astype(np.float32) / 255.0
    return img
```

### الخطوة 3.2: اختبار الدالة على صورة واحدة (قبل التطبيق الشامل)
```python
sample_id = train_df['id_code'].iloc[0]
original = np.array(Image.open(TRAIN_IMG_DIR + sample_id + '.png'))
processed = preprocess_image(TRAIN_IMG_DIR + sample_id + '.png')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(original)
axes[0].set_title('قبل المعالجة')
axes[0].axis('off')
axes[1].imshow(processed)
axes[1].set_title('بعد المعالجة')
axes[1].axis('off')
plt.show()
```

### الخطوة 3.3: تطبيق المعالجة على كل الداتا
```python
def load_dataset(dataframe, img_dir, target_size=224):
    images = []
    labels = []
    for idx, row in dataframe.iterrows():
        img_path = img_dir + row['id_code'] + '.png'
        img = preprocess_image(img_path, target_size)
        images.append(img)
        labels.append(row['diagnosis'])
        if idx % 500 == 0:
            print(f"تمت معالجة {idx} صورة...")
    return np.array(images), np.array(labels)

X_train, y_train = load_dataset(train_df, TRAIN_IMG_DIR)
X_val, y_val = load_dataset(val_df, TRAIN_IMG_DIR)

print(f"شكل بيانات التدريب: {X_train.shape}")
print(f"شكل بيانات التحقق: {X_val.shape}")
```
⚠️ لو ظهر Memory Error، وقفوا وابعتولي - هنستخدم Data Generator بدل التحميل الكامل.

### الخطوة 3.4: Data Augmentation الفيزيائي للفئات النادرة (Grade 3 و4)
```python
def augment_image(img):
    augmented = []
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (224, 224))
        augmented.append(rotated)
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)
    return augmented

minority_mask = (y_train == 3) | (y_train == 4)
minority_images = X_train[minority_mask]
minority_labels = y_train[minority_mask]

augmented_images = []
augmented_labels = []
for img, label in zip(minority_images, minority_labels):
    aug_versions = augment_image(img)
    augmented_images.extend(aug_versions)
    augmented_labels.extend([label] * len(aug_versions))

X_train_augmented = np.concatenate([X_train, np.array(augmented_images)])
y_train_augmented = np.concatenate([y_train, np.array(augmented_labels)])

print(f"حجم Train قبل الـ Augmentation: {X_train.shape[0]}")
print(f"حجم Train بعد الـ Augmentation: {X_train_augmented.shape[0]}")
```
⚠️ الـ Augmentation يتطبق على Train فقط - خالص متلمسوش X_val وy_val.

**ملحوظة مهمة:** ده الـ Augmentation الفيزيائي بس (تدوير/عكس). أما حساب **Class Weights** (اللي كان في الدليل الأول) فبقى مؤجل لـ **T08 - Imbalance Handling** بعدين، مش جزء من T03.

### الخطوة 3.5: حفظ الداتا المعالجة (لتوفير وقت لاحقاً)
```python
np.save('/kaggle/working/X_train.npy', X_train_augmented)
np.save('/kaggle/working/y_train.npy', y_train_augmented)
np.save('/kaggle/working/X_val.npy', X_val)
np.save('/kaggle/working/y_val.npy', y_val)
```

### الخطوة 3.6: التحقق النهائي - Checklist
```python
print("=== ملخص نهائي ===")
print(f"شكل X_train: {X_train_augmented.shape}")
print(f"شكل y_train: {y_train_augmented.shape}")
print(f"شكل X_val: {X_val.shape}")
print(f"شكل y_val: {y_val.shape}")
print(f"مدى قيم البكسل: {X_train_augmented.min()} - {X_train_augmented.max()}")

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
random_indices = random.sample(range(len(X_train_augmented)), 5)
for i, idx in enumerate(random_indices):
    axes[i].imshow(X_train_augmented[idx])
    axes[i].set_title(f"Grade {y_train_augmented[idx]}")
    axes[i].axis('off')
plt.show()
```

### ✅ Done when (T03):
"Cropping, resizing, clahe, normalization applied and verified" → يتحقق لما القائمة دي كلها ✅:
- [ ] مفيش صور تالفة أو مفقودة (متحقق من T01).
- [ ] كل الصور بقت 224×224.
- [ ] قيم البكسل بين 0 و1.
- [ ] الـ Augmentation اتطبق على Train بس.
- [ ] عينة الصور بعد المعالجة شكلها منطقي بصرياً.
- [ ] الداتا اتحفظت كـ `.npy` files.

---

## ملخص الترتيب العملي للفريق

| الترتيب | مين | يعمل إيه | Task |
|---|---|---|---|
| 1 | Omar Khaled | فحص الجودة الكامل + توثيق | T01 |
| 2 | Omar Ashour | تقسيم Train/Validation | T02 |
| 3 | Omar Khaled + Omar Ashour + Ahmed Samy | دالة المعالجة + تطبيقها + Augmentation فيزيائي + حفظ | T03 |
| 4 (لاحقاً، مش دلوقتي) | Yehia Salah + Omar Ibrahim + Eyad Tamer | حساب Class Weights / Focal Loss | T08 |

**متبدأوش T03 قبل ما T02 يخلص، ومتبدأوش T02 قبل ما T01 يوثّق نتايجه بالكامل** - كل واحدة فيهم مبنية على اللي قبلها بشكل حقيقي مش شكلي.
