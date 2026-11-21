# دليل تظبيط الداتا (Data Preprocessing) - خطوة بخطوة
## مشروع RetinoGrade AI - Diabetic Retinopathy Grading

هدف المستند ده: تطلعوا في الآخر بداتا **نضيفة، موحدة، ومجهزة** جاهزة تتغذي مباشرة لأي موديل (ML كلاسيكي أو CNN). كل خطوة فيها الكود، وإيه اللي المفروض يحصل، وليه بنعملها.

---

## الخطوة 0: الإعدادات الأساسية (Imports)

حطوا الكود ده في أول خلية في الـ Notebook، وشغلوها مرة واحدة بس في بداية الشغل:

```python
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

# المسارات الأساسية
BASE_DIR = '/kaggle/input/competitions/aptos2019-blindness-detection/'
TRAIN_IMG_DIR = BASE_DIR + 'train_images/'
TRAIN_CSV = BASE_DIR + 'train.csv'

df = pd.read_csv(TRAIN_CSV)
print(f"عدد الصور الكلي: {len(df)}")
df.head()
```

**إيه المفروض يحصل:** يطلعلكوا `عدد الصور الكلي: 3662` والجدول بأول 5 صفوف.

---

## الخطوة 1: فحص الجودة الأولي (Quality Audit)

قبل ما تلمسوا صورة واحدة بالمعالجة، لازم تعرفوا المشاكل الموجودة فعلياً في الداتا.

### 1.1 فحص الصور التالفة أو المفقودة
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
```
**إيه المفروض يحصل:** المتوقع يطلع **0 و0** (داتا APTOS نضيفة عادةً)، لكن لازم تتأكدوا بأنفسكوا مش تفترضوا.

**لو طلع رقم غير صفر:** احذفوا الـ id_codes دي من الـ `df` قبل ما تكملوا:
```python
df = df[~df['id_code'].isin(broken_images + missing_images)].reset_index(drop=True)
```

### 1.2 فحص أحجام الصور (Dimensions)
```python
import random
sample_ids = random.sample(list(df['id_code']), 15)
sizes = []

for id_code in sample_ids:
    img = Image.open(TRAIN_IMG_DIR + id_code + '.png')
    sizes.append(img.size)

for id_code, size in zip(sample_ids, sizes):
    print(f"{id_code}: {size}")
```
**إيه المفروض يحصل:** هتلاقوا أحجام **مختلفة تماماً** (زي 2144×1424، 3388×2588...). ده طبيعي 100% في صور طبية حقيقية، ومعناه إن خطوة الـ Resize في الآخر **إجبارية** مش اختيارية.

### 1.3 فحص الإضاءة (Brightness Check)
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
**إيه المفروض يحصل:** هتلاقوا تفاوت واضح - بعض الصور مظلمة (قيم قريبة من 40-60) وبعضها أوضح (قيم 100+). ده سبب استخدام CLAHE في الخطوة 3.

### 1.4 توثيق النتايج
افتحوا خلية Markdown جديدة في الـ Notebook واكتبوا فيها بأسلوبكوا:
- عدد الصور التالفة/المفقودة اللي لقيتوها.
- ملاحظة إن الأحجام مش موحدة.
- ملاحظة إن فيه تفاوت في الإضاءة.

ده هيبقى أساس فقرة "Data Quality Findings" في تقريركوا النهائي.

---

## الخطوة 2: التقسيم الأول - قبل أي معالجة (Train/Validation Split)

**ليه بنقسم الداتا الأول قبل المعالجة؟** عشان نضمن إن أي حاجة نتعلمها من الداتا (زي حساب متوسط أو تطبيق تقنية معينة) متتعملش على بيانات الـ Validation، وده بيحافظ على نزاهة التقييم لاحقاً.

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
**إيه المفروض يحصل:** النسب المئوية للفئات في Train وValidation لازم تكون **متقاربة جداً** من بعض (بفضل `stratify`). لو مش متقاربة، فيه غلطة في الكود.

⚠️ **قاعدة ذهبية:** `random_state=42` ده رقم ثابت، **كل أعضاء الفريق يستخدموا نفس الرقم بالظبط**، عشان الكل يشتغل على نفس التقسيم ونقدر نقارن نتايجنا ببعض بشكل عادل.

---

## الخطوة 3: بناء دالة المعالجة (Preprocessing Function)

دي أهم دالة في المشروع كله - هتتطبق على كل صورة قبل ما تدخل للموديل.

```python
def preprocess_image(img_path, target_size=224):
    # 1. قراءة الصورة
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. إزالة الحواف السودا (Auto-cropping)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        img = img[y:y+h, x:x+w]
    
    # 3. توحيد المقاس
    img = cv2.resize(img, (target_size, target_size))
    
    # 4. تحسين التباين بـ CLAHE (على قناة الإضاءة فقط)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 5. تطبيع القيم (Normalization) لمدى 0-1
    img = img.astype(np.float32) / 255.0
    
    return img
```

### اختبروا الدالة على صورة واحدة قبل ما تطبقوها على كل الداتا
```python
sample_id = train_df['id_code'].iloc[0]
original = np.array(Image.open(TRAIN_IMG_DIR + sample_id + '.png'))
processed = preprocess_image(TRAIN_IMG_DIR + sample_id + '.png')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(original)
axes[0].set_title('قبل المعالجة')
axes[0].axis('off')

axes[1].imshow(processed)
axes[1].set_title('بعد المعالجة (Crop + Resize + CLAHE)')
axes[1].axis('off')

plt.show()
```
**إيه المفروض يحصل:** الصورة التانية المفروض تبقى **أصغر (من غير حواف سودا زيادة)**، **مقاس موحد**، و**الأوعية الدموية أوضح** من الصورة الأصلية. لو الفرق مش واضح بصرياً، جربوا صورة من فئة أعلى (2 أو 3) عشان الفرق يبان أكتر.

⚠️ **لو الدالة رمت خطأ (Error):** جربوها على 3-4 صور مختلفة الأول قبل ما تطبقوها على الداتا كلها، عشان تتأكدوا إنها مستقرة (بعض الصور النادرة ممكن الـ auto-cropping يفشل معاها لو الخلفية مش سودا تماماً).

---

## الخطوة 4: تطبيق المعالجة على كل الداتا

### ⚠️ قرار مهم قبل ما تبدأوا: هل تحملوا كل الصور في الذاكرة، ولا تستخدموا Generator؟

**الخيار البسيط (لمستواكوا الحالي، ينفع مع 3,662 صورة):**
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

**إيه المفروض يحصل:** الكود هياخد وقت (دقايق معدودة لعدد الصور ده)، وهيطلعلكوا تحديثات كل 500 صورة. في الآخر:
- `X_train.shape` المفروض يطلع حاجة زي `(2929, 224, 224, 3)`.
- `X_val.shape` المفروض يطلع حاجة زي `(733, 224, 224, 3)`.

⚠️ **لو ظهر Memory Error:** يبقى لازم نستخدم طريقة تانية (Data Generator بيحمل الصور على دفعات بدل كلها مرة واحدة) - في الحالة دي ارجعولي وأديكوا الكود البديل، ده وارد يحصل لو كان الـ RAM المتاح في Kaggle مش كافي.

---

## الخطوة 5: معالجة مشكلة عدم توازن الفئات (Class Imbalance)

فاكرين إن فئة 0 عندها 1,805 صورة وفئة 3 عندها 193 بس؟ دلوقتي هنتعامل مع المشكلة دي.

### 5.1 احسبوا الـ Class Weights (هتستخدموها وقت التدريب لاحقاً)
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("أوزان الفئات:", class_weight_dict)
```
**إيه المفروض يحصل:** هتلاقوا الفئات النادرة (3، 4) واخدة أوزان أعلى بكتير من فئة 0. الديكشنري ده هتستخدموه بعدين مباشرة وقت `model.fit()`.

### 5.2 Data Augmentation للفئات النادرة بس (اختياري بس موصى بيه بقوة)
```python
def augment_image(img):
    augmented = []
    
    # دوران
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (224, 224))
        augmented.append(rotated)
    
    # انعكاس أفقي
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)
    
    return augmented

# طبقوا الـ Augmentation بس على صور الفئات 3 و4 في train_df
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

print(f"حجم بيانات التدريب قبل الـ Augmentation: {X_train.shape[0]}")
print(f"حجم بيانات التدريب بعد الـ Augmentation: {X_train_augmented.shape[0]}")
```
**إيه المفروض يحصل:** عدد صور فئتي 3 و4 المفروض يزيد (كل صورة بقت 4 نسخ إضافية: 3 دورانات + انعكاس واحد)، وده هيقلل نسبة عدم التوازن شوية (مش هيحلها 100% بس هيحسنها).

⚠️ **ملحوظة مهمة:** الـ Augmentation ده يتطبق على **Train فقط**، خالص متطبقوهوش على `X_val` أو `y_val` - الـ Validation لازم يفضل زي الواقع الحقيقي من غير تعديل.

---

## الخطوة 6: التحقق النهائي قبل الانتقال لبناء الموديل

### Checklist قبل ما تقولوا "الداتا جاهزة"
```python
print("=== ملخص نهائي ===")
print(f"شكل X_train: {X_train_augmented.shape}")
print(f"شكل y_train: {y_train_augmented.shape}")
print(f"شكل X_val: {X_val.shape}")
print(f"شكل y_val: {y_val.shape}")
print(f"مدى قيم البكسل (المفروض يكون بين 0 و1): {X_train_augmented.min()} - {X_train_augmented.max()}")
print(f"عدد الفئات الفريدة: {np.unique(y_train_augmented)}")

# عرض 5 صور عشوائية بعد كل المعالجة للتأكد البصري
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
random_indices = random.sample(range(len(X_train_augmented)), 5)
for i, idx in enumerate(random_indices):
    axes[i].imshow(X_train_augmented[idx])
    axes[i].set_title(f"Grade {y_train_augmented[idx]}")
    axes[i].axis('off')
plt.show()
```

### قائمة التأكد النهائية (لازم كل بند يكون ✅ قبل ما تنتقلوا لبناء الموديل)
- [ ] مفيش صور تالفة أو مفقودة.
- [ ] كل الصور بقت نفس المقاس (224×224).
- [ ] قيم البكسل بين 0 و1 (Normalized).
- [ ] الـ Train والـ Validation متقسمين بـ stratify وبنفس الـ random_state.
- [ ] الـ Augmentation اتطبق على Train بس، مش على Validation.
- [ ] عندكوا Class Weights جاهزة للاستخدام وقت التدريب.
- [ ] شفتوا عينة من الصور بعد المعالجة بعنيكوا وشكلها منطقي (مش سودا أو تالفة).

---

## إيه اللي بعد كده؟

لما كل بنود الـ Checklist تبقى ✅، يبقى الداتا رسمياً **جاهزة**، وتقدروا تنتقلوا لمرحلة بناء أول موديل (Baseline). متبدأوش في بناء الموديل قبل ما تتأكدوا من كل بند في القائمة دي - أي مشكلة في الداتا هنا هتتضاعف مشاكلها في مرحلة التدريب.

ارجعولي لما تخلصوا الخطوات دي (حتى لو واجهتوكوا مشكلة في أي خطوة) وهنكمل سوا لمرحلة بناء الموديل بنفس مستوى التفصيل ده.
