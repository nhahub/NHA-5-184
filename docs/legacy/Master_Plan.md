# خطة العمل الشاملة - مشروع اعتلال الشبكية السكري
## Intelligent Diabetic Retinopathy Grading and Quality Assurance System

---

## 0. قبل أي حاجة - إعداد الفريق (الأسبوع 1)

### 0.1 حسابات وأدوات لازم تتعملها
- [ ] كل عضو يعمل حساب على **kaggle.com** (Register → تفعيل Phone Verification لو طُلب).
- [ ] اعملوا **GitHub repository** مشترك للفريق (اسم مقترح: `dr-grading-depi-r5`)، وضيفوا كل الأعضاء كـ Collaborators. هنا هيتحط الكود كله، مش على أجهزة شخصية متفرقة.
- [ ] اعملوا **مستند مشترك** (Google Docs أو Notion) لتسجيل النتائج والملاحظات أولاً بأول (Accuracy, QWK، مشاكل واجهتكوا).
- [ ] حددوا **قناة تواصل ثابتة** (WhatsApp/Discord group) لتحديثات يومية سريعة.

### 0.2 توزيع الأدوار المبدئي (تقدروا تعدلوا حسب رغبتكوا)
| العضو | المسؤولية الأساسية |
|---|---|
| Omar Khaled (Team Leader) | تنسيق عام + Data Collection & Preprocessing |
| Omar Ashour | Quality Model (EyeQ) |
| Eyad Tamer | Core ML/DL Model (التصنيف الأساسي) |
| Omar Ibrahim | Explainability (Grad-CAM) + IDRiD Validation |
| Ahmed Samy | Azure + MLOps (MLflow) |
| Yahia Salah | Streamlit UI + التوثيق والتقرير النهائي |

### 0.3 بيئة العمل
- [ ] ابدأوا بـ **Kaggle Notebooks** (مش Colab) — الداتا بتبقى متاحة تلقائياً وفيه GPU مجاني (~30 ساعة أسبوعياً).
- [ ] كل عضو يفعّل **GPU Accelerator** من Settings داخل الـ Notebook وقت التدريب بس (مش وقت كتابة الكود).
- [ ] لو خلصتوا ساعات GPU بتاعة Kaggle، استخدموا **Google Colab** كمصدر إضافي (منفصل تماماً وله حد شهري خاص).

---

## 1. المرحلة 1: جمع واستكشاف البيانات (الأسبوع 1-2)

### 1.1 الداتا الأساسية - ابدأوا بيها فوراً
**APTOS 2019 Blindness Detection**
- الرابط: https://www.kaggle.com/c/aptos2019-blindness-detection/data
- الخطوات: افتحوا اللينك → "Join Competition" → وافقوا على الشروط → Code → New Notebook.
- تأكيد إن الداتا وصلت:
```python
import os
print(os.listdir('/kaggle/input/aptos2019-blindness-detection/'))
```

### 1.2 استكشاف الداتا (EDA) - قبل أي كود تدريب
```python
import pandas as pd
df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')
print(df['diagnosis'].value_counts())
```
- [ ] اعرضوا صورة واحدة من كل فئة (0 إلى 4) بجانب بعض.
- [ ] وثّقوا في الـ Notebook: عدد الصور، توزيع الفئات، ملاحظاتكوا البصرية.
- [ ] **لاحظوا واكتبوا رسمياً:** فئة "0" (سليم) أكتر بكتير من الفئات التانية — ده أساس مشكلة الـ Class Imbalance اللي هتتعاملوا معاها في المرحلة 3.

### 1.3 تقسيم الداتا (مهم جداً - اعملوه صح من الأول)
- [ ] Train/Validation/Test split (مقترح 70/15/15).
- [ ] **ثبّتوا الـ random seed** (مثلاً `random_state=42`) عشان كل عضو يشتغل على نفس التقسيم بالظبط.
- [ ] لو استخدمتوا EyePACS لاحقاً: **Patient-level split إجباري** — صورة عين يمين وشمال لنفس المريض لازم تبقى كلها في نفس الـ Split (مش نص في train ونص في test) عشان منمنعش Data Leakage.

---

## 2. المرحلة 2: نموذج أولي - Baseline (الأسبوع 3-4)

- [ ] اكتبوا CNN بسيط جداً (2-3 conv layers) من الصفر — الهدف إثبات إن الـ pipeline كامل (تحميل → تدريب → تقييم) شغال، مش الدقة.
- [ ] قيّموا بـ Accuracy + Confusion Matrix.
- [ ] وثّقوا النتيجة كمرجع (baseline) هتقارنوا بيه أي تحسين لاحق.
- [ ] ارفعوا الكود على الـ GitHub repo فور ما يشتغل.

---

## 3. المرحلة 3: Transfer Learning + حل مشكلة عدم التوازن (الأسبوع 5-8)

### 3.1 Transfer Learning
- [ ] جربوا **EfficientNet** أو **ResNet50** pretrained (من `torchvision` أو `tensorflow.keras.applications`).
- [ ] فعّلوا Fine-tuning على الطبقات الأخيرة الأول.
- [ ] قارنوا بالـ baseline — المفروض يبقى أحسن، لو مش أحسن فيه حاجة غلط في الكود.

### 3.2 مقياس التقييم الرسمي
- [ ] طبّقوا **Quadratic Weighted Kappa (QWK)** — ده المقياس المعتمد رسمياً في مسابقة APTOS نفسها:
```python
from sklearn.metrics import cohen_kappa_score
qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
```

### 3.3 معالجة Class Imbalance
- [ ] طبّقوا **Class Weights** في دالة الخسارة (وزن أعلى للفئات 3 و4).
- [ ] طبّقوا **Data Augmentation** (rotation, flip, zoom) على الفئات النادرة بس.
- [ ] قيّموا تاني بالـ QWK وشوفوا الـ Confusion Matrix — ركزوا هل تحسن الأداء على الفئات النادرة تحديداً مش بس الرقم الكلي.

### 3.4 المعالجة المسبقة للصور
- [ ] **CLAHE** (Contrast Limited Adaptive Histogram Equalization) بـ OpenCV لتحسين وضوح الأوعية الدموية.
- [ ] **Auto-cropping** لإزالة الحواف السوداء حول صورة الشبكية.

---

## 4. المرحلة 4: التوسع على EyePACS (اختياري - الأسبوع 9، لو الوقت سمح)

### 4.1 لينكات EyePACS
**النسخة الأصلية الكاملة (متستخدموهاش من البداية):**
- https://www.kaggle.com/c/diabetic-retinopathy-detection/data

**النسخة المُعاد تحجيمها (دي اللي هتحتاجوها):**
- https://www.kaggle.com/datasets/mohlamin/resized-eyepacs-diabetic-retinopathy-dataset
- https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized

### 4.2 خطوات الاستخدام
- [ ] من داخل Kaggle Notebook: "Add Input" → دوروا "resized eyepacs" → ضيفوه مباشرة (من غير تحميل يدوي).
- [ ] استخدموه كـ **اختبار إضافي فقط** (extra test set) للموديل المدرب على APTOS، مش لإعادة تدريب من الصفر.
- [ ] هدف الخطوة دي: قياس **قابلية التعميم** (Generalization) — هل الموديل شغال كويس على بيانات من مصدر مختلف؟

---

## 5. المرحلة 5: التفسير - Grad-CAM (الأسبوع 10-11)

### 5.1 تطبيق Grad-CAM
- [ ] طبّقوا Grad-CAM على الموديل النهائي المدرب.
- [ ] شوفوا الـ heatmap على صور من APTOS الأول للتأكد إنه بيشتغل صح.

### 5.2 لينك IDRiD (للتحقق من الدقة)
- **أسهل لينك (Kaggle):** https://www.kaggle.com/datasets/mohamedabdalkader/indian-diabetic-retinopathy-image-dataset-idrid
- **المصدر الرسمي:** https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid

### 5.3 خطوات المقارنة
- [ ] من "Add Input" ضيفوا IDRiD للـ Notebook.
- [ ] اختاروا كام صورة من فولدر **Segmentation** (فيه الـ masks الحقيقية من الأطباء).
- [ ] مرروها على موديلكوا، اطلعوا Grad-CAM heatmap.
- [ ] قارنوا **بصرياً** (مش رياضياً بالضرورة) هل الموديل بيبص في نفس أماكن الإصابات اللي الأطباء حددوها.

---

## 6. المرحلة 6: EyeQ - بوابة الجودة (اختياري - آخر أولوية)

### 6.1 لينك EyeQ
- **الصفحة الرسمية:** https://github.com/HzFu/EyeQ
- ملحوظة: الـ repo فيه labels بس (CSV)، مش صور — لازم تدمجوها مع صور EyePACS اللي نزلتوها في المرحلة 4.

### 6.2 خطوات الاستخدام
- [ ] `git clone https://github.com/HzFu/EyeQ.git`
- [ ] هتلاقوا `Label_EyeQ_train.csv` و `Label_EyeQ_test.csv` داخل فولدر `data/`.
- [ ] طابقوا أسماء الصور (image ID) في الـ CSV مع صور EyePACS.
- [ ] استخدموا سكريبت `EyeQ_process_main.py` الموجود في الـ repo للمعالجة.
- [ ] درّبوا موديل بسيط (ResNet18 مثلاً) يصنف الصورة: Good / Usable / Reject.

---

## 7. المرحلة 7: الواجهة والتجميع (الأسبوع 12-13)

- [ ] ابنوا واجهة **Streamlit**: رفع صورة → عرض (حالة الجودة، الدرجة 0-4، نسبة الثقة، Grad-CAM heatmap).
- [ ] ادمجوا كل الموديلات في pipeline واحد متصل.
- [ ] لو فيه وقت: استخدموا **MLflow** لتتبع تجاربكوا (تسجيل الـ hyperparameters والنتائج لكل تجربة).
- [ ] لو فيه وقت: انشروا الموديل على **Azure Machine Learning**.

---

## 8. المرحلة 8: التوثيق والتسليم النهائي (الأسبوع 14-16)

- [ ] اكتبوا التقرير النهائي (زي هيكل الـ Proposal بالظبط: Overview → كل Milestone بتفاصيله → Summary → Conclusion).
- [ ] وضّحوا فيه بالتحديد: التحديات اللي واجهتوها (Class Imbalance، جودة الصور، إلخ) والحلول اللي طبقتوها.
- [ ] جهّزوا العرض التقديمي (PPT) مع Demo حي للـ Streamlit app.
- [ ] راجعوا الـ GitHub repo: كود منظم، README واضح فيه خطوات التشغيل.

---

## 9. قاعدة ذهبية طول المشروع

**رتّبوا الأولويات بالترتيب ده لو ضاق الوقت في أي لحظة:**
1. موديل تصنيف كويس على APTOS شغال ومُقيّم صح بـ QWK (أساسي - لا غنى عنه).
2. معالجة Class Imbalance (أساسي).
3. Grad-CAM بصري بسيط (أساسي - ده اللي بيميز المشروع).
4. واجهة Streamlit (أساسي للعرض).
5. التوسع على EyePACS (بونص).
6. EyeQ + IDRiD بمقارنة رياضية دقيقة (IoU/Dice) (بونص - لو فضل وقت فعلاً).

**متنتقلوش لمرحلة جديدة إلا لو اللي قبلها شغال ومُختبر فعلاً.** مشروع بسيط شغال 100% أحسن بكتير من مشروع طموح نص شغال.

---

## 10. مرجع سريع لكل اللينكات

| المصدر | الرابط |
|---|---|
| APTOS 2019 | kaggle.com/c/aptos2019-blindness-detection/data |
| EyePACS (أصلية) | kaggle.com/c/diabetic-retinopathy-detection/data |
| EyePACS (resized) #1 | kaggle.com/datasets/mohlamin/resized-eyepacs-diabetic-retinopathy-dataset |
| EyePACS (resized) #2 | kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized |
| EyeQ | github.com/HzFu/EyeQ |
| IDRiD (Kaggle) | kaggle.com/datasets/mohamedabdalkader/indian-diabetic-retinopathy-image-dataset-idrid |
| IDRiD (رسمي) | ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid |
