# Component: image-quality gate (EyeQ: Good / Usable / Reject)

**In short:** a small model looks at the photo *before* it is graded and says **Good** (grade it), **Usable** (grade it, warn, ask for a human review) or **Reject** (ask for a new photo, show no grade). On 1,199 photos of patients it never saw it is **94.3% accurate** (macro-F1 0.906), it stops 89% of the Reject photos, and it refuses only 1.2% of the gradable ones. It matters: on Reject photos the grading model makes **4× as many far errors**.

Code: [`src/quality/quality.py`](../../src/quality/quality.py) (the gate), [`src/pipeline.py`](../../src/pipeline.py) (gate + grading), [`src/training/train_quality.py`](../../src/training/train_quality.py) (training).

## How to run
```python
from pipeline import RetinoPipeline
r = RetinoPipeline().predict("eye.png")          # weights from RG_MODEL and RG_QUALITY_MODEL or the default paths
r["quality"]["label"], r["quality"]["action"]    # ("Usable", "warn"); actions: ok | warn | retake
```
`quality` is an **additive** key: every existing key of the grading output stays the same. With action `retake` a user interface must show the message instead of the grade. The app does this (`docs/components/app.md`). Behaviour table: Good → ok; Usable → warn and human review; Reject → retake.

Standard usage needs the quality model file (17 MB, private Kaggle job output `rg-quality-train`, `quality_b0_224_s42.keras`; a copy without optimizer state is used locally as `models/quality_b0_224.keras`). Without it the app still runs and says the quality was not checked.

## The labels are attached to the right pictures (verified before training)
EyeQ (github.com/HzFu/EyeQ) labels are *names* of EyePACS 2015 images, not images. The mapping was verified in three steps before any training, on the 11,297 photos of the dev and rest patients:
1. Names: all 11,297 exist in the hi-res mirror we use (and in the other mirror).
2. Labels: EyeQ's own DR grade equals the EyePACS grade for all 11,297 (by chance about 57%).
3. Pictures: in a sample of 600, the hi-res picture matches the low-res mirror's picture of the same name in **600 of 600** (lowest similarity 0.996), and Reject photos are measurably darker (median brightness 67 vs 112) and blurrier (sharpness 2.2 vs 5.7) than Good ones.

## Data and splits (patient level)
- EyeQ train labels: 12,543 names (Good 8,347 / Usable 1,876 / Reject 2,320). 1,246 belong to the EyePACS **held-out test patients** and are **excluded before any file is opened**. (No test patient was ever loaded.)
- Training 8,595, validation 1,503 (epoch selection only), both from "rest" patients; the **1,199 photos of the dev patients** are the held-out evaluation (831 patients). The validation and training patients are disjoint from the evaluation patients, and no EyePACS held-out test patient is present.

## Training
- EfficientNetB0 at **224×224**, ImageNet weights, 3 classes, class weights (power 0.5), best epoch chosen by validation macro-F1.
- **No contrast enhancement (CLAHE) and only geometric augmentation:** the gate must see darkness and haze, so changing brightness would change the very thing it judges. Very dark photos are not cropped or brightened.
- 3 seeds (Kaggle job `rg-quality-train`, 27 min of GPU); the deployed one is **seed 42**, fixed before training. It is not retrained after seeing the numbers.

## Evaluation (dev patients, never used for training or epoch choice; CI: bootstrap over patients)
| Metric | Seed 42 (deployed) | 3 seeds mean ± std |
|---|---|---|
| Accuracy | **0.942** [0.929–0.955] | 0.940 ± 0.003 |
| Macro-F1 | 0.906 [0.884–0.925] | 0.901 ± 0.006 |
| Reject recall | 0.889 [0.848–0.928] | 0.919 ± 0.026 |
| Gradable photos refused as Reject | 1.2% | – |

Confusion matrix of the deployed seed (rows = true, columns = predicted):

| | Good | Usable | Reject | recall |
|---|---|---|---|---|
| Good | 772 | 16 | 0 | 0.98 |
| Usable | 16 | 158 | 12 | 0.85 |
| Reject | 1 | 24 | 200 | 0.89 |

| Class | precision | recall | F1 |
|---|---|---|---|
| Good | 0.978 | 0.980 | 0.979 |
| Usable | 0.798 | 0.849 | 0.823 |
| Reject | 0.943 | 0.889 | 0.915 |

The Good/Reject boundary is clean (0 of 788 Good called Reject); Usable is the hard, human-defined middle class.

**Reject threshold trade-off** (the app rejects when the top class is Reject). How many bad photos are stopped, at the price of refusing gradable ones: P(Reject) ≥ 0.3: stops 92.4%, refuses 2.4%; ≥ 0.5: 88.9%, 1.2%; ≥ 0.7: 80.0%, 0.6%. Choosing a stricter threshold (the app uses argmax) is possible; it is a clinical decision.

## Does quality matter for grading?
The **frozen grading model** was applied to the same 1,199 dev photos (its stored predictions) and grouped by quality (ground-truth EyeQ label):

| Quality | n | grade accuracy | far errors (≥ 2 grades) | referable AUC | referable sensitivity / specificity |
|---|---|---|---|---|---|
| Good | 788 | 0.737 | 5.6% | 0.915 | 0.71 / 0.94 |
| Usable | 186 | 0.769 | 9.1% | 0.910 | 0.66 / 0.94 |
| **Reject** | 225 | 0.684 | **23.6%** | 0.855 | 0.82 / **0.71** |

Reject photos get far more large errors and far more false referrals (specificity 0.71). Grouped by the gate's *own* labels the picture is the same (Reject: far errors 25%, specificity 0.70, n = 212), so the app's behaviour (no grade for Reject) is justified by the data.

## Verified end to end on real photos (Kaggle CPU job `rg-quality-check`, code `add456e`)
- Raw dev photos → `QualityGate.predict` vs the stored predictions of the training job (seed 42): **300 of 300 labels identical**, maximum probability difference 0.006.
- The combined pipeline output keeps every grading key and adds `quality`; on 24 raw photos the labels were Good 16, Usable 2, Reject 6.
- Time per photo: **67 ms** on a Kaggle CPU (median; p90 70 ms).

## Limits
- Trained only on EyePACS labels; another camera can behave differently. No quality labels exist for APTOS or IDRiD.
- Usable is subjective, which shows in its lower precision. The team may prefer a 2-class gate (gradable / ungradable).
- The gate does not see *where* the problem is (a glare region can still hide a lesion).

## Five examiner questions
1. **Why not use CLAHE for the gate?** It equalises contrast, which would hide exactly the darkness and haze that make a photo bad. The grading model does use it; the gate must not.
2. **What happens to a Reject photo?** The app asks for a new photo and shows no grade, because the grading model makes far more large mistakes on such photos (24% vs 6%).
3. **How do you know the EyeQ labels match the right images?** We checked names (11,297/11,297), the DR grade stored in EyeQ (11,297/11,297) and picture similarity between two independent mirrors (600/600).
4. **Can test patients leak into the gate?** No: their names are removed by a fixed hash rule before any file is opened, and training/validation/evaluation patients are disjoint.
5. **Why 224 pixels?** It is enough for quality, cheaper (67 ms per photo) and consistent with published EyeQ models; the grading model uses 384.
