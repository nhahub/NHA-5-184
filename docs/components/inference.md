# Component: inference (model v2)

**In short:** [`src/retinograde_inference.py`](../../src/retinograde_inference.py) turns one fundus photo into a grade (0–4), probabilities and two referral flags. It is the only place where a photo is prepared for the model, so the app, Grad-CAM and the tests all see exactly the same pixels the model was trained on.

## How to run
```bash
# weights: argument, else the environment variable RG_MODEL, else models/B_full_s42_384_mix_model.keras
python src/retinograde_inference.py eye.png --model models/B_full_s42_384_mix_model.keras
```
```python
from retinograde_inference import RetinoGrade
rg = RetinoGrade()                       # needs TensorFlow 2.21 (requirements.txt)
r = rg.predict("eye.png")                # path | bytes | uploaded file | RGB uint8 array
```
v1 (old cascade, 224px) is for comparison only: `RetinoGrade(version="v1", stage1_path=..., stage2_path=...)`.

## What it returns
All keys of v1 are kept. v2 is one 5-class model without stages, so the v1 keys are *derived*:

| Key | Meaning in v2 |
|---|---|
| `grade`, `grade_name` | argmax of the 5 probabilities (frozen rule) |
| `confidence` | probability of that grade (not calibrated; see the calibration component) |
| `stage1_dr_probability` | `1 − P(grade 0)` |
| `stage2_grade_probabilities` | `P(grade k) / (1 − P(0))` for k = 1…4: the grade probabilities *given* that there is DR. All 0 if the model gives DR no probability at all |
| `referable` | `P(grade ≥ 2) ≥ 0.3269` (frozen threshold) |
| `sight_threatening` | `P(grade ≥ 3) ≥ 0.0277` (frozen threshold) |
| `threshold` | the referable threshold (in v1: the Stage 1 cut-off 0.3) |
| `model_version`, `disclaimer` | `"v2"`, the medical disclaimer |
| *added:* `probabilities`, `p_referable`, `p_sight_threatening`, `thresholds`, `model_name` | the 5 raw probabilities, the two sums behind the flags, both thresholds |

**The flags use their own thresholds, not the grade.** An eye can be graded 1 and still be flagged, because the sight-threatening threshold is very low (0.0277). That is how the model was frozen and tested.

## Why this design
- **One image-size setting** (`IMG_SIZE = 384`): preprocessing and `transform_mask` take their default from it, and loading a model of another size is refused.
- **Bit-exact input:** training stored `round(preprocessed × 255)` as whole numbers, so inference rounds the same way.
- **Weights outside the code:** path or environment variable; a missing file gives a clear error.
- **Compiled graph:** the model runs through one `tf.function`; same numbers, less Python overhead per image.
- **v1 is explicit:** the old positional call `RetinoGrade(a, b)` is refused, so nobody uses v1 by accident.

## Evidence
Kaggle CPU job `rg-infer-check` (code `03b17d7`, real frozen model, train pool + EyePACS **dev** only; no test split):

| Check | Result |
|---|---|
| Preprocessing vs the stored 384px training arrays, 120 raw APTOS train images | **120/120 bit-exact** |
| Predictions vs those stored when the model was trained, 300 raw EyePACS dev images | max probability difference 0.0096 · same grade 299/300 · both flags the same 300/300 |

The one differing grade is a near-tie: the stored values come from a GPU and these from a CPU, in mixed precision.

CPU time per image, file → result (Kaggle CPU, 30 images, batch size 1):

| Threads | Preprocess | Model | Total median | Total p90 |
|---|---|---|---|---|
| 1 | 87 ms | 128 ms | **214 ms** | 289 ms |
| 2 | 74 ms | 111 ms | **183 ms** | 257 ms |
| 4 | 72 ms | 98 ms | **164 ms** | 248 ms |

Loading the model takes 1.5 s, once. The 655 ms reported earlier was a fold model of the same architecture called without the compiled graph, on another Kaggle machine, so not all of the difference is due to the code.

Tests: `tests/test_inference_v2.py` (19, no TensorFlow) and `tests/test_real_model_v2.py` (4, need TensorFlow and the model file).

## Limits
- The probabilities are not calibrated here.
- It does not check image quality: a black photo still gets a grade. The quality gate does that.
- It needs TensorFlow 2.21 / Keras 3.13 (Python 3.11–3.13).

## Five examiner questions
1. **Why keep `stage1_…` keys if v2 has no stages?** Other components already read them. They are computed from the 5 probabilities with the formulas above, so nothing downstream breaks.
2. **How do you know inference matches training?** We re-ran raw training photos through the inference code and compared every pixel with the stored training arrays: 120 of 120 identical. Predictions on 300 unseen dev photos match the stored ones.
3. **Why is "referable" not simply "grade ≥ 2"?** The argmax grade is a poor screening decision. A threshold on P(grade ≥ 2) lets us choose the sensitivity/specificity trade-off; it was fixed on development patients before the test.
4. **Why does rounding matter?** Training images were stored as whole numbers 0–255. Skipping the rounding would feed slightly different values. The effect is small, but it would no longer be the model we tested.
5. **What happens with a wrong or missing file?** Unreadable images raise a clear `ValueError`, a missing model raises `FileNotFoundError` naming `RG_MODEL`, and a model of the wrong size is refused.
