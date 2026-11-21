# RetinoGrade AI — Model v2 report (Phases 1–3)

**In short:** we replaced the v1 two-model cascade with **one EfficientNetB0 model that predicts all 5 grades from 384×384 images, trained on APTOS 2019 plus 8,000 EyePACS images**. On the 733 held-out APTOS 2019 images, scored once at the end, it reaches **QWK 0.898 and 83.8% accuracy**, against v1's QWK 0.840 and 78.3% on the same images. It was also tested on two datasets it never saw: EyePACS patients (referable AUC 0.907, sight-threatening AUC 0.970) and IDRiD (referable AUC 0.946). It detects referable and sight-threatening disease well. Exact grading of mild (grade 1) and severe (grade 3) disease is still the weak point.

Commit IDs quoted here refer to the development history (tag `v2-dev-history` in the original working copy; the public repo holds a snapshot of the same files). Period: 19–21 Nov 2026. All heavy compute ran on Kaggle (and once on Colab), never locally. Every number below comes from a logged run; details are in [`experiments.md`](experiments.md), test tables in [`test_results.md`](test_results.md), and the frozen settings in [`frozen_config.json`](frozen_config.json).

---

## 1. Starting point (v1)

- **Model:** a cascade. Stage 1 (EfficientNetB0, DR yes/no, threshold 0.3), then Stage 2 (EfficientNetB0, grades 1–4). Both at 224×224.
- **Reported result:** QWK 0.831–0.840 on a 733-image validation set. That same set was also used to choose epochs and the threshold, so the number was optimistic.
- **Never tested:** a single 5-grade model, more than one seed, or any external dataset.

## 2. What was done, step by step

### Phase 1 — set-up and data audit
| Step | What and why | Result |
|---|---|---|
| Tooling | `kaggle/make_kernel.py` builds and pushes **private** Kaggle jobs from the repo. `src/evaluation/` holds the metrics (QWK, sensitivity/specificity, AUC, bootstrap CIs) and the automatic results table. | Every run is reproducible from a commit. |
| v1 reproduction (`rg-v1-repro`) | Re-scored v1 under the current TensorFlow version. | QWK 0.840: matches the earlier result. |
| Data audit (`rg-recon`) | Checked how the v1 arrays were made. | Found: (1) training images were preprocessed slightly differently from what the app uses; (2) 1,560 rotated copies of grade 3–4 images in training; (3) **137 groups of near-duplicate images, 37 with conflicting labels**; (4) **50 training images were near-copies of validation images**, so v1's validation score was inflated. |
| Data v2 (`rg-data-v2`) | Rebuilt all images with the exact app preprocessing (stored losslessly as uint8). Test = the 733 old validation images, untouched until the end. 50 leaking images dropped. 5-fold cross-validation with duplicates kept in the same fold. IDRiD prepared as an external test. | Train pool 2,879 · test 733 · IDRiD 516. |

### Phase 2 — choosing the model (5-fold cross-validation, never the test set)
| Experiment | Result | Decision |
|---|---|---|
| A = cascade retrained · B = single 5-class softmax · C = ordinal regression | QWK: A 0.898 · B 0.895 ± 0.002 · C 0.904 ± 0.003 (3 seeds) | Within noise. |
| B vs C at the **same specificity** | B is as good or better (referable AUC 0.981 vs 0.976). C's apparent sensitivity gain came only from a looser threshold. | **B chosen.** |
| B vs A for "any DR" | AUC 0.997 for both. | Cascade not needed. |
| Test-time augmentation and model averaging | +0.005 QWK or less (within noise), at 4–5× the cost. | Not adopted. |

### "Headroom check" — is there still room to improve?
| Check | Result |
|---|---|
| Duplicates across folds (image fingerprints) | Only 5 pairs. Removing them changes nothing, so no leakage. |
| Error analysis | 75% of errors are one grade off. Image quality does not explain errors once grade mix is accounted for. Some confident mistakes look like label noise. |
| Learning curve | At 224px the model stops improving from about 50% of the data, so more of the same data won't help. |
| **EyePACS check** (unseen patients) | **Large drop:** referable AUC 0.98 → 0.76, and grade 0 vs grade 1 at chance. The likely cause is a shortcut: in APTOS, healthy and diseased eyes partly come from different clinics/cameras. **This was the biggest weakness found.** |

### Improvements (each adopted with a rule fixed beforehand)
| Change | Evidence | Decision |
|---|---|---|
| **384px images** instead of 224px | QWK +0.016 [+0.009, +0.023] over 5 folds × 3 seeds; better in 13/15 runs. | Adopted. CPU time rises ~20%. |
| **Train on APTOS + 8,000 EyePACS images** | EyePACS referable AUC **+0.117** [+0.095, +0.139] (0.78 → 0.90), confirmed on 3 seeds. APTOS QWK −0.010 (within the allowed −0.02). | Adopted. |

### Freeze and final model
- **Final model:** trained once on all training data (APTOS 2,879 + 8,000 EyePACS) for 9 fine-tune epochs, the median best epoch of the cross-validation runs.
- **Where it was trained:** the Kaggle GPU queue was stuck for 5 h, so on Google Colab, with the same TensorFlow version. It is stored in the private Kaggle dataset `rg-final-model`.
- **Frozen before testing** (commit `86f7381`, [`frozen_config.json`](frozen_config.json)):
  - grade = the most likely of the 5 grades;
  - **referable** if P(grade ≥2) ≥ 0.327;
  - **sight-threatening** if P(grade ≥3) ≥ 0.028.
  - Both thresholds were set for 90% specificity on EyePACS development patients.

## 3. Final performance (Phase 3, one-shot test)

| Test set (never used before) | QWK | Accuracy | Within 1 grade | Referable AUC | Sight-threat. AUC | Referable sens / spec | Sight-threat. sens / spec |
|---|---|---|---|---|---|---|---|
| **APTOS 2019 held-out (733)** | **0.898** [0.872–0.922] | **83.8%** | 96.0% | 0.985 | 0.945 | 0.950 / 0.938 | 0.908 / 0.762 |
| EyePACS test patients (3,428) | 0.669 [0.630–0.705] | 73.8% | 90.3% | 0.907 | 0.970 | 0.760 / 0.901 | 0.943 / 0.888 |
| IDRiD, India (516) | 0.714 [0.661–0.758] | 51.2% | 87.4% | 0.946 | 0.916 | 0.780 / 0.959 | 0.961 / 0.695 |

**v1 vs v2 on the same 733 APTOS images (paired):**

| | v1 | v2 | Change [95% CI] |
|---|---|---|---|
| QWK | 0.840 | **0.898** | +0.058 [+0.034, +0.093] |
| Accuracy | 78.3% | **83.8%** | +5.5 points |
| Referable sensitivity, at the same 93.8% specificity | 0.906 | **0.950** | +0.044 |
| Sight-threatening AUC | 0.915 | **0.945** | +0.030 |

**Per-grade recall on APTOS test:** No DR 0.97 · Mild 0.68 · Moderate 0.81 · Severe 0.36 · Proliferative 0.63.

**Strengths:**
- **Severe disease:** it reliably finds referable and sight-threatening disease on all three datasets (AUC 0.91–0.99). It catches 94–96% of sight-threatening eyes on EyePACS and IDRiD.
- **Generalizes beyond APTOS:** it now works on a second population (EyePACS referable AUC 0.78 → 0.91).
- **Trustworthy numbers:** the test results match the cross-validation predictions.

**Limitations (state these honestly):**
- **Exact grade is hard:** IDRiD accuracy is 51%; the model often grades one level lower than IDRiD's graders. Grade 1 and grade 3 recall are low.
- **Over-referral:** the sight-threatening alarm was tuned on EyePACS. On APTOS and IDRiD it over-refers (specificity 0.70–0.76).
- **Label noise:** APTOS has label noise (28 of 98 duplicate pairs disagree), which caps achievable accuracy.
- **Screening support only:** this is a decision-support tool, not a diagnostic device.

## 4. What changed in the repository

**Team files not modified:** `src/retinograde_inference.py`, `src/training/stage1_binary_train.py`, `src/training/stage2_severity_cascade.py`, `src/training/eyepacs_pretrain_stage2.py`, `src/preprocessing/eyepacs_prepare.py`.

**New or extended:**
- `src/training/train.py` (generic trainer: candidates A/B/C, folds, seeds, 384px, mixed APTOS+EyePACS, full-pool mode) and `src/training/predict.py` (re-scoring, TTA).
- `src/data/aptos_v2.py` (v2 data, augmentation, `StackedArrays`, `RG_DATA_ROOTS`) and `src/model_io.py` (safe model loading).
- `src/evaluation/`: metrics, report, and one script per experiment, each runnable on CPU from the saved predictions.
- `src/explain/gradcam.py`: Grad-CAM for model B (written; not yet run on the final model).
- `kaggle/`: job helper and 15 private Kaggle job scripts (data build, training, scoring, test).
- `notebooks/colab_final_model.ipynb`: how the final model was trained on Colab.
- `docs/`: `experiments.md` (every run and decision), `test_results.md`, `frozen_config.json`, this report.
- `tests/`: 30 local tests (`pytest`), all passing.

**Never committed:** images, datasets, model files, tokens. Galleries for label review live in the gitignored `docs/figures_private/`.

## 5. Problems met and how they were handled
| Problem | Handling |
|---|---|
| A "faster float32" model copy gave wrong predictions on real images | Caught by comparing against stored predictions before any result used it. Dropped; models load as saved (rule recorded in `src/model_io.py`). |
| Kaggle GPU queue stuck for ~5 h | Final model trained on Colab with pinned versions; test run on Kaggle CPU. |
| Kaggle token expired mid-run | User generated a new token; nothing was lost. |
| Duplicate groups with conflicting labels | Kept within one fold; listed for team label review. |

## 6. Compute
About 20 GPU-hours on Kaggle (free weekly quota) plus 41 minutes on Colab. Local machine: editing, tests and analysis only.

## 7. Next steps (not started; need a go-ahead)
1. Switch `src/retinograde_inference.py` to the v2 model at 384px, keeping the same output keys for the app and API.
2. Run Grad-CAM on the final model and validate it against the IDRiD lesion masks.
3. Image-quality gate (EyeQ), confidence calibration, Streamlit app, API, MLflow, Azure deployment, documentation.

---
*AI assistance: the experiment design, code in `src/evaluation`, `src/training/train.py`, `kaggle/` and this report were produced with Claude Code (Anthropic) under the team's direction and approval at each step. All results come from runs logged in `docs/experiments.md`.*
