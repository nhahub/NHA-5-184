# RetinoGrade AI — Intelligent Diabetic Retinopathy Grading and Quality Assurance System

**Diabetic retinopathy (DR) grading from retinal fundus photos, with explainability and quality assurance.**
Graduation project of the Digital Egypt Pioneers Initiative (DEPI), track *AI & Data Science (Microsoft ML)*, Round 5.

> ⚠️ **Decision-support / screening-triage research prototype — not a diagnostic device.** Every output must be reviewed by a qualified eye-care professional.

## Project overview

Diabetic Retinopathy is a diabetes-related eye disease that can lead to vision loss if not detected and managed early. This project develops an automated, explainable pipeline (Deep Learning + Explainable AI) that:

1. checks the quality of the input retinal image;
2. rejects ungradable or low-quality images;
3. classifies gradable images into one of five DR grades (0 No DR · 1 Mild · 2 Moderate · 3 Severe · 4 Proliferative);
4. provides the model's prediction confidence, plus *referable* (grade ≥ 2) and *sight-threatening* (grade ≥ 3) flags;
5. uses Grad-CAM to highlight the image regions that contributed to the prediction.

Step 3–4 are done (model v2 below). Steps 1–2 (quality gate on EyeQ) and 5 (Grad-CAM on v2) are the next milestones.

## Current model (v2, frozen 2026-11-21)

One **EfficientNetB0** with a 5-class softmax, **384×384** input, trained on **APTOS 2019 + 8,000 EyePACS** images. Each test set below was scored **once**, after the settings were frozen:

| Test set (never used for training or tuning) | QWK | Accuracy | Referable AUC | Sight-threatening AUC |
|---|---|---|---|---|
| **APTOS 2019 held-out, 733 images** | **0.898** | **83.8%** | 0.985 | 0.945 |
| EyePACS test patients, 3,428 images | 0.669 | 73.8% | 0.907 | 0.970 |
| IDRiD (India), 516 images | 0.714 | 51.2% | 0.946 | 0.916 |

The previous version (v1, a two-model cascade at 224px) scored QWK 0.840 / 78.3% on the same 733 APTOS images. The improvement is **+0.058 QWK** [95% CI +0.034, +0.093].
Full report: [`docs/v2_model_report.md`](docs/v2_model_report.md) · all experiments: [`docs/experiments.md`](docs/experiments.md) · test tables: [`docs/test_results.md`](docs/test_results.md).

## Repository layout

```
APSOT Dataset/               team notebooks: APTOS preparation & preprocessing, one-stage training experiments
                             (01–06), two-stage training (stage 1 notebook + v1 Stage 1 model file)
Dataset 2015/                team notebooks: EyePACS 2015 audit & cleaning, image validation, quality analysis
Datasets/                    links to the Kaggle datasets
src/
  retinograde_inference.py   inference API (RetinoGrade class, preprocess_rgb = the frozen preprocessing contract)
  data/aptos_v2.py           v2 data access, augmentation, batched prediction
  training/train.py          generic trainer (candidates A/B/C, folds, seeds, 384px, APTOS+EyePACS, full-pool model)
  training/predict.py        re-scoring saved models (TTA, guarded one-shot test)
  evaluation/                metrics (QWK, sensitivity/specificity, AUC, bootstrap CIs) + one script per experiment
  explain/gradcam.py         Grad-CAM for the nested EfficientNet (Phase 4)
  model_io.py                how to load trained models safely
  training/stage1_*.py, stage2_*.py, eyepacs_*.py, preprocessing/   v1 scripts written by the team (kept unchanged)
kaggle/
  make_kernel.py             builds / pushes / pulls private Kaggle jobs
  kernels/                   the job scripts (data build, training, scoring, final test)
notebooks/colab_final_model.ipynb   how the final model was trained on Colab
docs/                        report, experiment log, frozen config, test results, proposal, legacy plans
tests/                       local tests (no TensorFlow needed)
notion/                      task-board export (the live board is in Notion)
```

The notebooks are the team's exploration history; `src/` is the reproducible pipeline that produced model v2.

## Quick start (local, no GPU, no data)

Requires Python 3.11 or newer.

```bash
git clone https://github.com/OmarAshour1/Diabetic-Retinopathy-Project.git
cd Diabetic-Retinopathy-Project
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements-local.txt
python -m pytest              # 30 tests, all synthetic data
```

Training and evaluation run on **Kaggle** (private jobs), not on a laptop. See [`kaggle/README.md`](kaggle/README.md). `requirements.txt` pins the Kaggle environment (TensorFlow 2.21.0 / Keras 3.13.2), which the saved models need.

Re-create the analysis tables from downloaded results (CPU, seconds):

```bash
python src/evaluation/report.py --runs kaggle/runs          # results block in docs/experiments.md
python src/evaluation/final_test.py --preds kaggle/runs/rg-final-test-cpu/final_test_preds.npz
```

## Data and rules

- **Nothing in this repo contains images, datasets or model files.** APTOS 2019 and EyePACS are Kaggle competition data (non-commercial, **no redistribution**). They are used only inside private Kaggle jobs. See [`data/README.md`](data/README.md).
- Models live as private Kaggle outputs / datasets ([`models/README.md`](models/README.md)).
- Never commit credentials. The Kaggle token stays in `~/.kaggle/`; for Colab it is typed at runtime and deleted.

## Status

| Milestone | Status |
|---|---|
| M1 Data collection, preprocessing, EDA | done |
| M2 DR severity model + evaluation | **done (v2 frozen, tested)** · image-quality model (EyeQ) not built yet |
| M3 Grad-CAM + IDRiD validation, EyePACS generalization, Azure | Grad-CAM code written, not yet run on v2 · EyePACS done · Azure not started |
| M4 MLflow, Streamlit, monitoring | not started |
| M5 Documentation and presentation | in progress |

Note: `src/retinograde_inference.py` still loads the **v1** cascade. Switching it to the v2 model at 384px, with the same output keys, is the next step.

## Team

Omar Khaled Mohamed Nasr (team leader) · Omar Ashour Ragheb Abdelkarim · Eyad Tamer Amin Zayed Zayed · Omar Ibrahim Hussein Hussein Tibah · Ahmed Samy Soliman Atia · Yahia Salah Eldeen Fahmy.

*Parts of the code, experiments and documentation were produced with AI assistance (Claude Code). See the note at the end of [`docs/v2_model_report.md`](docs/v2_model_report.md).*
