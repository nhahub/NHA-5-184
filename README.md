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

One **EfficientNetB0** with a 5-class softmax, **384×384** input, trained on **APTOS 2019 + 8,000 EyePACS** images. v2 was scored **once** on each test set below, after its settings were frozen:

| Test set (no v2 model trained on it or was tuned on it) | QWK | Accuracy | Referable AUC | Sight-threatening AUC |
|---|---|---|---|---|
| **APTOS 2019 held-out, 733 images** | **0.898** | **83.8%** | 0.985 | 0.945 |
| EyePACS test patients, 3,428 images | 0.669 | 73.8% | 0.907 | 0.970 |
| IDRiD (India), 516 images | 0.714 | 51.2% | 0.946 | 0.916 |

The previous version (v1, a two-model cascade at 224px, now in [`archive/v1/`](archive/v1)) scored QWK 0.840 / 78.3% on the same 733 APTOS images. The improvement is **+0.058 QWK** [95% CI +0.034, +0.093].
Caveat: the 733 APTOS images were v1's validation set, and v2 started from some v1 settings, so the APTOS score may be slightly optimistic. EyePACS test patients and IDRiD have no such link.
Full report: [`docs/v2_model_report.md`](docs/v2_model_report.md) · all experiments: [`docs/experiments.md`](docs/experiments.md) · test tables: [`docs/test_results.md`](docs/test_results.md).

## Repository layout

```
src/                         ← model v2 pipeline
  retinograde_inference.py   inference API (RetinoGrade class; preprocess_rgb = the frozen preprocessing contract)
  data/aptos_v2.py           v2 data access, augmentation, batched prediction
  training/train.py          trainer (5-class model, folds, seeds, 384px, APTOS + EyePACS, full-pool final model)
  training/predict.py        re-scoring saved models
  evaluation/                metrics (QWK, sensitivity/specificity, AUC, bootstrap CIs) + one script per experiment
  explain/gradcam.py         Grad-CAM for the nested EfficientNet
  model_io.py                how to load trained models safely
tests/                       local tests (no TensorFlow needed)
kaggle/                      private Kaggle jobs: make_kernel.py + kernels/ (data build, training, scoring, final test)
notebooks/
  colab_final_model.ipynb    how the final v2 model was trained
  exploration/               the team's exploration notebooks (APTOS preparation + experiments, EyePACS 2015 audit)
docs/                        v2 report, experiment log, frozen config, test results, proposal
data/  models/               where the data and the models live (nothing is stored in the repo)
notion/                      task-board export
archive/                     v1 (model file, scripts, Stage 1 notebook) and old plans; not used by v2
```

`src/` is the reproducible pipeline that produced model v2. `notebooks/exploration/` documents how the team got there, and `archive/` keeps v1 for reference.

## Links

- Datasets: [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) · [EyePACS 2015 (224px mirror)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized) · [EyePACS 2015 (hi-res mirror)](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized) · [IDRiD](https://www.kaggle.com/datasets/aaryapatel98/indian-diabetic-retinopathy-image-dataset). Details in [`data/README.md`](data/README.md).
- Team task board (Notion): [Diabetic Retinopathy Project — Task Board](https://app.notion.com/p/Diabetic-Retinopathy-Project-Task-Board-3e1a4e61702980ef972cd3df4d63dd66)

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

Note: `src/retinograde_inference.py` runs model **v2** at 384px with the same output keys as v1 ([`docs/components/inference.md`](docs/components/inference.md)). Live status: [`docs/PROGRESS.md`](docs/PROGRESS.md).

## Team

Omar Khaled Mohamed Nasr (team leader) · Omar Ashour Ragheb Abdelkarim · Eyad Tamer Amin Zayed Zayed · Omar Ibrahim Hussein Hussein Tibah · Ahmed Samy Soliman Atia · Yahia Salah Eldeen Fahmy.

*Parts of the code, experiments and documentation were produced with the help of an AI coding assistant and reviewed by the team.*
