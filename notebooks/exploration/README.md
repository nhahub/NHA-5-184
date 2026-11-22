# Exploration notebooks (team, Oct–Nov 2026)

**In short:** these are the team's exploration notebooks from Milestone 1 and the first modelling rounds, run on Kaggle. They explain *how we got here* (data audit, preprocessing choices, first EfficientNetB0 experiments). The **current model (v2)** is built by the reproducible pipeline in [`src/`](../../src) and [`kaggle/`](../../kaggle), not by these notebooks. See [`docs/v2_model_report.md`](../../docs/v2_model_report.md).

Owners of each notebook: see the task board in [`notion/`](../../notion).

## `aptos/` — APTOS 2019
| Notebook | What it covers |
|---|---|
| `00_Dataset_Preparation_and_Preprocessing.ipynb` | APTOS data preparation and image preprocessing (crop, pad, resize, CLAHE). Its preprocessing became the frozen `preprocess_rgb` contract in `src/retinograde_inference.py`. |
| `01_Baseline_EfficientNetB0.ipynb` | Baseline EfficientNetB0 and first fine-tuning |
| `02_Classification_Head_Experiments.ipynb` | Classification-head variants |
| `03_Class_Weights_and_FineTuning.ipynb` | Class weights and deeper fine-tuning |
| `04_Dataset_and_Preprocessing_Experiments.ipynb` | Dataset and input-preprocessing experiments |
| `05_FineTuning_with_New_Dataset.ipynb` | Fine-tuning on the rebuilt dataset |
| `06_Model_Comparison_and_Analysis.ipynb` | Model comparison and prediction analysis |

## `eyepacs_2015/` — EyePACS 2015 (Kaggle mirror)
| Notebook | What it covers |
|---|---|
| `01_Dataset_Audit_and_Cleaning.ipynb` | Dataset discovery, label/path checks, duplicate detection, clean manifest |
| `02_Image_Validation_and_Preprocessing.ipynb` | Image-size checks, validation, 224×224 resize with padding |
| `03_Image_Quality_Analysis.ipynb` | Brightness, contrast, sharpness and retinal-region quality checks. **Relevant for the upcoming image-quality gate (EyeQ).** |

The original two-stage Stage 1 notebook and the v1 model file are in [`archive/v1/`](../../archive/v1).

> Some notebooks contain image outputs. If any show APTOS or EyePACS photos, their owners should clear those outputs, because the dataset rules forbid redistributing the images.
