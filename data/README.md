# Data (never committed to git, never downloaded to laptops)

**In short:** all datasets stay on Kaggle. Our scripts run there (see `kaggle/README.md`) and only small result files come back. Do not download the competition images to your laptop: it is slow, and the APTOS/EyePACS rules forbid sharing them.

| dataset | Kaggle source (attached to kernels) | used for |
|---|---|---|
| APTOS 2019 | competition `aptos2019-blindness-detection` (accept rules once: "Late Submission") | training / CV / test |
| v1 arrays | `omarashour206/aptos2019-preprocessed-dataset` | v1 reproduction only |
| v2 arrays | outputs of kernels `omarashour206/rg-data-v2` (224px) and `rg-data-v2-384` (384px, same split) | all v2 training / test |
| EyePACS 2015 (hi-res mirror, ~1024px) | `tanlikesmath/diabetic-retinopathy-resized` | mixed training (patient-level 'rest' split), dev and test patients; arrays built by kernel `rg-eyepacs-data` |
| EyePACS 2015 (224px mirror) | `sovitrath/diabetic-retinopathy-2015-data-colored-resized` | recon / EyeQ mapping, mirror comparison |
| IDRiD | `aaryapatel98/indian-diabetic-retinopathy-image-dataset` | external grading test (516) + Grad-CAM vs lesion masks |
| EyeQ quality labels | GitHub `HzFu/EyeQ` CSVs (fetched inside `rg-recon`, mapped in `eyeq_mapping.csv`) | quality gate (T04–T06) |

Kaggle mount paths: competitions at `/kaggle/input/competitions/<slug>/`, datasets at `/kaggle/input/datasets/<user>/<slug>/`. Our kernels search `/kaggle/input` recursively, so the exact layout doesn't matter.

**EyePACS patient split (fixed forever):** `sha256("eyepacs-split-v1:<patient>")` → dev (< 0.10), test (0.10–0.20), rest (≥ 0.20). Only *rest* is used for training; *dev* for thresholds; *test* was opened once, in the final evaluation.
