# Running jobs on Kaggle

**In short:** the laptop only edits code and sends jobs. All data processing and training runs on Kaggle as private "kernels" (scripts). We download only the small result files (JSON/CSV/NPZ). Images, arrays and models stay on Kaggle.

## One-time setup
1. `pip install -r requirements-local.txt` (no TensorFlow needed locally).
2. Kaggle credentials live in your user profile (`%USERPROFILE%\.kaggle\`), **never inside the repo**.
3. Accept the competition rules in the browser once (APTOS: "Late Submission").

## The loop
```bash
# 1. ship the current src/ to Kaggle as the private dataset omarashour206/retinograde-code
python kaggle/make_kernel.py code
cd kaggle/_build/code && kaggle datasets create -p . -r zip               # first time only (run from the folder: Windows CLI path bug)
cd kaggle/_build/code && kaggle datasets version -p . -r zip -m "<msg>"   # every later code change

# 2. build a kernel folder (script + kernel-metadata.json) and push it (asks y/N)
python kaggle/make_kernel.py build rg-recon
python kaggle/make_kernel.py push  rg-recon

# 3. watch it, then pull only the small outputs into kaggle/runs/<kernel>/
python kaggle/make_kernel.py status rg-recon
python kaggle/make_kernel.py pull   rg-recon

# 4. regenerate the results table in docs/experiments.md
python src/evaluation/report.py --runs kaggle/runs
```

## Kernels

| kernel | runs on | approx. cost | needs | produces |
|---|---|---|---|---|
| `rg-recon` | CPU | 0.5–1 h | code, APTOS, v1 arrays, EyePACS mirror, IDRiD, internet (EyeQ CSVs) | `recon.json`, `xval_map.csv`, `xtrain_map.csv`, `dup_groups.csv`, `eyeq_mapping.csv`, `pip_freeze.txt` |
| `rg-v1-repro` | GPU T4 | ~0.3 GPU-h | code, v1 arrays, `aptos-stage1-models` | `val_probs_v1.npz`, `v1_repro.json` |
| `rg-data-v2` | CPU | ~0.5 h | code, APTOS, IDRiD, **rg-recon output** | `aptos_v2_u8.npy` + meta, `idrid_u8.npy` + meta, `data_v2_summary.json` |
| `rg-train-<suffix>` | GPU T4 | ~0.1 (QUICK_TEST) to ~2 GPU-h (5 folds) | code, **rg-data-v2 output** | `runs/<run_id>/run.json`, `oof_preds.npz`, `*.keras` |

Training configs are JSON overrides of `DEFAULTS` in `src/training/train.py`, for example:
```bash
python kaggle/make_kernel.py build rg-train --suffix qt  --config '{"candidate":"B","folds":[0],"quick_test":true}'
python kaggle/make_kernel.py build rg-train --suffix b-s42 --config '{"candidate":"B","folds":[0,1,2,3,4],"seed":42}'
```
Order: `rg-recon` → (`rg-v1-repro` in parallel) → `rg-data-v2` → `rg-train-qt` smoke test → real runs.

## Colab fallback (only if the weekly Kaggle GPU quota runs out)
Open a new Colab notebook, then upload your Kaggle token **at runtime** (never save it in the notebook):
```python
from google.colab import files; files.upload()   # pick kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!pip -q install kaggle && kaggle datasets download -d omarashour206/retinograde-code --unzip -p /content/code
```
Then run the same `src/training/train.py` with `data_dir=` pointing at the downloaded v2 arrays (kernel output of `rg-data-v2`, via `kaggle kernels output omarashour206/rg-data-v2`).
