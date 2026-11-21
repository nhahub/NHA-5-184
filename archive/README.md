# Archive — v1 and one-time documents

**In short:** nothing in this folder is used by the current system (model **v2**). It is kept so the project history stays complete and presentable: how v1 worked, which files produced it, and earlier plans. For the current model see [`docs/v2_model_report.md`](../docs/v2_model_report.md).

## `v1/` — the first model (two-stage cascade, 224×224, APTOS only)
v1 = Stage 1 (DR yes/no) followed by Stage 2 (grade 1–4). On the 733 held-out APTOS images it scored QWK 0.840 / accuracy 78.3%; v2 scores 0.898 / 83.8% on the same images.

| Path | What it is | Replaced in v2 by |
|---|---|---|
| `v1/model/best_binary_dr_model.keras` | A v1 Stage 1 model file (17 MB), uploaded by the team. sha256 `20c19854…75eac32`. This is **not** the same file as the Stage 1 model used for the v1 reproduction (`stage1_v1`, sha256 `5efbb326…`, kept on Kaggle); it is an earlier or alternate training run. | `B_full_s42_384_mix` (private Kaggle dataset `omarashour206/rg-final-model`) |
| `v1/notebooks/Stage1_Binary_Classification_DR_vs_NoDR.ipynb` | Team notebook: Stage 1 binary training | `src/training/train.py` (candidate B, one 5-class model) |
| `v1/scripts/stage1_binary_train.py` | Stage 1 training script (team) | `src/training/train.py` |
| `v1/scripts/stage2_severity_cascade.py` | Stage 2 training and cascade evaluation (team) | `src/training/train.py` + `src/evaluation/` |
| `v1/scripts/eyepacs_prepare.py` | EyePACS → arrays with a patient-level split (team; only tested on synthetic data) | `kaggle/kernels/rg_eyepacs_data.py` |
| `v1/scripts/eyepacs_pretrain_stage2.py` | EyePACS pretraining for Stage 2 (team; never run on real data) | mixed APTOS + EyePACS training (`extra_data="eyepacs"` in `train.py`) |

The v1 *inference* code, `src/retinograde_inference.py`, is **not** archived. It is the app's inference API and still loads v1 until it is switched to v2 (next step).

## `docs/`
| Path | What it is |
|---|---|
| `docs/legacy/` | Early planning guides from when the team was starting. Partly outdated: the preprocessing code there was superseded by `preprocess_rgb` in `src/retinograde_inference.py`. |
| `docs/GITHUB_SYNC_PLAN.md` | The one-time plan used to bring the v2 work into this repository (PR #1, 2026-11-22). Its file paths describe the repo *before* this archive was created. |
