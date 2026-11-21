# Models (not generated here — copy the real files from the team's Kaggle outputs)
Expected v1 artifacts (≈15 MB each):
- `stage1_v1.keras`            — Stage 1 binary DR / no-DR (frozen copy of best_binary_dr_model.keras)
- `best_severity_model.keras`  — Stage 2 (Grades 1–4), v1 candidate (power 1.0)
- `best_severity_model_pow0.5.keras` — Stage 2 with softer class weights (optional)
- `val_probs_v1.npz`           — saved validation probabilities p1, p2, y_true (from Cell 14)
Never overwrite these; new models get new names (`*_v2`). Record file hashes in `docs/experiments.md`.

## Where the v1 files actually are (checked 2026-11-19)
- Kaggle `omarashour206/aptos-stage1-models`: `best_binary_dr_model.keras` (= **stage1_v1**) + `best_severity_model.keras` (same size as the local copy).
- Local `models/`: `best_severity_model.keras`, `best_severity_model_phaseA.keras`, and an older weights-only `best_effnet_stage1.weights.h5` (not the stage1_v1 model). There is no local `stage1_v1.keras` or `val_probs_v1.npz`. The Kaggle kernel `rg-v1-repro` regenerates `val_probs_v1.npz`.
- sha256 hashes: `docs/experiments.md` → "Model artifacts".

## v2 (final, frozen 2026-11-21)
- `B_full_s42_384_mix_model.keras` (81 MB) — EfficientNetB0 5-class, 384×384, APTOS + EyePACS. Private Kaggle dataset
  `omarashour206/rg-final-model` (trained on Colab, TF 2.21). Settings: `docs/frozen_config.json`.
- Fold models `B_f{0..4}_s42_384_mix_model.keras` — outputs of kernels `rg-train-mix-b-pilot` (folds 0–1) and `rg-train-mix-b-s42` (folds 2–4).
- Load with `src/model_io.load_model` (as saved; do not convert to float32 — see that file).
