# One-shot test results (Phase 3)

**In short:** the frozen model `B384-mix-v2` was scored **once** on three test sets that no v2 model trained on or was tuned on (the 733 APTOS images were v1's validation set; see the limitations in `v2_model_report.md`): APTOS 2019 held-out (733), IDRiD grading (516) and EyePACS test patients (3,428 images, 1,714 patients). Settings and thresholds were fixed in `docs/frozen_config.json` (commit 86f7381) **before** this run; nothing below was tuned on these images.

Kernel `rg-final-test-cpu` (Kaggle CPU, code 7d93e26, 94 min). Scoring: `python src/evaluation/final_test.py --preds kaggle/runs/rg-final-test-cpu/final_test_preds.npz`. 95% CIs from 2,000 bootstrap resamples (EyePACS: by patient). The v1 row uses v1's stored predictions on the same 733 APTOS images (its QWK/accuracy use v1's own rule; sensitivity/specificity use the frozen v2 thresholds on v1's probabilities).

Primary model: `B_full_s42_384_mix` · ensemble: 5 models · thresholds P(≥2) ≥ 0.327, P(≥3) ≥ 0.028 · grade = argmax

### APTOS test (n=733; grades [361, 74, 200, 39, 59])

| metric | primary model | ensemble | v1 cascade (thr 0.3) | Δ primary − v1 [CI] |
|---|---|---|---|---|
| QWK | 0.898 [0.872–0.922] | 0.886 [0.860–0.913] | 0.840 | 0.058 [0.034–0.093] |
| accuracy | 0.838 [0.810–0.864] | 0.816 [0.787–0.843] | 0.783 | 0.055 [0.031–0.085] |
| within 1 grade | 0.960 [0.947–0.974] | 0.956 [0.941–0.971] | 0.926 | 0.034 [0.016–0.052] |
| AUC any DR (≥1) | 0.997 [0.995–0.999] | 0.998 [0.996–0.999] | 0.987 | 0.010 [0.006–0.016] |
| AUC referable (≥2) | 0.985 [0.977–0.991] | 0.983 [0.976–0.990] | 0.980 | 0.004 [-0.002–0.011] |
| AUC sight-threat. (≥3) | 0.945 [0.924–0.965] | 0.940 [0.918–0.961] | 0.915 | 0.030 [0.014–0.046] |
| referable sensitivity | 0.950 [0.924–0.973] | 0.940 [0.911–0.964] | 0.906 | 0.044 [0.014–0.076] |
| referable specificity | 0.938 [0.914–0.960] | 0.936 [0.912–0.958] | 0.938 | 0.000 [-0.019–0.018] |
| sight-threat. sensitivity | 0.908 [0.850–0.963] | 0.949 [0.901–0.990] | 0.939 | -0.031 [-0.075–0.011] |
| sight-threat. specificity | 0.762 [0.730–0.795] | 0.690 [0.654–0.726] | 0.660 | 0.102 [0.076–0.128] |

Primary model confusion matrix (rows = true, cols = predicted):

| true \ pred | 0 | 1 | 2 | 3 | 4 | recall |
|---|---|---|---|---|---|---|
| 0 | 350 | 11 | 0 | 0 | 0 | 0.97 |
| 1 | 7 | 50 | 17 | 0 | 0 | 0.68 |
| 2 | 0 | 22 | 163 | 7 | 8 | 0.81 |
| 3 | 1 | 1 | 16 | 14 | 7 | 0.36 |
| 4 | 0 | 6 | 13 | 3 | 37 | 0.63 |

### IDRID test (n=516; grades [168, 25, 168, 93, 62])

| metric | primary model | ensemble |
|---|---|---|
| QWK | 0.714 [0.661–0.758] | 0.738 [0.689–0.779] |
| accuracy | 0.512 [0.467–0.556] | 0.510 [0.467–0.550] |
| within 1 grade | 0.874 [0.843–0.901] | 0.893 [0.866–0.919] |
| AUC any DR (≥1) | 0.931 [0.908–0.951] | 0.938 [0.917–0.955] |
| AUC referable (≥2) | 0.946 [0.926–0.964] | 0.950 [0.932–0.967] |
| AUC sight-threat. (≥3) | 0.916 [0.893–0.938] | 0.919 [0.896–0.940] |
| referable sensitivity | 0.780 [0.734–0.826] | 0.789 [0.744–0.834] |
| referable specificity | 0.959 [0.929–0.984] | 0.969 [0.944–0.990] |
| sight-threat. sensitivity | 0.961 [0.929–0.988] | 0.981 [0.956–1.000] |
| sight-threat. specificity | 0.695 [0.643–0.745] | 0.651 [0.600–0.703] |

Primary model confusion matrix (rows = true, cols = predicted):

| true \ pred | 0 | 1 | 2 | 3 | 4 | recall |
|---|---|---|---|---|---|---|
| 0 | 131 | 32 | 2 | 0 | 3 | 0.78 |
| 1 | 16 | 9 | 0 | 0 | 0 | 0.36 |
| 2 | 15 | 64 | 79 | 4 | 6 | 0.47 |
| 3 | 1 | 17 | 48 | 24 | 3 | 0.26 |
| 4 | 0 | 6 | 15 | 20 | 21 | 0.34 |

### EYEPACS test (n=3428, 1714 patients; grades [2544, 250, 476, 86, 72])

| metric | primary model | ensemble |
|---|---|---|
| QWK | 0.669 [0.630–0.705] | 0.694 [0.654–0.727] |
| accuracy | 0.738 [0.721–0.756] | 0.768 [0.752–0.785] |
| within 1 grade | 0.903 [0.893–0.915] | 0.903 [0.892–0.915] |
| AUC any DR (≥1) | 0.826 [0.808–0.846] | 0.843 [0.825–0.861] |
| AUC referable (≥2) | 0.907 [0.891–0.923] | 0.915 [0.899–0.930] |
| AUC sight-threat. (≥3) | 0.970 [0.953–0.983] | 0.969 [0.952–0.983] |
| referable sensitivity | 0.760 [0.724–0.799] | 0.793 [0.755–0.830] |
| referable specificity | 0.901 [0.888–0.913] | 0.879 [0.864–0.894] |
| sight-threat. sensitivity | 0.943 [0.897–0.981] | 0.968 [0.931–0.994] |
| sight-threat. specificity | 0.888 [0.874–0.901] | 0.810 [0.793–0.827] |

Primary model confusion matrix (rows = true, cols = predicted):

| true \ pred | 0 | 1 | 2 | 3 | 4 | recall |
|---|---|---|---|---|---|---|
| 0 | 2120 | 245 | 165 | 1 | 13 | 0.83 |
| 1 | 164 | 56 | 29 | 0 | 1 | 0.22 |
| 2 | 127 | 46 | 276 | 22 | 5 | 0.58 |
| 3 | 3 | 2 | 46 | 31 | 4 | 0.36 |
| 4 | 1 | 2 | 11 | 10 | 48 | 0.67 |

EyePACS by EyeQ quality (primary model):

| quality | n | QWK | AUC ≥2 | AUC ≥3 | ref. sens/spec |
|---|---|---|---|---|---|
| Good | 843 | 0.620 | 0.907 | 0.944 | 0.683/0.940 |
| Usable | 175 | 0.752 | 0.953 | 0.982 | 0.909/0.887 |
| Reject | 228 | 0.669 | 0.923 | 0.962 | 0.900/0.780 |
| no label | 2182 | 0.670 | 0.902 | 0.974 | 0.751/0.897 |

