# Experiment log

> **Commit IDs in this log** (e.g. code `7d93e26`, freeze `86f7381`) refer to the development history, which is kept as the tag `v2-dev-history` in the original working copy. The public GitHub repo received a snapshot of the same files (see `archive/docs/GITHUB_SYNC_PLAN.md`). Kaggle run outputs record the same IDs in `code_version`.

**In short:** every model run is recorded here: which data and code version it used, its settings, and its scores. The results table at the bottom is regenerated automatically from the Kaggle outputs (`python src/evaluation/report.py`). Models are picked using cross-validation on the training pool. The held-out **test** set is scored only once, at the end.

## Data versions

| version | what | where | fingerprint |
|---|---|---|---|
| v1 arrays | `X_train/X_val/y_train/y_val.npy`, float32 [0,1], 80/20 split (`random_state=42`), train has augmented G3/G4 copies **without ids** | Kaggle dataset `omarashour206/aptos2019-preprocessed-dataset` | X_train 2,702,880,896 B · X_val 441,348,224 B |
| v2 arrays | `aptos_v2_u8.npy` (uint8, lossless w.r.t. `preprocess_rgb`) + `aptos_v2_meta.csv` (id_code, grade, dup_group, split, fold) | output of Kaggle kernel `rg-data-v2` (v1, 2026-11-19) | meta `a690105d…3a918d` · images `c46bae98…65b1aa` |
| IDRiD grading | `idrid_u8.npy` + `idrid_meta.csv`, 516 images (413 train / 103 test, official grades) | same kernel; source `aaryapatel98/indian-diabetic-retinopathy-image-dataset` | external test only |

### v2 split design
- **test** = the 733 old-val ids. These are the only images v1 never trained on, so it is the only fair v1-vs-v2 comparison. v1's number on it is still slightly optimistic, because v1's epochs and threshold were chosen on these same images.
- **dropped** = train images whose near-duplicate group also contains a test image (leak guard).
- **train pool** = everything else, 5-fold `StratifiedGroupKFold` (groups = near-duplicate groups, seed 42). All model, threshold and cut-point selection uses out-of-fold (OOF) predictions.
- Augmentation is on the fly only (flips, rot90, mild brightness/contrast inside the retina). No stored copies.
- External checks: IDRiD grading images (516), plus EyePACS held-out patients if prepared.

### What recon found about the v1 data (rg-recon v2, 2026-11-19)
- **Preprocessing mismatch:** the v1 arrays were made with crop-black → **pad-to-square** → resize → CLAHE, **without** the white-frame crop. The inference contract (`preprocess_rgb`) adds the white-frame crop, so the deployed v1 sees slightly different images than it was trained on. That variant reproduces 83.5% of X_val pixel-exactly, and the rest differ by ≤1.2/255 on average. v2 is trained on exactly `preprocess_rgb`.
- **Old split:** X_val is *not* `train_test_split(0.2, stratify, random_state=42)` on `train.csv` order. Its 733 ids were recovered by a one-to-one assignment covering all 3,662 ids (val ∩ train = ∅, labels agree 100%).
- **X_train** = 2,929 originals + 1,560 exact rot90×3 / h-flip copies of every Grade 3–4 image (390 × 4). So G3/G4 were 5× oversampled *and* class-weighted.
- **Near-duplicates:** 137 groups (283 images; 64×64 correlation ≥ 0.98, with a clean gap between 0.99 and 0.995). 37 groups have conflicting labels. **50 v1-train images were near-copies of v1-val images**, so the v1 val numbers are optimistic. v2 drops these 50 from training.
- **v2 split:** test 733 (grades 361/74/200/39/59) · dropped 50 · train pool 2,879 in 5 folds (StratifiedGroupKFold by duplicate group).

## Model artifacts (never overwritten)

| artifact | source | sha256 |
|---|---|---|
| `best_severity_model.keras` (v1 Stage 2) | local `models/` = Kaggle `omarashour206/aptos-stage1-models` | `a9b9eba9328e3a44a565ff5a77a7bb746e50214f60bc9a7697ffa1f72a682549` |
| `best_severity_model_phaseA.keras` | local `models/` = Kaggle `omarashour206/aptos-stage2-models` | `ae0d5b46b8dbca6440df02fcd38c5e07abb24d119a54a1027a656efdf3aa5a80` |
| `best_effnet_stage1.weights.h5` (older weights-only file) | local `models/` = Kaggle `omarashour206/retinograde-best-model` | `65f8905b833c57fbbf9defbcb192975b291f435aacf71c0cc3b197efaba7271b` |
| `best_binary_dr_model.keras` (= **stage1_v1**) | Kaggle `omarashour206/aptos-stage1-models` (not local) | `5efbb3265f765e238b09526dc91118bbac0494af6dec241881095a10fafb3a31` |

## Candidates (Phase 2)

| id | what | status |
|---|---|---|
| v1 | cascade from v1 (frozen) | **reproduced** on old val under TF 2.21: QWK 0.840 [0.807–0.870] = the team's earlier v1 result (0.831–0.840) ✓ |
| A | cascade retrained on v2 folds | seed 42 done: OOF QWK 0.898 (no gain over B; 2 models) |
| B | single 5-class softmax, B0/224 | 3 seeds: OOF QWK 0.895 ± 0.002 |
| C | ordinal regression (Huber) + OOF-fitted cut points | 3 seeds: OOF QWK 0.904 ± 0.003 · **leading candidate** |
| G | 4-view flip TTA + 5-fold ensemble | planned (inference only) |
| E | winner at 384² | planned (needs `rg-data-v2` at 384) |

Decision rule: adopt a change only if it beats the noise (about ±0.03 QWK, mean ± std over ≥3 seeds). When candidates are within noise, prefer the simpler single model.

## Run notes
<!-- free-text notes per run: date, who, kernel version, anything unusual -->
- **2026-11-19 · batch 1 (B, C · 5 folds · seed 42 · code 8ca964b).** Each kernel used ~0.5 h of GPU compute over ~1 h of session time (Kaggle bills the session), so ~2 GPU-h of quota for both. Best epochs fall in fine-tuning (epochs 6–16). Caveats: (1) fold-val scores also chose each fold's best epoch, so OOF is slightly optimistic; (2) one seed only, so no claims yet; (3) OOF covers the train pool, not the test set, so it is **not** directly comparable to v1's 0.840 on the old val (that comparison happens once, in Phase 3). C's cross-fitted cut points changed 171/2,879 predictions with no QWK change (0.9006 vs 0.9008 default).
- **2026-11-19 · batch 2 (A seed 42; B, C seeds 7 + 1234 · 5 folds · code 8ca964b).** 5 kernels, ~1 h session each (~5 GPU-h quota). Kaggle allows 2 concurrent GPU sessions, so they were queued. All 25 runs report code `8ca964b tree:7ac61dd0674c`. Findings:
  - **QWK:** C 0.904 ± 0.003 > A 0.898 (1 seed) ≈ B 0.895 ± 0.002. The C–B gap (~0.009) is larger than the seed std but smaller than the ±0.03 rule from §8, so QWK alone does not decide it.
  - **Clinical metrics favour C clearly:** referable sensitivity 0.948 vs 0.900 (B) / 0.897 (A); sight-threatening sensitivity 0.806 vs 0.718 / 0.711. C pays with specificity (referable 0.903 vs 0.949; sight-threatening 0.915 vs 0.950) and accuracy (0.771 vs 0.817). For a screening triage tool, missing disease costs more than an extra referral, so this trade favours C.
  - **A (cascade)** brings no gain over the single softmax B and needs two models. Its cross-fitted Stage-1 threshold lands at 0.45–0.55 (not v1's 0.3). Dropped from further work; the inference API keeps the same output keys either way (the inference contract in `src/retinograde_inference.py`).
  - Fold-val QWK is still slightly optimistic (best epoch picked on the same fold). This affects all candidates equally.
- **2026-11-19 · B vs C at matched operating points (CPU, OOF only; `python src/evaluation/operating_points.py`).** Threshold-free comparison on 2,879 OOF images × 3 seeds. Scores: C raw regression output, B P(grade ≥ k). Paired image bootstrap, C − B:
  - **Referable (≥2):** AUC 0.976 vs **0.981** (−0.005 [−0.008, −0.003]; B wins 15/15 folds). Sensitivity at specificity 0.95: 0.833 vs **0.897** (−0.063 [−0.086, −0.038]). At specificity 0.90: 0.952 vs **0.965**.
  - **Sight-threatening (≥3):** AUC 0.946 vs 0.949 (−0.003 [−0.007, +0.002]). Sensitivity at specificity 0.95: 0.663 vs **0.713** (−0.047 [−0.088, −0.008]). At specificity 0.90: 0.837 vs 0.844 (n.s.).
  - Thresholds cross-fitted on the other folds give the same numbers, so the read-off is not optimistic.
  - **Conclusion:** C's higher sensitivity in the argmax/cut-point table is **only a more lenient operating point** (C's ≥2 cut at 1.5 sits lower on its ROC than B's argmax). It is **not a ranking gain**. At equal specificity, B is as good or better on both tasks. Using B's expected grade instead of P(≥k) gives the same conclusion.
  - **Implication:** referable and sight-threatening flags should come from a dedicated threshold on a continuous score, set for a target specificity (cross-fitted on OOF), not from the grade decision.
  - **Decision (user, 2026-11-19): switch the leading candidate from C to B.**
- **2026-11-19 · Any DR (grade ≥ 1): A's Stage 1 vs B's 1 − P(grade 0) (CPU, OOF; `python src/evaluation/any_dr.py`).** 2,879 images, 1,443 with DR. Paired on seed 42, the only seed A has:
  - **AUC:** A 0.9973 vs B 0.9969; B − A −0.0004 [−0.0013, +0.0005].
  - **Sensitivity at specificity 0.95:** B − A +0.002 [−0.003, +0.007]. At specificity 0.90: +0.002 [0.000, +0.005].
  - **Other seeds:** B seeds 7 / 1234 give AUC 0.9973 / 0.9956.
  - **B with TTA + 3-seed averaging vs A:** AUC +0.0000 [−0.0009, +0.0009]; sensitivity at specificity 0.95 +0.004 [+0.001, +0.008].
  - **Verdict:** no difference. Both are near the ceiling (AUC ≈ 0.997; about 99% sensitivity at specificity 0.95). A's dedicated binary model gives no measurable any-DR advantage. **A vs B is closed: B.**
- **2026-11-20 · E pilot: B at 384px (`rg-train-384-b-pilot`, seed 42, folds 0–1; `python src/evaluation/compare_384.py`).** The 384px data build reproduced the 224px split exactly (meta sha256 identical). Paired against 224px seed 42 on the same 1,142 images:
  - **QWK:** 0.915 vs 0.892, +0.023 [+0.009, +0.037].
  - **AUC:** ≥2 +0.004 [+0.001, +0.008]; ≥3 +0.011 [+0.002, +0.020].
  - **Sensitivity:** ≥3 at specificity 0.95 +0.059 [−0.026, +0.144]; ≥3 at specificity 0.90 +0.043 (not significant).
  - **Seed noise at 224px on the same folds:** QWK −0.000 / +0.012.
  - **Status:** meets the agreed pilot threshold (≈ +0.01 QWK). The full run is on hold pending the headroom check.
- **2026-11-20 · Headroom 1: near-duplicates across folds (`rg-headroom`, CPU; `python src/evaluation/dup_cv.py --cos 0.97 --corr 1.01`).** Training pool only (2,879 images).
  - **Method:** pHash (min over 8 flips/rotations), ImageNet-B0 embedding cosine (original and h-flip) and 64×64 correlation on 160k candidate pairs.
  - **Which measure works:** only the embedding cosine separates real matches. pHash ≤ 6 and correlation ≥ 0.95 match thousands of different healthy eyes. Pair galleries (`docs/figures_private/`, not committed) show same-eye retakes only at cosine ≥ 0.97; 0.93–0.965 pairs are different eyes.
  - **Exact duplicates:** cosine ≈ 1.0, 92 pairs, all inside rg-recon groups and all within one fold, so the grouped split works.
  - **Cross-fold near-duplicates:** 5 pairs (10 images; 1 label conflict). Removing them changes no metric at the third decimal (B QWK 0.895 → 0.895, AUC ≥3 0.949 → 0.949).
  - **Label noise:** 28 of 98 duplicate pairs carry different grades for the same image.
  - **Verdict:** no leakage; the CV numbers stand.
- **2026-11-20 · Headroom 2: B error analysis (OOF, 3-seed average; `python src/evaluation/error_analysis.py`).**
  - **Overall:** QWK 0.893, accuracy 0.820, 518 errors. **75% are adjacent** (|d| = 1); 130 are far (|d| ≥ 2).
  - **Main confusions:** 2→1 (91), 1→2 (81), 2→3 (77), 4→2 (56), 3→2 (39).
  - **Recall:** grade 0 0.99 · 1 0.64 · 2 0.72 · 3 0.53 · 4 0.53.
  - **Confident-wrong:** p ≥ 0.7 with all 3 seeds agreeing, 139 cases.
  - **Image quality:** after reweighting each quartile to the overall grade mix, brightness, contrast, resolution and retina area have **no significant effect** (all CIs include 0). The raw differences were caused by grade mix: APTOS resolution tracks the source clinic and its grade mix, 8% → 77% grade ≥ 2 across resolution quartiles. Only the "sharpest" quartile is worse (+0.08 [+0.03, +0.14]). It is 90% grade 0/1, and its high Laplacian variance looks like noise or compression, not quality.
  - **Gallery** (private): several confident-wrong cases look like **label noise or grader ambiguity**, e.g. laser-treated eyes labelled G3 vs predicted G4, and a G1 label on an eye with a fibrous band. Others look like **features too small at 224px**: confidently missed G1/G2 eyes look normal at this size, which fits the 384px gain. Clinical review is needed before any relabelling.
  - **Review sheet:** `docs/figures_private/label_review_B_oof.csv` (241 rows, with empty reviewer columns; not committed).
- **2026-11-20 · CPU latency, 224 vs 384 (`rg-latency` v3, Kaggle CPU 4 vCPU, batch 1, 30 timed images, median).** The path is read PNG → `preprocess_rgb` → B fold-0 model.

  | | 224px | 384px |
  |---|---|---|
  | Preprocessing | ~95 ms | ~95 ms |
  | Model (saved file, 4 threads) | 448 ms | 559 ms |
  | Total | 544 ms (p90 648) | 655 ms (p90 766) |
  | CPU output vs stored GPU OOF | max Δprob 0.007, 33/33 grades equal | max Δprob 0.006, 33/33 grades equal |

  - 384px costs ~+20% CPU time per image.
  - 1 → 4 threads helps little at batch size 1.
- **2026-11-20 · Incident: float32 model copy gave wrong predictions (caught before any result used it).**
  - **What happened:** to speed up CPU inference I wrote `load_float32`, which rebuilds the model in float32 and copies the weights. On real images only 79–85% of grades matched the saved model (max Δprob 0.9). A rebuild from the model's own config showed the same error, and a synthetic-input self-check did **not** catch it.
  - **Impact:** the first `rg-eyepacs-dev` run used it and was discarded; nothing else was affected. Training, OOF, TTA and the learning curve all use the saved models directly.
  - **Fix:** load the saved model as-is (`src/model_io.py`). It reproduces the GPU predictions on CPU and is only ~15% slower than the float32 path would have been.
  - **Rule:** any change to model loading must be checked against stored predictions on **real** images.
- **2026-11-20 · Headroom 3: learning curve (`rg-train-b-lc`, B 224px seed 42, folds 0–1, ~0.5 GPU-h; `python src/evaluation/learning_curve.py`).** Nested, class-stratified 25/50/75% subsets of each fold's training images; the validation folds stay complete (1,142 images).
  - **QWK:** 0.862 / 0.890 / 0.895 / 0.892 at 25/50/75/100%.
  - **AUC ≥3:** 0.930 / 0.948 / 0.949 / 0.949.
  - **Paired steps:** 25→50% QWK +0.028 [+0.011, +0.046], AUC ≥3 +0.018. 50→75% QWK +0.006 [−0.008, +0.020]. 75→100% QWK −0.004 [−0.017, +0.010], AUC ≥3 −0.000. Sensitivity ≥3 at specificity 0.95 is flat within ±0.09 CIs.
  - **Seed noise at 100%:** ±0.011 QWK.
  - **Verdict:** at 224px B **plateaus from ~50% of the data**. More APTOS-like images at this resolution would not help.
- **2026-11-20 · Headroom 4: domain gap APTOS → EyePACS dev patients (`rg-eyepacs-dev` v2, CPU, prediction only; `python src/evaluation/eyepacs_gap.py`).**
  - **Split:** EyePACS 2015 train, patient-level split fixed by `sha256("eyepacs-split-v1:<patient>")`. dev = 10% of patients (1,766 patients / 3,532 images; grades 2631/260/496/79/66). test = next 10% (1,714 / 3,428), listed by id only and **never opened**. rest = 80% (14,083 / 28,166) unused. IDRiD untouched.
  - **Models:** B seed-42 fold models, loaded as saved; average of the 5 models. CIs from a patient-level bootstrap.

  | | APTOS OOF (s42) | EyePACS dev, hi-res mirror |
  |---|---|---|
  | QWK | 0.896 | **0.432** [0.385–0.483] |
  | AUC any DR (≥1) | 0.997 | **0.687** [0.665–0.712] |
  | AUC referable (≥2) | 0.981 | **0.761** [0.733–0.788] |
  | AUC sight-threatening (≥3) | 0.949 | **0.910** [0.879–0.936] |
  | Sensitivity ≥2 at APTOS threshold (specificity 0.95 on APTOS) | 0.886 | **0.404** (spec 0.926) |
  | Sensitivity ≥3 at APTOS threshold | 0.703 | **0.428** (spec 0.980) |

  - **Grade 0 vs single grade (AUC), APTOS → EyePACS:** G1 0.992 → **0.523** (chance) · G2 0.998 → 0.703 · G3 1.000 → 0.891 · G4 0.999 → 0.962.
  - **Where the errors go on EyePACS:** recall G0 0.88, G1 0.02, G2 0.29, G3 0.13, G4 0.47. 229/260 G1 and 309/496 G2 eyes are called grade 0.
  - **Not a pipeline artifact:** the pre-squeezed 224px mirror gives the same AUCs (Δ ≤ 0.006, n.s.). Preprocessing had 0 failures. CPU loading is verified against GPU OOF (see latency entry).
  - **By EyeQ quality:** AUC ≥3 is 0.911 Good / 0.947 Usable / 0.816 Reject.
  - **384 vs 224** (fold 0+1 models, paired): AUC ≥1 +0.013 [+0.003, +0.020], AUC ≥2 +0.015 [+0.007, +0.023], AUC ≥3 +0.009 [−0.001, +0.019], QWK +0.012 (n.s.). The direction matches APTOS.
  - **Interpretation:**
    - APTOS separates grade 0 from grade 1 almost perfectly, which is not clinically plausible at 224px (mild DR = a few microaneurysms). On EyePACS the same model is at chance.
    - Together with the error analysis (APTOS resolution tracks grade: 8% → 77% grade ≥2 across resolution quartiles), the most likely explanation is a **source/camera shortcut**: in APTOS, healthy and diseased eyes partly come from different clinics or cameras. Part of the gap is also EyePACS label noise and the subtlety of grade 1.
    - Severe disease (G3/G4) transfers reasonably (AUC 0.89–0.96); mild and moderate DR does not.
    - **The APTOS CV numbers overstate real-world performance**, especially for grades 1–2. This is the largest remaining headroom, far larger than anything tuned so far.
- **2026-11-21 · E decided: B at 384px, full 5-fold × 3 seeds (`rg-train-384-b-*`; `python src/evaluation/compare_size.py`).** Seed-42 folds 0–1 were reused from the pilot (identical training code; later code versions only added opt-in options).
  - **384 vs 224:** QWK 0.911 ± 0.001 vs 0.895 ± 0.002, Δ +0.016 [+0.009, +0.023]; 384 is better in 13/15 seed×fold pairs.
  - **Other metrics:** accuracy +0.010; AUC ≥2 +0.003 [+0.001, +0.005]; AUC ≥3 +0.006 [+0.002, +0.011]; sensitivity ≥3 at specificity 0.95 0.714 → 0.742 (n.s.).
  - **Pre-registered rule** (ΔQWK ≥ +0.01 and AUC ≥3 not worse): **ADOPT 384.**
  - **Cost:** CPU latency +20% (655 vs 544 ms per image).
- **2026-11-21 · Mixed-training pilot (`rg-train-mix-b-pilot`, B 384 seed 42 folds 0–1, APTOS fold-train + 8,000 EyePACS rest images; `python src/evaluation/compare_mix.py`).**
  - **Setup:** epoch selection used 0.5 × QWK(APTOS fold-val) + 0.5 × QWK(EyePACS rval, 800 images). EyePACS dev patients were scored after training only. Paired against B 384 seed 42 APTOS-only on the same images:
  - **EyePACS dev** (3,532 images, 1,766 patients; CIs by patient):
    - AUC ≥2: 0.775 → **0.898**, +0.123 [+0.101, +0.147]
    - AUC ≥1: 0.696 → 0.840, +0.144
    - AUC ≥3: 0.916 → 0.969, +0.054
    - QWK: 0.447 → 0.693, +0.246
    - Sensitivity ≥2 at specificity 0.95: 0.374 → 0.654
    - Sensitivity ≥3 at specificity 0.95: 0.600 → 0.862
    - Grade 0 vs 1 AUC: 0.536 → 0.675
  - **APTOS fold-val:** QWK 0.915 → 0.903, −0.012 [−0.028, +0.004]; AUC ≥2 +0.000; AUC ≥3 +0.005; sensitivity ≥3 at specificity 0.95 +0.064 (n.s.).
  - **Pre-registered rule** (EyePACS ΔAUC ≥2 ≥ +0.05 and APTOS ΔQWK ≥ −0.02): **ADOPT mixed training.**
  - **Cost:** ~1 GPU-h per fold (10.3k training images).
- **2026-11-21 · Plan to the freeze (GPU budget ~25 GPU-h; ~12 used so far).** A full 3 × 5 mixed run (~14 GPU-h) would exceed the budget, so:
  1. Seed 42, folds 2–4 (~3 GPU-h): completes a 5-fold OOF set for thresholds, the final epoch count and the ensemble.
  2. Seeds 7 and 1234 on folds 0–1 (~4 GPU-h): a 3-seed paired check of mixed vs APTOS-only (project rule: at least 3 seeds for any claim). APTOS-only seeds 7/1234 are scored on EyePACS dev by `rg-score-dev-b384` (CPU).
  3. The final deployable model trained on the full pool (~1.2 GPU-h).
  4. The one-shot test (~0.3 GPU-h).
  - **Expected total:** ~20.5 GPU-h.
  - **Test authorisation:** the user asked on 2026-11-20 to take the model to its best configuration and stop after the test evaluation. That is the authorisation for the one-shot test.
- **2026-11-21 · Mixed training confirmed over 3 seeds (`rg-train-mix-b-s42`, `rg-train-mix-b-s7-s1234`, `rg-score-dev-b384`; `python src/evaluation/compare_mix3.py`).** B 384, folds 0–1, seeds 7 / 42 / 1234, APTOS-only vs mixed, paired.
  - **EyePACS dev** (3,532 images; seed-averaged Δ, patient bootstrap):
    - AUC ≥2: **+0.117 [+0.095, +0.139]**; every seed improves, 0.775–0.789 → 0.898–0.903
    - AUC ≥1: +0.135 [+0.111, +0.159]
    - AUC ≥3: +0.054 [+0.036, +0.074]
    - QWK: +0.222 [+0.184, +0.262]
    - Sensitivity ≥2 at specificity 0.95: +0.274
    - Sensitivity ≥3 at specificity 0.95: +0.255
  - **APTOS fold-val** (1,142 images): QWK −0.010 [−0.021, +0.000] (seeds +0.008 / −0.012 / −0.025); AUC ≥2 −0.001; AUC ≥3 −0.001.
  - **Verdict:** the pilot's decision holds over 3 seeds. A small APTOS QWK cost (within the −0.02 rule) buys a large gain on a second population.
  - **Seed-42 mixed 5-fold OOF on APTOS:** fold QWKs 0.880 / 0.922 / 0.886 / 0.899 / 0.900. Fold-model EyePACS dev QWK: 0.653–0.696.
  - **Final model epochs:** median best fine-tune epoch over the 9 mixed CV runs = **9**. The full-pool model (`rg-train-mix-b-full`, seed 42, APTOS 2,879 + 8,000 EyePACS rest) fine-tunes for 9 epochs on the same 15-epoch cosine schedule.
- **2026-11-21 · Final model trained on Colab.**
  - **Why:** both Kaggle copies (`rg-train-mix-b-full`, `-full2`) sat in the T4 queue for ~5 h (quota fine, no other sessions), so the user chose Colab. The Kaggle copies were cancelled, so the model was trained exactly once.
  - **How:** `notebooks/colab_final_model.ipynb`, Colab T4, TF 2.21.0 / Keras 3.13.2 pinned to the Kaggle versions, code 7d93e26, same config. 41 min: 3 head + 9 fine-tune epochs, loss 1.23 → 0.62. EyePACS dev QWK 0.667 (fold models 0.653–0.696).
  - **Where:** model file in the private Kaggle dataset `omarashour206/rg-final-model`. Code change for Colab: opt-in `RG_DATA_ROOTS`; Kaggle behaviour unchanged.
- **2026-11-21 · FREEZE (`docs/frozen_config.json`, commit 86f7381, before any test image was scored).**
  - **Model:** `B384-mix-v2`. Primary = `B_full_s42_384_mix`; ensemble = the 5 seed-42 mixed fold models; grade = argmax.
  - **Thresholds:** set at specificity 0.90 on the primary model's EyePACS dev predictions. Referable P(≥2) ≥ 0.327; sight-threatening P(≥3) ≥ 0.028.
- **2026-11-21 · Phase 3: one-shot test (`rg-final-test-cpu`, Kaggle CPU; full tables in `docs/test_results.md`).** Primary model, 95% CIs:

  | test set | QWK | accuracy | AUC ≥2 | AUC ≥3 | referable sens / spec | sight-threat. sens / spec |
  |---|---|---|---|---|---|---|
  | APTOS held-out (733) | **0.898** [0.872–0.922] | **0.838** | 0.985 | 0.945 | 0.950 / 0.938 | 0.908 / 0.762 |
  | IDRiD (516, external) | 0.714 [0.661–0.758] | 0.512 | 0.946 | 0.916 | 0.780 / 0.959 | 0.961 / 0.695 |
  | EyePACS test patients (3,428) | 0.669 [0.630–0.705] | 0.738 | 0.907 | 0.970 | 0.760 / 0.901 | 0.943 / 0.888 |

  - **vs v1 on the same 733 APTOS images** (paired): QWK 0.840 → 0.898, +0.058 [+0.034, +0.093]; accuracy 0.783 → 0.838, +0.055; AUC ≥3 +0.030; referable sensitivity +0.044 at equal specificity. v1's 0.840 was even tuned on these images.
  - **Matches the predictions:** APTOS test QWK 0.898 sits inside the CV range; EyePACS test AUC ≥2 0.907 vs dev 0.898.
  - **Weak points:**
    - Exact grading on IDRiD: accuracy 0.51, with many G2 → G1 and G3 → G2.
    - Grade 1 recall: APTOS 0.68, EyePACS 0.22.
    - Grade 3 recall: 0.26–0.36.
    - The sight-threatening threshold set on EyePACS is lenient on APTOS and IDRiD: high sensitivity (0.91–0.96), specificity 0.70–0.76.
  - **Ensemble vs primary:** within CIs everywhere (EyePACS QWK 0.694 vs 0.669). The single primary model stays the deployable one.
- **2026-11-19 · G = TTA + model averaging for B (`rg-predict-b-tta`, code 7f18cf6, ~5 min GPU).** The 15 saved B models re-scored their own fold-val images with 4 flip views. A 5-fold ensemble has no honest OOF score (each image was in training for 4 of the 5 models), so the 3 seed models of the same fold were averaged as the OOF stand-in (`python src/evaluation/ensemble_gain.py`). Changes vs a single model without TTA (paired bootstrap):
  - **TTA alone:** QWK +0.001 (n.s.); AUC ≥2 +0.001; AUC ≥3 +0.001 (n.s.).
  - **3-model average alone:** QWK −0.002 (n.s.); AUC ≥2 +0.001; AUC ≥3 +0.003.
  - **Both:** QWK +0.005 [−0.001, +0.010]; AUC ≥2 +0.002; AUC ≥3 +0.003. Sensitivity ≥3 at specificity 0.95: 0.714 → 0.727.
  - **Verdict:** measurable but far below the ±0.03 noise rule. TTA costs 4× inference time and an ensemble 5×, and it complicates Grad-CAM. Not adopted as a required part of the model; the 5-fold ensemble is still scored once on test as a secondary row.
- **2026-11-22 · Corrections to earlier notes (no metric changes).**
  - **Seeds:** A (cascade) was trained with 1 seed, so "A vs B" is a single-seed comparison. Mixed vs APTOS-only used 3 seeds on folds 0–1 only. The learning curve is 1 seed on 2 folds.
  - **Test independence:** the 733 APTOS images were v1's validation set and v2 started from some v1 settings, so the APTOS test score may be slightly optimistic. Added to the limitations in `v2_model_report.md`.
  - **Counts:** 11 Kaggle job scripts (not 15); 26 image outputs in the team notebooks (not 17).

## Results

<!-- RESULTS:START -->

_Generated by `src/evaluation/report.py` — do not edit by hand._

### v1 reproduction (old APTOS val, 733 imgs — used for v1 selection, optimistic)

| model | thr | QWK [95% CI] | acc | ref. sens/spec | sight-thr. sens/spec |
|---|---|---|---|---|---|
| v1 cascade | 0.3 | 0.840 [0.807–0.870] | 0.783 | 0.849/0.961 | 0.602/0.948 |

### Candidates — 5-fold OOF, mean ± std over seeds (model-selection numbers)

| config | seeds | QWK | acc | ref. sens | ref. spec | sight-thr. sens | sight-thr. spec |
|---|---|---|---|---|---|---|---|
| A 224px | 1 | 0.898 ± 0.000 | 0.822 ± 0.000 | 0.897 ± 0.000 | 0.948 ± 0.000 | 0.711 ± 0.000 | 0.956 ± 0.000 |
| B 224px | 3 | 0.895 ± 0.002 | 0.817 ± 0.003 | 0.900 ± 0.017 | 0.949 ± 0.010 | 0.718 ± 0.047 | 0.950 ± 0.008 |
| B 224px +TTA | 3 | 0.896 ± 0.003 | 0.820 ± 0.002 | 0.898 ± 0.016 | 0.951 ± 0.007 | 0.710 ± 0.054 | 0.950 ± 0.010 |
| C 224px | 3 | 0.904 ± 0.003 | 0.771 ± 0.012 | 0.948 ± 0.013 | 0.903 ± 0.011 | 0.806 ± 0.024 | 0.915 ± 0.014 |

### OOF per seed (cut points cross-fitted; 95% bootstrap CI)

| config | seed | QWK [CI] | QWK default cuts | ref. sens [CI] | sight-thr. sens [CI] | cut points |
|---|---|---|---|---|---|---|
| A 224px | 42 | 0.898 [0.886–0.908] | 0.897 | 0.897 [0.879–0.915] | 0.711 [0.668–0.758] | {0: 0.55, 1: 0.55, 2: 0.45, 3: 0.55, 4: 0.55} |
| B 224px | 7 | 0.897 [0.886–0.908] | 0.897 | 0.902 [0.885–0.919] | 0.772 [0.730–0.813] |  |
| B 224px +TTA | 7 | 0.896 [0.884–0.907] | 0.896 | 0.898 [0.880–0.914] | 0.772 [0.730–0.814] |  |
| B 224px | 42 | 0.896 [0.884–0.907] | 0.896 | 0.915 [0.899–0.931] | 0.690 [0.643–0.735] |  |
| B 224px +TTA | 42 | 0.899 [0.887–0.910] | 0.899 | 0.915 [0.899–0.930] | 0.671 [0.623–0.718] |  |
| B 224px | 1234 | 0.893 [0.881–0.904] | 0.893 | 0.882 [0.865–0.901] | 0.692 [0.646–0.741] |  |
| B 224px +TTA | 1234 | 0.893 [0.880–0.904] | 0.893 | 0.882 [0.864–0.901] | 0.687 [0.642–0.735] |  |
| C 224px | 7 | 0.906 [0.897–0.915] | 0.904 | 0.960 [0.948–0.971] | 0.830 [0.792–0.867] | [0.586, 1.153, 2.473, 3.482] |
| C 224px | 42 | 0.901 [0.891–0.910] | 0.901 | 0.951 [0.938–0.963] | 0.806 [0.768–0.844] | [0.533, 1.249, 2.578, 3.667] |
| C 224px | 1234 | 0.906 [0.898–0.915] | 0.902 | 0.934 [0.920–0.948] | 0.782 [0.740–0.824] | [0.566, 1.275, 2.545, 3.334] |

### Individual runs (fold-val)

| run_id | QWK | ref. sens/spec | sight-thr. sens/spec | n train/val | min | code |
|---|---|---|---|---|---|---|
| A_f0_s42_224 | 0.894 | 0.922/0.949 | 0.643/0.956 | 2307/572 | 12.9 | 8ca964b tr |
| A_f1_s42_224 | 0.914 | 0.958/0.922 | 0.739/0.961 | 2309/570 | 8.6 | 8ca964b tr |
| A_f2_s42_224 | 0.878 | 0.903/0.957 | 0.716/0.939 | 2305/574 | 8.0 | 8ca964b tr |
| A_f3_s42_224 | 0.897 | 0.855/0.963 | 0.734/0.953 | 2294/585 | 8.1 | 8ca964b tr |
| A_f4_s42_224 | 0.898 | 0.866/0.935 | 0.712/0.970 | 2301/578 | 7.5 | 8ca964b tr |
| B_f0_s1234_224 | 0.853 | 0.867/0.946 | 0.543/0.954 | 2307/572 | 8.6 | 8ca964b tr |
| B_f0_s1234_224_tta | 0.855 | 0.867/0.949 | 0.543/0.956 | 0/572 | 0.2 | 7f18cf6 tr |
| B_f0_s42_224 | 0.872 | 0.927/0.938 | 0.571/0.954 | 2307/572 | 8.6 | 8ca964b tr |
| B_f0_s42_224_tta | 0.875 | 0.904/0.941 | 0.571/0.954 | 0/572 | 0.6 | 7f18cf6 tr |
| B_f0_s7_224 | 0.887 | 0.922/0.958 | 0.743/0.934 | 2307/572 | 8.9 | 8ca964b tr |
| B_f0_s7_224_tta | 0.885 | 0.917/0.958 | 0.700/0.942 | 0/572 | 0.2 | 7f18cf6 tr |
| B_f1_s1234_224 | 0.922 | 0.894/0.958 | 0.841/0.946 | 2309/570 | 5.8 | 8ca964b tr |
| B_f1_s1234_224_tta | 0.921 | 0.903/0.955 | 0.841/0.948 | 0/570 | 0.2 | 7f18cf6 tr |
| B_f1_s42_224 | 0.908 | 0.919/0.940 | 0.739/0.959 | 2309/570 | 6.5 | 8ca964b tr |
| B_f1_s42_224_tta | 0.912 | 0.941/0.940 | 0.705/0.963 | 0/570 | 0.6 | 7f18cf6 tr |
| B_f1_s7_224 | 0.918 | 0.898/0.955 | 0.807/0.946 | 2309/570 | 5.8 | 8ca964b tr |
| B_f1_s7_224_tta | 0.915 | 0.898/0.952 | 0.818/0.942 | 0/570 | 0.2 | 7f18cf6 tr |
| B_f2_s1234_224 | 0.881 | 0.854/0.986 | 0.657/0.947 | 2305/574 | 5.2 | 8ca964b tr |
| B_f2_s1234_224_tta | 0.883 | 0.858/0.989 | 0.672/0.953 | 0/574 | 0.2 | 7f18cf6 tr |
| B_f2_s42_224 | 0.872 | 0.872/0.963 | 0.657/0.945 | 2305/574 | 5.3 | 8ca964b tr |
| B_f2_s42_224_tta | 0.883 | 0.872/0.980 | 0.612/0.951 | 0/574 | 0.4 | 7f18cf6 tr |
| B_f2_s7_224 | 0.871 | 0.872/0.971 | 0.701/0.943 | 2305/574 | 4.4 | 8ca964b tr |
| B_f2_s7_224_tta | 0.868 | 0.867/0.977 | 0.672/0.937 | 0/574 | 0.2 | 7f18cf6 tr |
| B_f3_s1234_224 | 0.905 | 0.915/0.949 | 0.747/0.939 | 2294/585 | 5.1 | 8ca964b tr |
| B_f3_s1234_224_tta | 0.909 | 0.915/0.951 | 0.747/0.939 | 0/585 | 0.2 | 7f18cf6 tr |
| B_f3_s42_224 | 0.913 | 0.911/0.946 | 0.785/0.941 | 2294/585 | 5.6 | 8ca964b tr |
| B_f3_s42_224_tta | 0.915 | 0.923/0.951 | 0.785/0.933 | 0/585 | 0.4 | 7f18cf6 tr |
| B_f3_s7_224 | 0.914 | 0.928/0.954 | 0.810/0.937 | 2294/585 | 4.6 | 8ca964b tr |
| B_f3_s7_224_tta | 0.912 | 0.919/0.957 | 0.810/0.939 | 0/585 | 0.2 | 7f18cf6 tr |
| B_f4_s1234_224 | 0.894 | 0.878/0.932 | 0.630/0.978 | 2301/578 | 4.1 | 8ca964b tr |
| B_f4_s1234_224_tta | 0.886 | 0.866/0.932 | 0.589/0.980 | 0/578 | 0.2 | 7f18cf6 tr |
| B_f4_s42_224 | 0.909 | 0.945/0.900 | 0.671/0.980 | 2301/578 | 4.6 | 8ca964b tr |
| B_f4_s42_224_tta | 0.904 | 0.933/0.900 | 0.658/0.982 | 0/578 | 0.3 | 7f18cf6 tr |
| B_f4_s7_224 | 0.893 | 0.891/0.932 | 0.781/0.941 | 2301/578 | 4.4 | 8ca964b tr |
| B_f4_s7_224_tta | 0.895 | 0.887/0.929 | 0.836/0.933 | 0/578 | 0.2 | 7f18cf6 tr |
| C_f0_s1234_224 | 0.880 | 0.894/0.929 | 0.714/0.904 | 2307/572 | 10.3 | 8ca964b tr |
| C_f0_s42_224 | 0.889 | 0.908/0.921 | 0.814/0.884 | 2307/572 | 10.3 | 8ca964b tr |
| C_f0_s7_224 | 0.895 | 0.908/0.927 | 0.829/0.896 | 2307/572 | 9.1 | 8ca964b tr |
| C_f1_s1234_224 | 0.914 | 0.903/0.931 | 0.818/0.940 | 2309/570 | 6.1 | 8ca964b tr |
| C_f1_s42_224 | 0.914 | 0.907/0.928 | 0.841/0.913 | 2309/570 | 5.0 | 8ca964b tr |
| C_f1_s7_224 | 0.918 | 0.919/0.928 | 0.864/0.892 | 2309/570 | 5.8 | 8ca964b tr |
| C_f2_s1234_224 | 0.898 | 0.876/0.951 | 0.791/0.933 | 2305/574 | 5.2 | 8ca964b tr |
| C_f2_s42_224 | 0.884 | 0.885/0.931 | 0.776/0.913 | 2305/574 | 5.4 | 8ca964b tr |
| C_f2_s7_224 | 0.884 | 0.854/0.945 | 0.776/0.919 | 2305/574 | 5.3 | 8ca964b tr |
| C_f3_s1234_224 | 0.907 | 0.885/0.943 | 0.797/0.917 | 2294/585 | 5.5 | 8ca964b tr |
| C_f3_s42_224 | 0.907 | 0.923/0.931 | 0.886/0.883 | 2294/585 | 4.6 | 8ca964b tr |
| C_f3_s7_224 | 0.911 | 0.885/0.943 | 0.785/0.927 | 2294/585 | 5.2 | 8ca964b tr |
| C_f4_s1234_224 | 0.906 | 0.861/0.924 | 0.808/0.949 | 2301/578 | 5.4 | 8ca964b tr |
| C_f4_s42_224 | 0.907 | 0.903/0.924 | 0.849/0.915 | 2301/578 | 5.4 | 8ca964b tr |
| C_f4_s7_224 | 0.906 | 0.895/0.924 | 0.781/0.923 | 2301/578 | 5.5 | 8ca964b tr |

<!-- RESULTS:END -->
