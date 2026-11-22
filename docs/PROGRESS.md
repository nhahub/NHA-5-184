# Progress (Phase 4)

**In short:** live status of the current build (steps 1–5). "Done and verified" means it was run against the real frozen v2 model. Everything after step 5 waits for a new go-ahead.

Status values: **Done and verified · Built but unverified · In progress · Blocked · Not started**. Last update: 2026-11-22 (step 2 finished).

| # | Item | Status | Evidence | What remains | Needed from the team |
|---|---|---|---|---|---|
| 0 | Model v2 (frozen, one-shot test) | Done and verified | `docs/v2_model_report.md`, `docs/test_results.md` | – | – |
| 1 | Inference on v2 at 384px | Done and verified | `rg-infer-check`: 120/120 images bit-exact with training, 299/300 grades and 300/300 flags equal to the stored predictions; 164–214 ms per image on CPU; `tests/test_inference_v2.py` (19 pass) · [component](components/inference.md) | Local run of `tests/test_real_model_v2.py` on the laptop (TensorFlow environment being set up) | Review and merge the PR |
| 2 | Grad-CAM on v2 + IDRiD lesion validation | Done and verified | `rg-gradcam` (real model, EyePACS dev + IDRiD, code 220f041): sanity check on 25 dev images, 25-image gallery (private), 81 IDRiD images; referral heatmap hits a lesion in 32% vs 5% by chance, enrichment 4.1×; `tests/test_explain.py` · [component](components/gradcam.md) | – | Read the IDRiD rules page before publishing any image; decide whether to use IDRiD pictures in the report |
| 3 | Image-quality gate (EyeQ) | In progress | filename mapping: 12,543/12,543 names match and the DR grade agrees in 100% | picture-level check in the hi-res mirror, training, evaluation, integration | – |
| 4 | Confidence calibration + "needs human review" | Not started | | | choose the operating points |
| 5 | Streamlit app | Not started | | | – |
| – | FastAPI, Docker, MLflow, full test suite, Azure, monitoring, report, slides | Not started | | not part of this round | new go-ahead |
