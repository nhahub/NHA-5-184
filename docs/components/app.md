# Component: Streamlit demo app

**In short:** a one-page demo. Upload a fundus photo; the app checks its quality, then shows the DR grade (0–4), the class probabilities, a calibrated confidence, whether a human should look at it, the referable and sight-threatening flags, and a Grad-CAM heatmap. It is a **research prototype for decision support, not a medical device**; the disclaimer is shown on the page and with every result.

## How to run (from a fresh clone)
Python 3.11–3.13, CPU only.
```bash
git clone https://github.com/OmarAshour1/Diabetic-Retinopathy-Project.git && cd Diabetic-Retinopathy-Project
python -m venv .venv-app
.venv-app\Scripts\activate                      # Windows   (Linux/macOS: source .venv-app/bin/activate)
pip install -r requirements-app.txt             # TensorFlow 2.21, Keras 3.13, Streamlit 1.48, ...
```
Put the two model files in `models/` (not in the repo; see `models/README.md`), or point to them:
```bash
set RG_MODEL=C:\path\B_full_s42_384_mix_model.keras          # grading model (81 MB)
set RG_QUALITY_MODEL=C:\path\quality_b0_224.keras            # quality gate (17 MB)
streamlit run app/streamlit_app.py
```
The server listens on **localhost only** (`.streamlit/config.toml`), so other devices cannot reach it, and it sends no usage statistics. Open http://localhost:8501. If the quality model is missing the app still works but says clearly that image quality was *not* checked; if the grading model is missing it shows an error naming `RG_MODEL`.

## What the page shows
1. **Upload**: PNG or JPG up to 25 MB. Corrupt files, non-images, files under 64 pixels and over 25 MB give a message instead of a crash.
2. **Quality**: Good (continue), Usable (continue with a warning and a review notice) or Reject (a new photo is requested and *no grade is shown*).
3. **Result**: grade 0–4 with its name, class probabilities, calibrated confidence, the flags "Referable DR" (grade ≥ 2) and "Sight-threatening DR" (grade ≥ 3), and the *needs human review* notice with its reasons.
4. **Heatmap**: Grad-CAM for the predicted grade or the referral decision (a toggle). It is coarse and does not outline lesions.
5. **All output values**: the full JSON, for debugging.

## Evidence
- Headless tests of the page with the real models, `tests/test_app.py` (7 tests, TensorFlow environment): waiting message, a photo produces grade/flags/confidence/disclaimer, the heatmap toggle, bad/tiny/oversized files explained, and a photo rejected by the real gate shows no grade.
- A server start check: the health endpoint answers 200.
- First use by a person in a browser (2026-11-22, one real photo without a known grade): quality message, grade, flags and both heatmaps worked. It showed one display bug, now fixed: heat was painted as speckles on the black background (`display_mask` in `src/explain/gradcam.py`). Reject, warning and wrong-file paths have only been tried by the automatic tests. No photo of a patient is in the repository.

## Why this design
- **Streamlit** is the simplest tool the team can read and present, and it runs on a laptop CPU.
- **The quality gate comes first** because the grading model is measurably unreliable on poor photos (see the quality component: far errors on Reject photos are about 4× those on Good photos).
- **No grade for a rejected photo**: showing a number for an image the system calls ungradable invites misuse.
- **Flags and grade use raw probabilities** (frozen and tested); the calibrated confidence is display only.
- **Localhost by default**, no data stored: photos live in memory for the session.

## Limits
- One photo at a time, no user accounts, no storage.
- Calibration and the review rule are proposals awaiting the team's decision (`docs/components/calibration_review.md`).
- The referral flags can be on although the grade is lower, because they use their own thresholds.

## Five examiner questions
1. **What happens with a blurry photo?** The quality gate labels it Usable (warning and review notice) or Reject (asks for a new photo and shows no grade).
2. **Why not put the app online?** This is a research prototype; it handles medical photos, so it runs locally for now. Deployment is a later, separate decision.
3. **Why does the confidence say "calibrated"?** The raw confidence of the model is adjusted by temperature scaling fitted on cross-validation data; it changes only the displayed number, never the grade.
4. **What does "needs human review" mean?** The model is unsure (top probability below 50%), the referral decision is borderline, or the photo quality is only Usable. A person should decide.
5. **Can a wrong heatmap mislead users?** Yes, that's why the page says it is coarse and does not outline lesions, and the disclaimer says a professional must review every result.
