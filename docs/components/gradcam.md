# Component: Grad-CAM and IDRiD lesion validation

**In short:** Grad-CAM shows which regions of the photo pushed the model towards a decision. On 81 IDRiD images with expert lesion masks, the heatmap for the *referral decision* points at real lesions far more often than chance: the hottest pixel lies on a lesion in **32%** of images, against 5% expected by chance, 11% for a "look at the centre" baseline and 0% for an optic-disc baseline. The heatmap is coarse and is not a lesion detector.

## What it does
[`src/explain/gradcam.py`](../../src/explain/gradcam.py) produces a heatmap for one of two questions:
- **predicted grade** (`target="pred"`): what made the model choose this grade?
- **referral decision** (`target="referable"`): what made it refer the eye (log P(grade ≥ 2))?

The EfficientNet backbone is nested inside the model, so the code takes the last convolutional feature map and re-applies the classification head by hand to get gradients. On 8 images the hand-made head gives the same probabilities as the model to within 0.00004.

[`src/explain/lesion_overlap.py`](../../src/explain/lesion_overlap.py) compares a heatmap with lesion masks, using plain pixel measures, no TensorFlow.

## How to run
```bash
# everything below runs as a private Kaggle CPU job (kaggle/kernels/rg_gradcam.py, ~8 min)
python kaggle/make_kernel.py build rg-gradcam && python kaggle/make_kernel.py push rg-gradcam
python kaggle/make_kernel.py pull rg-gradcam --gallery          # gallery -> docs/figures_private/ (gitignored)
python src/evaluation/gradcam_idrid.py                          # tables below, from the per-image CSV
```

## Does the map depend on what the model learned? (sanity check)
Grad-CAM was computed on 25 EyePACS **dev** images (5 per true grade; images the model never trained on), for both targets, with the trained model and with copies whose weights were re-initialised from the top down (Adebayo et al.). Numbers: Spearman correlation inside the retina.

| Randomised | predicted grade | referral |
|---|---|---|
| dense head only | +0.25 | −0.21 |
| dense head + last block | +0.07 | +0.11 |
| all layers | +0.08 | +0.00 |
| *(reference: maps of two different images)* | +0.07 | +0.07 |

*Note (2026-11-22):* the first version of this table ranked tied zeros by position in the rank correlation. That was fixed (`spearman`) and the job was run again (`rg-gradcam`, code `1e2ecd8`); the averages moved by at most 0.02, so the conclusion is the same. In the "last block" row, one of the 25 images has a constant map and is left out, for both targets.

A map that ignored the model would stay near +1. They drop to the level of unrelated maps, so the explanations do depend on the learned weights. The one caution: for the predicted grade, randomising only the last layer still leaves a weak +0.25.

## IDRiD validation
Data: the 81 IDRiD segmentation images, masks for microaneurysms (MA), haemorrhages (HE), hard exudates (EX) and soft exudates (SE), plus the optic disc. License: see below. **Analysis only.** No grading metric is computed on IDRiD, and the model and its thresholds are not changed.

**First, are mask and heatmap aligned?** The masks go through the same crop / padding / resize as the photo (`transform_mask`). Checked numerically on the pixels the model sees: the optic disc is brighter than its surroundings in **81/81** images, hard exudates in **81/81**, soft exudates in **40/40** (that type exists in 40 images), haemorrhages darker in **80/80**, microaneurysms darker in **81/81**; mirroring the masks left–right destroys the contrast. Pictures with mask contours drawn on the photo: `docs/figures_private/rg-gradcam/gallery/idrid_alignment_proof_*.png`. Lesions smaller than one 384px pixel would vanish in the resize, so `transform_mask(..., keep_small=True)` keeps any lesion that touches a pixel (it kept 30,485 MA pixels against 12,970 with plain resizing).

**Strict numbers (undilated lesions, n = 81; mean over images):**

| Heatmap | hottest pixel on a lesion | share of heat on lesions | precision in the hottest 10% | lesions inside the hottest 20% | enrichment in the hottest 10% |
|---|---|---|---|---|---|
| **Grad-CAM, referral** | **0.32** | **0.140** | **0.166** | **0.573** | **4.0×** |
| Grad-CAM, predicted grade | 0.26 | 0.087 | 0.128 | 0.369 | 2.4× |
| Grad-CAM, randomised model | 0.03 | 0.049 | 0.040 | 0.162 | 0.7× |
| Baseline: retina centre | 0.11 | 0.070 | 0.077 | 0.285 | 1.5× |
| Baseline: optic disc | 0.00 | 0.023 | 0.010 | 0.073 | 0.2× |
| Baseline: random | 0.05 | 0.048 | 0.048 | 0.200 | 1.0× |

Grad-CAM (referral) minus each baseline, with 95% CIs from a paired bootstrap over images: random +0.27 [+0.18, +0.37] on the pointing game, retina centre +0.21 [+0.09, +0.33], optic disc +0.32 [+0.22, +0.42], randomised model +0.30 [+0.20, +0.40]. Every CI excludes 0.

With a 16-pixel tolerance (one Grad-CAM cell is 32 px; lesions then cover 47% of the retina) Grad-CAM referral is still first: 0.96 pointing / 2.0× enrichment, random 0.47 / 1.0×. The strict table is the honest headline, because the tolerant one is easy to score well on.

**By lesion type** (share of the lesion pixels inside the hottest 20%): hard exudates 0.64, soft exudates 0.61, haemorrhages 0.46, **microaneurysms 0.35**. **By lesion load:** hit rate 0.19 (few lesions), 0.37, 0.41 (many lesions); enrichment 4.2×, 4.4×, 3.3×.

## Honest conclusion
- The referral heatmap is **meaningfully linked to lesions**: far better than chance, than a centre bias, than the optic disc and than a randomised model. It works best on exudates.
- It is **not a precise detector.** Two images in three still have their hottest pixel off every marked lesion (strict). Microaneurysms are mostly missed: they are 1–3 pixels at this resolution and the heatmap grid is 12×12.
- The map for the **predicted grade is weaker and noisier** (hottest pixel on a lesion in 26%). In the EyePACS dev gallery some predicted-grade maps light up the image border or the optic disc, so **part of the attention is not on lesions**. This is the main risk to state.
- IDRiD is one dataset, all images contain lesions, and the number of images is small.
- **Corrected on 2026-11-22:** the first version of the retina mask spilled into the black corners of some images (75.8% of the square on average). It was replaced and the whole job was run again (`rg-gradcam`, code `8f15ccf`). The changes are small: enrichment 4.1× → 4.0×, chance level 4.6% → 4.8%, hottest pixel on a lesion unchanged at 32%. A second rerun (code `1e2ecd8`, for the sanity check) reproduced all IDRiD numbers exactly.

## Failure analysis
Looked at the 81 images and at the numbers (`src/evaluation/gradcam_idrid.py`, plus the private galleries).

- **Real misses are rare.** With a 16-pixel tolerance, the hottest pixel of the referral map is away from every marked lesion in **3 of 81** images: IDRiD_24 (P(≥2) = 0.10), IDRiD_26 (grade 3, P(≥2) = 0.997: a genuine miss of the explanation), and IDRiD_29, which the model grades 0 although it has lesions (a *grading* error, so the map explains a healthy call).
- **A strict hit (0 tolerance) fails mostly for a technical reason:** lesions are a few pixels wide while one Grad-CAM cell is 32 px, so a peak that sits *next to* a lesion counts as a miss. The "worst by energy" pictures include maps that clearly sit on the lesion cluster (IDRiD_12, IDRiD_38, IDRiD_44).
- **The map for the predicted grade is the weak one:** for images the model grades 0 or 1 (27 of 81) it never lands strictly on a lesion (0%); its hits come from grades 2–4 (36%, 42%, 67%). Several of these maps light up the **optic disc** (IDRiD_52) or the **image border** (IDRiD_41, IDRiD_62). This is the clearest sign that part of what the model uses is not a lesion, and it is the honest limitation to state.
- **The referral map depends on the decision:** it hits a lesion strictly in 40% of the images flagged referable (60 images), but in only 10% of those not flagged (21 images), where there is little referable signal to explain.

## License (checked 2026-11-22)
IDRiD's own IEEE DataPort page, the Zenodo mirror and the Kaggle copy state **CC BY 4.0**: use, sharing and slides are allowed with the citation Porwal et al., *Indian Diabetic Retinopathy Image Dataset (IDRiD)*, IEEE DataPort 2018, doi:10.21227/H25W98. The challenge's Rules page and the MDPI data paper could not be read automatically, so extra conditions there are unconfirmed. Until a person has read <https://idrid.grand-challenge.org/Rules/>, no IDRiD image is used in documents, screenshots or the app.

## Why this design
- **Two targets, same model:** the grade explanation and the referral explanation can differ; a clinician mainly needs the referral one.
- **Baselines that could fool us:** a heatmap that always looks at the centre would score well on some measures. Comparing against centre, optic disc, random and a randomised model makes "better than chance" a fair claim.
- **Strict masks first:** the dilated version exists only for information.

## Five examiner questions
1. **Is Grad-CAM proof that the model uses lesions?** No; it is evidence. The map beats every baseline on the pointing game, but it can be wrong and it does not show what the model would do without the lesion.
2. **Why is the pointing score only 32%?** With strict masks, the hottest pixel must lie exactly on a marked lesion; the heatmap grid is 12×12, lesions can be a few pixels, and masks are incomplete. Against 5% by chance it is a clear signal.
3. **How do you know the masks line up with the images?** The optic disc mask lies on bright pixels in 81/81 images, the exudate masks on bright ones, haemorrhages on dark ones, and mirrored masks lose the contrast.
4. **What if the model had random weights?** The randomised-model heatmap scores like the random baseline (hottest pixel on a lesion in 3%), and the sanity check shows maps change when weights do.
5. **Could you show a patient this heatmap?** As a research illustration with the disclaimer, yes. It is coarse and should not be used to decide where a lesion is.
