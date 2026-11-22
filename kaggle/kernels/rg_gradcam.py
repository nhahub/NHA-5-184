"""
rg-gradcam (Kaggle, CPU only, ~30 min). Grad-CAM on the REAL frozen v2 model + IDRiD lesion validation.
ANALYSIS ONLY: nothing here changes the model, the thresholds or any setting. No grading metric is
computed on IDRiD and no test split is read (EyePACS images come from the DEV patients).

  A. sanity check: maps of the trained model vs randomised copies (head / top block / all layers)
  B. gallery: 5 EyePACS dev images per true grade (25), for the predicted grade and for the referral decision
  C. IDRiD segmentation set (81 images with expert lesion masks: MA, HE, EX, SE + optic disc):
     C1 alignment proof: the optic disc mask must sit on bright pixels, hard-exudate masks on bright/yellow
        pixels, haemorrhage/microaneurysm masks on dark pixels, all in the model's 384px frame
     C2 pointing game, energy, precision/recall@k of Grad-CAM vs random / retina-centre / optic-disc
        baselines and vs the Grad-CAM of a fully randomised model (negative control)
Outputs: gradcam_summary.json, gradcam_idrid.csv (per image), gallery/*.png (private, gitignored).
"""
import glob
import json
import os
import re
import sys
import time
import zipfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image

CONFIG = {}  # <- injected by make_kernel.py

OUT, INPUT = "/kaggle/working", "/kaggle/input"
GAL = f"{OUT}/gallery"
os.makedirs(GAL, exist_ok=True)


def bootstrap_code():
    for root, _, files in os.walk(INPUT):
        if "retinograde_inference.py" in files and os.path.basename(root) == "src":
            break
    else:
        z = glob.glob(f"{INPUT}/**/src.zip", recursive=True)[0]
        zipfile.ZipFile(z).extractall(f"{OUT}/_code")
        root = f"{OUT}/_code/src" if os.path.isdir(f"{OUT}/_code/src") else f"{OUT}/_code"
    sys.path[:0] = [root, os.path.join(root, "training")]
    v = glob.glob(f"{INPUT}/**/CODE_VERSION.txt", recursive=True)
    return open(v[0]).read().strip() if v else "unknown"


CODE_VERSION = bootstrap_code()
import retinograde_inference as R  # noqa: E402
from data import aptos_v2 as D  # noqa: E402
from explain import gradcam as G  # noqa: E402
from explain import lesion_overlap as L  # noqa: E402

T0 = time.time()
TARGETS = ("pred", "referable")


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def put(img, text, scale=0.42):
    img = img.copy()
    for i, t in enumerate(text.split("\n")):
        for col, th in (((0, 0, 0), 3), ((255, 255, 0), 1)):
            cv2.putText(img, t, (4, 14 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)
    return img


def save_rows(rows, name, per_page):
    for k in range(0, len(rows), per_page):
        page = np.concatenate(rows[k:k + per_page], 0)
        cv2.imwrite(f"{GAL}/{name}_{k // per_page}.png", cv2.cvtColor(page, cv2.COLOR_RGB2BGR))


def contour(img, mask, colour, th=1):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cv2.drawContours(img.copy(), cs, -1, colour, th)


# ----------------------------------------------------------------------------- EyePACS dev images
def dev_images(m):
    """5 dev images per true grade: 3 the model grades correctly, 2 it gets wrong (stored dev predictions)."""
    z = np.load(D.find_file("eyepacs_dev_preds.npz"))
    pred = z["probs5"].argmax(1)
    rng = np.random.default_rng(7)
    pick = []
    for g in range(5):
        ok = np.flatnonzero((z["y"] == g) & (pred == g))
        bad = np.flatnonzero((z["y"] == g) & (pred != g))
        a = rng.choice(ok, min(3, len(ok)), replace=False) if len(ok) else []
        b = rng.choice(bad, min(5 - len(a), len(bad)), replace=False) if len(bad) else []
        pick += list(a) + list(b)
    want = {str(z["stem"][i]): int(z["y"][i]) for i in pick}
    paths = {}
    for dp, _, fns in os.walk(INPUT):
        if "resized_train" in dp and "cropped" not in dp:
            for fn in fns:
                k = re.match(r"^(\d+_(left|right))\.(jpe?g|png)$", fn, re.I)
                if k and k.group(1) in want:
                    paths[k.group(1)] = os.path.join(dp, fn)
    return [(s, want[s], m.prepare(paths[s])) for s in want if s in paths]


# ----------------------------------------------------------------------------- IDRiD
def idrid_items():
    imgs, masks = {}, {}
    for dp, _, fns in os.walk(INPUT):
        if "segmentation" not in dp.lower():
            continue
        for fn in fns:
            a = re.match(r"^(IDRiD_\d+)\.jpe?g$", fn, re.I)
            b = re.match(r"^(IDRiD_\d+)_(MA|HE|EX|SE|OD)\.tiff?$", fn, re.I)
            if a:
                imgs[a.group(1)] = os.path.join(dp, fn)
            elif b:
                masks.setdefault(b.group(1), {})[b.group(2).upper()] = os.path.join(dp, fn)
    return [(k, imgs[k], masks.get(k, {})) for k in sorted(imgs)]


BAD_MASKS = []


def load_mask(path, shape):
    """IDRiD masks are RGB, RGBA or palette TIFs. Only the colour channels count: an opaque alpha channel would
    mark every pixel as lesion (this happened for IDRiD_81 in the first run)."""
    im = Image.open(path)
    a = np.array(im.convert("RGB") if im.mode in ("RGBA", "P", "CMYK", "LA") else im)
    a = (a > 0).any(axis=2) if a.ndim == 3 else a > 0
    if a.shape != shape:
        raise ValueError(f"{path}: mask {a.shape} vs image {shape}")
    if a.mean() > 0.5:                 # a lesion mask never covers half of the photo
        BAD_MASKS.append(f"{os.path.basename(path)} covers {a.mean():.2f} (mode {im.mode})")
        return np.zeros(shape, np.uint8)
    return a.astype(np.uint8)


def idrid(m, rand, summary):
    fm, fr = G._feature_model(G._parts(m.model)[0]), G._feature_model(G._parts(rand)[0])
    items = idrid_items()
    summary["idrid"] = {"images": len(items), "masks_per_type": {t: sum(t in k for _, _, k in items) for t in (*L.TYPES, "OD")}}
    log(summary["idrid"])
    rows, align, proof, ex = [], [], [], []
    for n, (name, path, mk) in enumerate(items):
        rgb = R.load_rgb(path)
        img01, geom = R.preprocess_rgb(rgb, R.IMG_SIZE, return_geometry=True)
        x = R.to_model_input(img01)
        retina = L.retina_mask(x)
        near = {t: R.transform_mask(load_mask(p, rgb.shape[:2]), geom).astype(bool) for t, p in mk.items()}
        keep = {t: R.transform_mask(load_mask(p, rgb.shape[:2]), geom, keep_small=True).astype(bool) for t, p in mk.items()}
        lesion = np.zeros(retina.shape, bool)
        for t in L.TYPES:
            if t in keep:
                lesion |= keep[t]
        od = near.get("OD", np.zeros(retina.shape, bool))

        # C1. alignment numbers on the pixels the model sees (green channel = best lesion contrast)
        green, red = x[..., 1], x[..., 0]
        rec = {"image": name}
        for t, mask in {**{k: near[k] for k in near}}.items():
            mm = mask & retina
            if mm.sum() >= 5:
                ring = L.dilate(mm, 8) & retina & ~L.dilate(mm, 2)
                rec[f"{t}_in"] = float(green[mm].mean())
                rec[f"{t}_ring"] = float(green[ring].mean()) if ring.any() else np.nan
                # control: the same mask mirrored left-right must lose the contrast
                fl = mm[:, ::-1] & retina
                fr_ring = L.dilate(fl, 8) & retina & ~L.dilate(fl, 2)
                rec[f"{t}_contrast_mirrored"] = float(green[fl].mean() - green[fr_ring].mean()) if fl.sum() >= 5 and fr_ring.any() else np.nan
        rec["MA_pixels_nearest"], rec["MA_pixels_keep_small"] = int(near["MA"].sum()) if "MA" in near else 0, int(keep["MA"].sum()) if "MA" in keep else 0
        align.append(rec)

        # C2. heatmaps and metrics
        heats, p = {}, None
        for tg in TARGETS:
            heats[tg], p, used = G.gradcam(m.model, x, tg, fm)
            if tg == "pred":
                grade = used
        neg = G.gradcam(rand, x, "referable", fr)[0]
        maps = {"gradcam_pred": heats["pred"], "gradcam_referable": heats["referable"], "random_weights": neg,
                "centre": L.baseline_map("centre", retina), "optic_disc": L.baseline_map("optic_disc", retina, od)}
        base = {"image": name, "grade": int(grade), "p_referable": float(p[2:].sum()), "p_st": float(p[3:].sum()),
                "referable_flag": bool(p[2:].sum() >= R.REFERABLE_THRESHOLD), "lesion_px": int((lesion & retina).sum()),
                **{f"{t}_px": int((keep[t] & retina).sum()) if t in keep else 0 for t in L.TYPES}}
        if lesion.any():
            for tol in (0, 16):
                for kind, h in maps.items():
                    rows.append({**base, "tol": tol, "map": kind, **L.metrics(h, lesion, retina, tol)})
                rows.append({**base, "tol": tol, "map": "random", **L.random_expectation(lesion, retina, tol)})
            for kind in ("gradcam_referable", "gradcam_pred", "centre", "optic_disc"):
                reg = L.top_k_region(maps[kind], retina, 20)
                for r in rows:
                    if r["image"] == name and r["map"] == kind:
                        for t in L.TYPES:
                            if t in keep and (keep[t] & retina).any():
                                r[f"recall20_{t}"] = float(reg[keep[t] & retina].mean())

        # pictures: preprocessed image | masks | Grad-CAM (referral) with lesion contours | Grad-CAM (pred)
        u8 = np.clip(x, 0, 255).astype(np.uint8)
        col = {"MA": (255, 0, 0), "HE": (160, 0, 255), "EX": (255, 255, 0), "SE": (0, 255, 255), "OD": (0, 255, 0)}
        seg = u8.copy()
        for t in (*L.TYPES, "OD"):
            if t in near:
                seg = contour(seg, keep[t] if t != "OD" else od, col[t])
        ov_r, ov_p = G.overlay(x, heats["referable"]), G.overlay(x, heats["pred"])
        row_img = np.concatenate([put(u8, f"{name}  model grade {grade}\nP(>=2) {p[2:].sum():.2f}  P(>=3) {p[3:].sum():.2f}"),
                                  put(seg, "masks: MA red HE violet\nEX yellow SE cyan OD green"),
                                  put(contour(ov_r, lesion, (255, 255, 255)), "Grad-CAM referral + lesions"),
                                  put(contour(ov_p, lesion, (255, 255, 255)), "Grad-CAM predicted grade")], 1)
        energy = next((r["energy"] for r in rows if r["image"] == name and r["map"] == "gradcam_referable" and r["tol"] == 16), np.nan)
        ex.append((energy, row_img))
        if n < 8:
            zoom = []
            for t in ("OD", "EX", "HE"):
                if t in near and (near[t] & retina).any():
                    ys, xs = np.nonzero(near[t] & retina)
                    cy, cx = int(ys.mean()), int(xs.mean())
                    y0, x0 = np.clip(cy - 48, 0, 288), np.clip(cx - 48, 0, 288)
                    crop = contour(u8, near[t], col[t])[y0:y0 + 96, x0:x0 + 96]
                    zoom.append(put(cv2.resize(crop, (384, 384), interpolation=cv2.INTER_NEAREST), f"{name} {t} zoom x4"))
            while len(zoom) < 3:
                zoom.append(np.zeros((384, 384, 3), np.uint8))
            proof.append(np.concatenate([put(seg, name + " all masks")] + zoom, 1))
        if n % 20 == 0:
            log(name, "done")

    al = pd.DataFrame(align)
    summary["alignment"] = {}
    for t in ("OD", "EX", "SE", "HE", "MA"):
        if f"{t}_in" in al:
            c = (al[f"{t}_in"] - al[f"{t}_ring"]).dropna()
            mir = al[f"{t}_contrast_mirrored"].dropna()
            summary["alignment"][t] = {"images": int(len(c)), "mean_contrast_inside_minus_ring": float(c.mean()),
                                       "images_with_expected_sign": int((c > 0).sum() if t in ("OD", "EX", "SE") else (c < 0).sum()),
                                       "mean_contrast_if_mask_mirrored": float(mir.mean())}
    summary["bad_masks_ignored"] = BAD_MASKS
    summary["small_lesions"] = {"MA_pixels_nearest_resize": int(al.MA_pixels_nearest.sum()), "MA_pixels_keep_small": int(al.MA_pixels_keep_small.sum())}
    pd.DataFrame(rows).to_csv(f"{OUT}/gradcam_idrid.csv", index=False)
    save_rows(proof, "idrid_alignment_proof", 4)
    ex = [e for e in ex if np.isfinite(e[0])]
    ex.sort(key=lambda e: -e[0])
    save_rows([r for _, r in ex[:8]], "idrid_best", 4)
    save_rows([r for _, r in ex[-12:]], "idrid_worst", 4)
    mid = len(ex) // 2
    save_rows([r for _, r in ex[mid - 4:mid + 4]], "idrid_middle", 4)


if __name__ == "__main__":
    summary = {"code_version": CODE_VERSION, "model_name": R.MODEL_NAME}
    m = R.RetinoGrade(D.find_file(os.path.basename(R.DEFAULT_MODEL_PATH)))
    dev = dev_images(m)
    log("dev images", len(dev))

    # the Grad-CAM module re-applies the head by hand: its probabilities must equal the model's
    diff = max(float(np.abs(G.gradcam(m.model, x, "pred")[1] - m.probabilities(x[None])[0]).max()) for _, _, x in dev[:8])
    summary["gradcam_probs_vs_model_max_diff"] = diff

    # A. sanity
    summary["sanity"] = {}
    for tg in TARGETS:
        summary["sanity"][tg] = G.randomization_check(m.model, [x for _, _, x in dev], tg, retina_fn=L.retina_mask)
        s = summary["sanity"][tg]
        log("sanity", tg, {k: round(float(np.nanmean(v)), 3) for k, v in s.items() if isinstance(v, list)})

    # B. gallery
    fm = G._feature_model(G._parts(m.model)[0])
    rows = []
    for stem, y, x in dev:
        hp, p, g = G.gradcam(m.model, x, "pred", fm)
        hr = G.gradcam(m.model, x, "referable", fm)[0]
        u8 = np.clip(x, 0, 255).astype(np.uint8)
        rows.append((y, np.concatenate([put(u8, f"true grade {y}  predicted {g}  p={p[g]:.2f}\nP(>=2) {p[2:].sum():.2f}  P(>=3) {p[3:].sum():.3f}"),
                                        put(G.overlay(x, hp), f"Grad-CAM predicted grade {g}"),
                                        put(G.overlay(x, hr), "Grad-CAM referral (>=2)")], 1)))
    rows.sort(key=lambda r: r[0])
    save_rows([r for _, r in rows], "eyepacs_dev_gallery", 5)
    summary["gallery"] = {"n": len(rows), "per_true_grade": np.bincount([y for y, _ in rows], minlength=5).tolist()}

    # C. IDRiD
    idrid(m, G.randomized_copy(m.model, "all"), summary)
    summary["minutes"] = round((time.time() - T0) / 60, 1)
    with open(f"{OUT}/gradcam_summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    log(json.dumps({k: v for k, v in summary.items() if k != "sanity"}, default=float)[:3000])
