"""
rg-eyeq-verify (Kaggle, CPU only, ~15 min). Is an EyeQ quality label really attached to the right PICTURE
in the hi-res EyePACS mirror we use? Verification only; opens dev + rest patients, never test patients.

  1. names   : every EyeQ name of a dev/rest patient exists in both mirrors
  2. labels  : the DR grade stored in EyeQ equals EyePACS trainLabels.csv for the same name
  3. pictures: the same name shows the same eye in both mirrors (two independent uploads of EyePACS).
               Each hi-res thumbnail must find its own low-res namesake as best match among 400 candidates
  4. meaning : Reject images should be measurably worse (darker / less sharp) than Good ones
  5. gallery : 10 Good / Usable / Reject examples to look at (private, gitignored)
Output: eyeq_verify_summary.json, gallery/eyeq_*.png
"""
import glob
import hashlib
import json
import os
import re
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

CONFIG = {}  # <- injected by make_kernel.py

OUT, INPUT = "/kaggle/working", "/kaggle/input"
GAL = f"{OUT}/gallery"
os.makedirs(GAL, exist_ok=True)
N_PAIRS, N_STATS = 600, 450
QUAL = {0: "Good", 1: "Usable", 2: "Reject"}


def u(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16) / 16 ** 15


def split_of(patient):
    x = u(f"eyepacs-split-v1:{patient}")
    return "dev" if x < 0.10 else ("test" if x < 0.20 else "rest")


def index(pred):
    out = {}
    for dp, _, fns in os.walk(INPUT):
        if pred(dp):
            for fn in fns:
                m = re.match(r"^(\d+_(left|right))\.(jpe?g|png)$", fn, re.I)
                if m:
                    out.setdefault(m.group(1), os.path.join(dp, fn))
    return out


def thumb(path, thr=12):
    """Black border removed at grey level `thr` (None = no crop), squeezed to 48x48 grey, normalised.
    The low-res mirror is squeezed to a square, so both are compared as squares."""
    im = cv2.imread(path)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if thr is not None:
        ys, xs = np.nonzero(g > thr)
        if len(ys) > 200:
            g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    t = cv2.resize(g, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (t - t.mean()) / (t.std() + 1e-6)


def thumbs(path):
    return np.stack([thumb(path, t) for t in (None, 12, 25, 45)])


def stats(path):
    im = cv2.imread(path)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (512, int(512 * g.shape[0] / g.shape[1])), interpolation=cv2.INTER_AREA)
    m = g > 10
    inner = cv2.erode(m.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    lap = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1), cv2.CV_64F)
    return float(g[m].mean()) if m.any() else 0.0, float(lap[inner].var()) if inner.any() else 0.0


if __name__ == "__main__":
    s = {}
    q = pd.read_csv(next(p for p in glob.glob(f"{INPUT}/**/eyeq_mapping.csv", recursive=True)))
    q["stem"] = q.image.str.replace(r"\.\w+$", "", regex=True)
    q["patient"] = q.stem.str.split("_").str[0]
    q["split"] = q.patient.map(split_of)
    q = q[(q.eyeq_split == "train") & (q.split != "test")].reset_index(drop=True)        # test patients excluded
    s["n_eyeq_dev_rest"] = len(q)

    hi, lo = index(lambda d: "resized_train" in d and "cropped" not in d), index(lambda d: "colored_images" in d)
    s["names_in_hires"], s["names_in_lowres"] = int(q.stem.isin(hi).sum()), int(q.stem.isin(lo).sum())

    lab = pd.read_csv(next(p for p in glob.glob(f"{INPUT}/**/trainLabels.csv", recursive=True)))
    lab = dict(zip(lab.image.astype(str).str.replace(r"\.\w+$", "", regex=True), lab.level.astype(int)))
    s["dr_grade_equals_trainLabels"] = int((q.stem.map(lab) == q.DR_grade).sum())

    rng = np.random.default_rng(0)
    pick = q.iloc[np.sort(rng.choice(len(q), N_PAIRS, replace=False))]
    with Pool(os.cpu_count()) as p:
        A = np.stack(p.map(thumbs, [hi[x] for x in pick.stem], chunksize=8))          # (n, 4 croppings, 2304)
        B = np.stack(p.map(thumbs, [lo[x] for x in pick.stem], chunksize=8))
    d = A.shape[2]
    # similarity of two pictures = best over the 4x4 cropping combinations
    C = np.max([A[:, i] @ B[:, j].T / d for i in range(4) for j in range(4)], axis=0)
    own, other = np.diag(C), C[~np.eye(len(C), dtype=bool)]
    best = C.argmax(1)
    miss = np.flatnonzero(best != np.arange(len(C)))
    s["pictures"] = {"n": N_PAIRS, "own_corr_median": float(np.median(own)), "own_corr_p01": float(np.percentile(own, 1)),
                     "own_corr_min": float(own.min()), "other_corr_median": float(np.median(other)),
                     "other_corr_p99": float(np.percentile(other, 99)),
                     "own_namesake_is_best_match": int(len(C) - len(miss)),
                     "misses": [{"stem": pick.stem.iat[i], "own": float(own[i]), "best": float(C[i].max())} for i in miss]}
    lowest = np.argsort(own)[:12]
    tiles = []
    for i in lowest:                       # the 12 least similar namesakes, hi-res squeezed like the low-res mirror
        st = pick.stem.iat[i]
        a_, b_ = cv2.imread(hi[st]), cv2.imread(lo[st])
        t = np.hstack([cv2.resize(a_, (200, 200), interpolation=cv2.INTER_AREA), cv2.resize(b_, (200, 200))])
        cv2.putText(t, f"{st} r={own[i]:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 255), 1)
        tiles.append(t)
    cv2.imwrite(f"{GAL}/eyeq_lowest_namesakes.png", np.vstack([np.hstack(tiles[k:k + 3]) for k in range(0, 12, 3)]))
    print(s, flush=True)

    rows = []
    for k in range(3):
        g = q[q.quality == k]
        g = g.iloc[rng.choice(len(g), min(N_STATS // 3, len(g)), replace=False)]
        with Pool(os.cpu_count()) as p:
            st = np.array(p.map(stats, [hi[x] for x in g.stem], chunksize=8))
        rows.append({"quality": QUAL[k], "n": len(g), "brightness_median": float(np.median(st[:, 0])),
                     "sharpness_median": float(np.median(st[:, 1])), "brightness_p10": float(np.percentile(st[:, 0], 10)),
                     "sharpness_p10": float(np.percentile(st[:, 1], 10))})
        tiles = []
        for x in g.stem.iloc[:10]:
            im = cv2.imread(hi[x])
            h, w = im.shape[:2]
            r = 200 / max(h, w)
            t = np.zeros((200, 200, 3), np.uint8)
            im = cv2.resize(im, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)
            t[:im.shape[0], :im.shape[1]] = im
            tiles.append(t)
        cv2.imwrite(f"{GAL}/eyeq_{QUAL[k]}.png", np.vstack([np.hstack(tiles[:5]), np.hstack(tiles[5:])]))
    s["quality_vs_measurements"] = rows
    with open(f"{OUT}/eyeq_verify_summary.json", "w") as f:
        json.dump(s, f, indent=1)
    print(json.dumps(s, indent=1))
