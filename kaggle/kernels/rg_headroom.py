"""
rg-headroom (Kaggle, CPU only, ~20 min). Headroom check items 1 + 2, TRAIN POOL ONLY (2,879 images).
The 733 held-out test images are never loaded (rows with split != "train" are skipped).

1. Near-duplicates across CV folds by image content (beyond rg-recon's 64x64-correlation >= 0.98 groups):
   * perceptual hash (DCT pHash, 64 bit), min Hamming distance over the 8 flips/rotations
   * ImageNet EfficientNetB0 embeddings (cosine), max over {original, h-flip}
   * 64x64 grey correlation (min over flips/rotations) for every candidate pair
   -> dup_candidates.csv  +  gallery/dups_*.png (pairs, to decide the threshold by eye)
2. Image statistics for error analysis (computed from the RAW APTOS file + the preprocessed array):
   raw width/height, retina brightness/contrast, sharpness (variance of Laplacian), retina area,
   preprocessed brightness  -> image_stats.csv
   + gallery/wrong_*.png: confident-wrong and far-grade (|error| >= 2) OOF cases of model B
     (probabilities averaged over the 3 seeds; OOF files from the rg-train-b-* kernels)

Everything written here stays in this PRIVATE kernel's output; the PNGs are pulled only into the
gitignored docs/figures_private/ folder for the team's label review.
"""
import glob
import json
import os
import sys
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

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
from data import aptos_v2 as D  # noqa: E402


# ----------------------------------------------------------------------------- helpers
def dihedral(img):
    """8 flips/rotations of an HxW(xC) array."""
    out = []
    for k in range(4):
        r = np.rot90(img, k)
        out += [r, r[:, ::-1]]
    return out


def phash_bits(gray32):
    d = cv2.dct(gray32.astype(np.float32))[:8, :8].ravel()
    return d > np.median(d[1:])


def gray(img, size):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (size, size), interpolation=cv2.INTER_AREA)


def hamming_matrix(A, B):
    """A, B: (N, 64) bool -> (N, N) Hamming distances."""
    a, b = A.astype(np.int32), B.astype(np.int32)
    return 64 - (a @ b.T + (1 - a) @ (1 - b).T)


def corr64_min(x, y):
    """Correlation of 64x64 grey images, best over the 8 dihedral variants of y."""
    gx = gray(x, 64).astype(np.float32).ravel()
    gx = (gx - gx.mean()) / (gx.std() + 1e-6)
    best = -1.0
    for v in dihedral(gray(y, 64)):
        gv = v.astype(np.float32).ravel()
        gv = (gv - gv.mean()) / (gv.std() + 1e-6)
        best = max(best, float((gx * gv).mean()))
    return best


def raw_stats(path):
    """Quality-like statistics of the RAW image (before CLAHE)."""
    im = cv2.imread(path)
    if im is None:
        return {}
    h, w = im.shape[:2]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask = g > 10
    scale = 512.0 / max(h, w)                              # sharpness at a fixed scale
    gs = cv2.resize(g, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(mask.astype(np.uint8), gs.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    ms = cv2.erode(ms.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0   # ignore the retina border
    lap = cv2.Laplacian(gs, cv2.CV_64F)
    return {"raw_w": w, "raw_h": h,
            "raw_brightness": float(g[mask].mean()) if mask.any() else 0.0,
            "raw_contrast": float(g[mask].std()) if mask.any() else 0.0,
            "raw_sharpness": float(lap[ms].var()) if ms.any() else 0.0,
            "retina_frac": float(mask.mean())}


def label(img, text, scale=0.38):
    img = img.copy()
    for i, t in enumerate(text.split("\n")):
        cv2.putText(img, t, (3, 12 + 12 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), 1, cv2.LINE_AA)
    return img


def save_grid(tiles, path, ncol):
    th, tw = tiles[0].shape[:2]
    while len(tiles) % ncol:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + ncol], 1) for i in range(0, len(tiles), ncol)]
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))


# ----------------------------------------------------------------------------- 1. duplicates
def duplicates(X, meta, summary):
    import tensorflow as tf
    n = len(meta)
    imgs = [np.asarray(X[i]) for i in meta.row]

    # pHash: original of i vs 8 dihedral variants of j
    H0 = np.stack([phash_bits(gray(im, 32)) for im in imgs])
    Hmin = np.full((n, n), 64, np.int32)
    for t in range(8):
        Ht = np.stack([phash_bits(dihedral(gray(im, 32))[t]) for im in imgs])
        Hmin = np.minimum(Hmin, hamming_matrix(H0, Ht))

    # ImageNet embeddings (0-255 input, Keras EfficientNet normalises internally)
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", pooling="avg")

    def embed(batch_fn):
        out = []
        for i in range(0, n, 64):
            b = np.stack([batch_fn(im) for im in imgs[i:i + 64]]).astype(np.float32)
            out.append(base.predict_on_batch(b))
        e = np.concatenate(out)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    E, Ef = embed(lambda im: im), embed(lambda im: im[:, ::-1])
    S = np.maximum(E @ E.T, E @ Ef.T)
    np.fill_diagonal(S, -1)
    np.fill_diagonal(Hmin, 64)

    # candidate pairs: 3 nearest neighbours by embedding, or very close hash / embedding
    cand = set()
    nn = np.argsort(-S, axis=1)[:, :3]
    for i in range(n):
        for j in nn[i]:
            cand.add((min(i, int(j)), max(i, int(j))))
    ii, jj = np.where(np.triu((S >= 0.90) | (Hmin <= 10), 1))
    cand |= set(zip(ii.tolist(), jj.tolist()))
    rows = []
    for i, j in sorted(cand):
        rows.append({"id_a": meta.id_code.iat[i], "id_b": meta.id_code.iat[j],
                     "grade_a": int(meta.grade.iat[i]), "grade_b": int(meta.grade.iat[j]),
                     "fold_a": int(meta.fold.iat[i]), "fold_b": int(meta.fold.iat[j]),
                     "same_recon_group": bool(meta.dup_group.iat[i] == meta.dup_group.iat[j]),
                     "emb_cos": round(float(S[i, j]), 4), "phash_ham": int(Hmin[i, j]),
                     "corr64": round(corr64_min(imgs[i], imgs[j]), 4)})
    df = pd.DataFrame(rows)
    df["cross_fold"] = df.fold_a != df.fold_b
    df.to_csv(f"{OUT}/dup_candidates.csv", index=False)
    summary["dups"] = {
        "n_images": n, "n_candidate_pairs": len(df),
        "emb_cos_hist": np.histogram(S[np.triu_indices(n, 1)], bins=[-1, .8, .85, .9, .93, .95, .97, .98, .99, 1.01])[0].tolist(),
        "cross_fold_pairs_by_threshold": {
            f"cos>={t}": int(((df.emb_cos >= t) & df.cross_fold).sum()) for t in (0.90, 0.93, 0.95, 0.97, 0.98)},
        "cross_fold_pairs_ham<=6": int(((df.phash_ham <= 6) & df.cross_fold).sum()),
        "cross_fold_pairs_corr>=0.95": int(((df.corr64 >= 0.95) & df.cross_fold).sum()),
    }

    # gallery: strongest cross-fold pairs, then samples per similarity band (to set the threshold by eye)
    idx = {c: k for k, c in enumerate(meta.id_code)}
    th = 150

    def pair_tile(r):
        a = cv2.resize(imgs[idx[r.id_a]], (th, th))
        b = cv2.resize(imgs[idx[r.id_b]], (th, th))
        txt = f"cos {r.emb_cos:.3f} ham {r.phash_ham} r {r.corr64:.2f}\nG{r.grade_a}/G{r.grade_b} f{r.fold_a}/f{r.fold_b}"
        return label(np.concatenate([a, b], 1), txt)

    cf = df[df.cross_fold].sort_values("emb_cos", ascending=False)
    top = cf.head(60)
    for k in range(0, len(top), 20):
        save_grid([pair_tile(r) for r in top.iloc[k:k + 20].itertuples()], f"{GAL}/dups_top_{k // 20}.png", 4)
    rng = np.random.default_rng(0)
    for lo, hi in [(0.85, 0.90), (0.90, 0.93), (0.93, 0.96), (0.96, 1.01)]:
        band = cf[(cf.emb_cos >= lo) & (cf.emb_cos < hi)]
        if len(band):
            pick = band.iloc[rng.choice(len(band), min(12, len(band)), replace=False)]
            save_grid([pair_tile(r) for r in pick.itertuples()], f"{GAL}/dups_band_{lo:.2f}-{hi:.2f}.png", 4)


# ----------------------------------------------------------------------------- 2. stats + wrong-case gallery
def image_stats(meta, summary):
    d = next(dp for dp, dns, fns in os.walk(INPUT) if "train.csv" in fns and "train_images" in dns)
    paths = [f"{d}/train_images/{i}.png" for i in meta.id_code]
    with Pool(os.cpu_count()) as p:
        st = p.map(raw_stats, paths, chunksize=16)
    df = pd.DataFrame(st)
    df.insert(0, "id_code", meta.id_code.values)
    summary["stats_rows"] = len(df)
    return df


def wrong_gallery(X, meta, stats, summary):
    probs = {}
    for seed in (42, 7, 1234):
        for f in range(5):
            p = glob.glob(f"{INPUT}/**/B_f{f}_s{seed}_224/oof_preds.npz", recursive=True)
            assert len(p) == 1, (f, seed, p)
            z = np.load(p[0])
            for i, pr in zip(z["id_code"], z["probs5"]):
                probs.setdefault(str(i), []).append(pr)
    m = meta.set_index("id_code")
    P = np.stack([np.mean(probs[i], 0) for i in meta.id_code])
    agree = np.array([len({int(np.argmax(p)) for p in probs[i]}) == 1 for i in meta.id_code])
    y, pred, conf = meta.grade.to_numpy(), P.argmax(1), P.max(1)
    stats = stats.copy()
    stats["pred"], stats["conf"], stats["seeds_agree"] = pred, conf, agree
    stats.to_csv(f"{OUT}/image_stats.csv", index=False)

    rowidx = dict(zip(meta.id_code, meta.row))
    th = 180

    def tile(i):
        im = cv2.resize(np.asarray(X[rowidx[meta.id_code.iat[i]]]), (th, th))
        txt = (f"{meta.id_code.iat[i]}\ntrue G{y[i]}  pred G{pred[i]}  p={conf[i]:.2f}\n"
               + " ".join(f"{v:.2f}" for v in P[i]))
        return label(im, txt, 0.34)

    cw = np.where((pred != y) & (conf >= 0.7) & agree)[0]
    cw = cw[np.argsort(-conf[cw])][:60]
    far = np.where(np.abs(pred - y) >= 2)[0]
    far = far[np.argsort(-conf[far])][:60]
    for name, sel in (("wrong_confident", cw), ("wrong_far", far)):
        for k in range(0, len(sel), 20):
            save_grid([tile(i) for i in sel[k:k + 20]], f"{GAL}/{name}_{k // 20}.png", 5)
    summary["wrong"] = {"confident_wrong(conf>=0.7, 3 seeds agree)": int(((pred != y) & (conf >= 0.7) & agree).sum()),
                        "far_errors(|d|>=2)": int((np.abs(pred - y) >= 2).sum()), "n": int(len(y))}


if __name__ == "__main__":
    X, meta_all = D.load()
    D.check_splits(meta_all)
    meta = meta_all.reset_index().rename(columns={"index": "row"})
    meta = meta[meta.split == "train"].reset_index(drop=True)          # train pool only
    assert len(meta) == 2879, len(meta)
    summary = {"code_version": CODE_VERSION, "n_train_pool": len(meta)}
    duplicates(X, meta, summary)
    st = image_stats(meta, summary)
    wrong_gallery(X, meta, st, summary)
    with open(f"{OUT}/headroom_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=int)
    print(json.dumps(summary, indent=1, default=int))
