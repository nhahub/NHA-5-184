"""
rg-recon (Kaggle, CPU only, ~0.5-1 h). Read-only reconnaissance of every dataset we depend on.

Outputs (/kaggle/working, all small):
  recon.json          every check below (machine-readable)
  recon_summary.md    the same, human-readable
  xval_map.csv        X_val row -> APTOS id_code (+ pixel diff vs preprocess_rgb re-run)
  xtrain_map.csv      X_train row -> id_code + dihedral transform (originals vs augmented copies)
  dup_groups.csv      perceptual-hash near-duplicate groups over the 3,662 APTOS train images
  dup_pairs.csv       candidate pairs with hamming distance + correlation
  eyeq_mapping.csv    EyeQ quality labels matched to the EyePACS mirror files (labels/paths only)
  pip_freeze.txt      exact Kaggle environment (used to pin requirements.txt)
  dup_examples.png    visual check of top near-duplicate pairs (stays on Kaggle, not downloaded)
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
import traceback
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

OUT = "/kaggle/working"
INPUT = "/kaggle/input"
R = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
MD = []


# ----------------------------------------------------------------------------- bootstrap repo code
def bootstrap_code():
    for root, _, files in os.walk(INPUT):
        if "retinograde_inference.py" in files and os.path.basename(root) == "src":
            break
    else:
        zips = glob.glob(f"{INPUT}/**/src.zip", recursive=True)
        if not zips:
            raise RuntimeError("retinograde-code dataset not attached")
        zipfile.ZipFile(zips[0]).extractall(f"{OUT}/_code")
        root = f"{OUT}/_code/src" if os.path.isdir(f"{OUT}/_code/src") else f"{OUT}/_code"
    sys.path[:0] = [root, os.path.join(root, "training")]
    v = glob.glob(f"{INPUT}/**/CODE_VERSION.txt", recursive=True)
    R["code_version"] = open(v[0]).read().strip() if v else "unknown"
    return root


bootstrap_code()
from retinograde_inference import preprocess_rgb, load_rgb  # noqa: E402


def save():
    with open(f"{OUT}/recon.json", "w") as f:
        json.dump(R, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(f"{OUT}/recon_summary.md", "w") as f:
        f.write("\n".join(MD) + "\n")


def section(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            print(f"\n######## {name}")
            try:
                res = fn(*a, **k)
            except Exception:
                R[name] = {"error": traceback.format_exc()}
                MD.append(f"## {name}\n**ERROR**\n```\n{traceback.format_exc()[-1500:]}\n```")
                print(traceback.format_exc())
                res = None
            R.setdefault("timing_s", {})[name] = round(time.time() - t0, 1)
            save()
            return res
        return run
    return deco


def find_dir(pred, root=INPUT):
    for dp, dns, fns in os.walk(root):
        if pred(dp, dns, fns):
            return dp
    return None


# ----------------------------------------------------------------------------- 1. environment
@section("env")
def env():
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    open(f"{OUT}/pip_freeze.txt", "w").write(freeze)
    import sklearn
    vers = {"python": sys.version.split()[0], "numpy": np.__version__, "opencv": cv2.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__}
    try:
        import tensorflow as tf
        import keras
        vers.update(tensorflow=tf.__version__, keras=keras.__version__)
    except Exception as e:
        vers["tensorflow"] = f"import failed: {e}"
    R["env"] = vers
    R["inputs"] = sorted({os.path.relpath(dp, INPUT) for dp, _, _ in os.walk(INPUT)
                          if os.path.relpath(dp, INPUT).count(os.sep) <= 2})
    MD.append("## Environment\n" + "\n".join(f"- {k}: {v}" for k, v in vers.items()))


# ----------------------------------------------------------------------------- 2. legacy arrays
@section("arrays")
def arrays():
    d = find_dir(lambda dp, dns, fns: "X_val.npy" in fns and "y_train.npy" in fns)
    Xtr = np.load(f"{d}/X_train.npy", mmap_mode="r")
    Xva = np.load(f"{d}/X_val.npy", mmap_mode="r")
    ytr = np.load(f"{d}/y_train.npy")
    yva = np.load(f"{d}/y_val.npy")
    sample = np.asarray(Xva[:64], dtype=np.float32)
    corners = np.concatenate([sample[:, :6, :6].ravel(), sample[:, -6:, -6:].ravel()]) * 255
    on_grid = float(np.mean(np.abs(sample * 255 - np.round(sample * 255)) < 1e-3))
    R["arrays"] = {
        "dir": d, "X_train": [list(Xtr.shape), str(Xtr.dtype)], "X_val": [list(Xva.shape), str(Xva.dtype)],
        "y_train_bincount": np.bincount(ytr.astype(int), minlength=5).tolist(),
        "y_val_bincount": np.bincount(yva.astype(int), minlength=5).tolist(),
        "X_val_min_max": [float(sample.min()), float(sample.max())],
        "background_level_0_255_percentiles_50_95_99": np.percentile(corners, [50, 95, 99]).round(2).tolist(),
        "fraction_values_on_uint8_grid": on_grid,      # 1.0 => uint8 storage is lossless
    }
    MD.append("## Legacy arrays\n```\n" + json.dumps(R["arrays"], indent=1) + "\n```")
    return d


# ----------------------------------------------------------------------------- 3. APTOS raw -> preprocess all
def _prep(path):
    img = preprocess_rgb(load_rgb(path), 224)   # v1 arrays are 224px; the inference default is now 384
    return np.round(img * 255).astype(np.uint8)


def thumbs(u8):
    """(N,224,224,3) uint8 -> (N, 16*16*3) float32 (INTER_AREA commutes with flips/rot90 on 14x14 blocks)."""
    return np.stack([cv2.resize(x, (16, 16), interpolation=cv2.INTER_AREA) for x in u8]).reshape(len(u8), -1).astype(np.float32)


def dihedral(x, t):
    """t in 0..7 on (H,W,C) or (N,H,W,C) images: t//4 = h-flip first, t%4 = rot90 count."""
    ax = (-3, -2)
    if t >= 4:
        x = np.flip(x, axis=-2)
    return np.rot90(x, t % 4, axes=ax)


@section("aptos_raw")
def aptos_raw():
    d = find_dir(lambda dp, dns, fns: "train.csv" in fns and "train_images" in dns)
    df = pd.read_csv(f"{d}/train.csv")
    paths = [f"{d}/train_images/{i}.png" for i in df.id_code]
    with Pool(os.cpu_count()) as p:
        u8 = np.stack(p.map(_prep, paths, chunksize=16))
    R["aptos_raw"] = {"dir": d, "n": len(df), "bincount": np.bincount(df.diagnosis, minlength=5).tolist(),
                      "duplicate_id_codes": int(df.id_code.duplicated().sum())}
    MD.append(f"## APTOS raw\n- {len(df)} images, grades {R['aptos_raw']['bincount']}")
    return df, u8, paths


# ----------------------------------------------------------------------------- 4. legacy preprocessing + provenance
# v1 recon (2026-11-19): only 280/733 X_val rows were pixel-exact under the current preprocess_rgb.
# The legacy guide (archive/docs/legacy/Data_Preprocessing_Guide.md) crops the black border and then
# STRETCHES to 224 (no white-frame crop, no pad-to-square). We test candidate variants, keep the one
# that reproduces X_val exactly, and map every legacy row to an APTOS id with a one-to-one assignment.
def _crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs:
        x, y, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea))
        img = img[y:y + h, x:x + w]
    return img


def _crop_white(img, rule=True):
    area = img.shape[0] * img.shape[1]
    m = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 15, 230)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs:
        x, y, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea))
        if not rule or w * h >= 0.7 * area:
            img = img[y:y + h, x:x + w]
    return img


def _pad_square(img):
    h, w = img.shape[:2]
    s = max(h, w)
    t, l = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, t, s - h - t, l, s - w - l, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def _resize_clahe(img, size=224):
    img = cv2.resize(img, (size, size))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


VARIANTS = {
    "legacy_guide(crop_black+stretch)": lambda im: _resize_clahe(_crop_black(im)),
    "crop_black+pad": lambda im: _resize_clahe(_pad_square(_crop_black(im))),
    "crop_black+crop_white(rule)+stretch": lambda im: _resize_clahe(_crop_white(_crop_black(im))),
    "crop_black+crop_white(no rule)+pad": lambda im: _resize_clahe(_pad_square(_crop_white(_crop_black(im), False))),
}


def _prep_variant(args):
    path, name = args
    return VARIANTS[name](load_rgb(path))


def _nn_exact_rate(rows_u8, cand_u8):
    rt, ct = thumbs(rows_u8), thumbs(cand_u8)
    d2 = (rt ** 2).sum(1)[:, None] + (ct ** 2).sum(1)[None] - 2 * rt @ ct.T
    j = d2.argmin(1)
    md = np.array([np.abs(rows_u8[i].astype(np.int16) - cand_u8[j[i]]).max() for i in range(len(j))])
    return float(np.mean(md <= 1))


@section("legacy_variant_search")
def legacy_variant_search(arr_dir, df, paths, u8_current):
    Xva = np.load(f"{arr_dir}/X_val.npy", mmap_mode="r")
    va_u8 = np.round(np.asarray(Xva, dtype=np.float32) * 255).astype(np.uint8)
    rates = {"current(preprocess_rgb)": _nn_exact_rate(va_u8, u8_current)}
    best_name, best_u8 = "current(preprocess_rgb)", u8_current
    for name in VARIANTS:
        with Pool(os.cpu_count()) as p:
            u = np.stack(p.map(_prep_variant, [(pt, name) for pt in paths], chunksize=16))
        rates[name] = _nn_exact_rate(va_u8, u)
        print(f"  {name}: X_val exact rate {rates[name]:.3f}")
        if rates[name] > rates[best_name]:
            best_name, best_u8 = name, u
        else:
            del u
    R["legacy_variant_search"] = {"x_val_exact_rate_by_variant": rates, "best": best_name}
    MD.append("## Legacy preprocessing variant search (fraction of X_val rows reproduced pixel-exactly)\n```\n"
              + json.dumps(R["legacy_variant_search"], indent=1) + "\n```")
    return best_name, best_u8


@section("provenance")
def provenance(arr_dir, df, legacy_u8):
    """One-to-one assignment of all legacy rows (X_val + X_train originals) to the 3,662 APTOS ids."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.model_selection import train_test_split

    grade = df.diagnosis.to_numpy()
    Xva = np.load(f"{arr_dir}/X_val.npy", mmap_mode="r")
    Xtr = np.load(f"{arr_dir}/X_train.npy", mmap_mode="r")
    yva = np.load(f"{arr_dir}/y_val.npy").astype(int)
    ytr = np.load(f"{arr_dir}/y_train.npy").astype(int)

    # X_train: find each row's dihedral transform (augmented copies = rot90/flip of an original)
    raw_t8 = np.stack([thumbs(dihedral(legacy_u8, t).copy()) for t in range(8)])
    norms = (raw_t8 ** 2).sum(-1)
    tr_t = np.zeros((len(Xtr), raw_t8.shape[-1]), np.float32)
    tr_transform = np.zeros(len(Xtr), int)
    tr_nn = np.zeros(len(Xtr), int)
    for s in range(0, len(Xtr), 256):
        b = np.round(np.asarray(Xtr[s:s + 256], dtype=np.float32) * 255).astype(np.uint8)
        bt = thumbs(b)
        tr_t[s:s + len(b)] = bt
        d2 = (bt ** 2).sum(1)[None, :, None] + norms[:, None, :] - 2 * np.einsum("bk,tnk->tbn", bt, raw_t8)
        k = d2.transpose(1, 0, 2).reshape(len(b), -1).argmin(1)
        tr_transform[s:s + len(b)], tr_nn[s:s + len(b)] = np.divmod(k, len(legacy_u8))

    va_u8 = np.round(np.asarray(Xva, dtype=np.float32) * 255).astype(np.uint8)
    orig_rows = np.where(tr_transform == 0)[0]
    rows_t = np.vstack([thumbs(va_u8), tr_t[orig_rows]])
    rows_y = np.concatenate([yva, ytr[orig_rows]])
    ids_t = raw_t8[0]
    d2 = (rows_t ** 2).sum(1)[:, None] + (ids_t ** 2).sum(1)[None] - 2 * rows_t @ ids_t.T
    cost = np.sqrt(np.maximum(d2, 0) / rows_t.shape[1]) + 5.0 * (rows_y[:, None] != grade[None, :])
    r_idx, c_idx = linear_sum_assignment(cost)
    assign = np.full(len(rows_t), -1)
    assign[r_idx] = c_idx

    def diffs(img, j):
        if j < 0:
            return 255, 255.0
        d = np.abs(img.astype(np.int16) - legacy_u8[j])
        return int(d.max()), round(float(d.mean()), 4)

    nv = len(va_u8)
    vrows = []
    for i in range(nv):
        mx, mn = diffs(va_u8[i], assign[i])
        vrows.append((i, df.id_code.values[assign[i]] if assign[i] >= 0 else None,
                      int(grade[assign[i]]) if assign[i] >= 0 else -1, int(yva[i]), mx, mn))
    vm = pd.DataFrame(vrows, columns=["row", "id_code", "grade_csv", "y_val", "max_abs_diff", "mean_abs_diff"])
    vm.to_csv(f"{OUT}/xval_map.csv", index=False)

    trows = []
    orig_assign = dict(zip(orig_rows, assign[nv:]))
    for r in range(len(Xtr)):
        j = orig_assign.get(r, tr_nn[r]) if tr_transform[r] == 0 else tr_nn[r]
        b = np.round(np.asarray(Xtr[r], dtype=np.float32) * 255).astype(np.uint8)
        d = np.abs(b.astype(np.int16) - dihedral(legacy_u8[j], tr_transform[r]))
        trows.append((r, df.id_code.values[j], int(tr_transform[r]), int(d.max()), round(float(d.mean()), 4),
                      int(ytr[r]), int(grade[j])))
    tm = pd.DataFrame(trows, columns=["row", "id_code", "transform", "max_abs_diff", "mean_abs_diff",
                                      "y_train", "grade_csv"])
    tm.to_csv(f"{OUT}/xtrain_map.csv", index=False)

    val_ids, tr_orig_ids = set(vm.id_code), set(tm.id_code[tm["transform"] == 0])
    _, ref_val = train_test_split(df, test_size=0.2, stratify=df.diagnosis, random_state=42)
    res = {
        "x_val": {"rows": nv, "unique_ids": len(val_ids), "exact(max<=1)": int((vm.max_abs_diff <= 1).sum()),
                  "label_agreement": float((vm.grade_csv == vm.y_val).mean()),
                  "same_id_set_as_train_test_split_0.2_strat_42": val_ids == set(ref_val.id_code)},
        "x_train": {"rows": len(tm), "transform_counts": tm["transform"].value_counts().sort_index().to_dict(),
                    "originals": int((tm["transform"] == 0).sum()), "unique_original_ids": len(tr_orig_ids),
                    "exact(max<=1)": int((tm.max_abs_diff <= 1).sum()),
                    "label_agreement": float((tm.y_train == tm.grade_csv).mean()),
                    "augmented_rows_by_label": np.bincount(tm.y_train[tm["transform"] != 0], minlength=5).tolist()},
        "coverage": {"ids_assigned": len(val_ids | tr_orig_ids), "of": len(df),
                     "val_train_id_overlap": len(val_ids & tr_orig_ids)},
    }
    R["provenance"] = res
    MD.append("## Legacy arrays -> APTOS ids (one-to-one)\n```\n" + json.dumps(res, indent=1, default=str) + "\n```")


# ----------------------------------------------------------------------------- 5. near-duplicates
PHASH_MAX = 10      # candidate pairs
CORR_MIN = 0.98     # verified duplicate if 64x64 gray corr >= this (v1 recon: clean gap 0.99-0.995; 0.95 chained 487 imgs)


def phash64(u8):
    out = np.zeros(len(u8), dtype=np.uint64)
    for i, x in enumerate(u8):
        g = cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        d = cv2.dct(g)[:8, :8].ravel()
        bits = d > np.median(d[1:])
        out[i] = np.packbits(bits).view(">u8")[0]
    return out


def popcount(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return np.unpackbits(x.view(np.uint8).reshape(len(x), 8), axis=1).sum(1)


@section("near_duplicates")
def near_duplicates(df, u8):
    h = phash64(u8)
    g64 = np.stack([cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (64, 64), interpolation=cv2.INTER_AREA)
                    for x in u8]).reshape(len(u8), -1).astype(np.float32)
    g64 -= g64.mean(1, keepdims=True)
    g64 /= np.linalg.norm(g64, axis=1, keepdims=True) + 1e-6
    pairs, nn = [], np.full(len(h), 64)
    for i in range(len(h) - 1):
        dist = popcount(h[i] ^ h[i + 1:])
        nn[i] = min(nn[i], dist.min())
        nn[i + 1:] = np.minimum(nn[i + 1:], dist)
        for k in np.where(dist <= PHASH_MAX)[0]:
            j = i + 1 + k
            pairs.append((i, j, int(dist[k]), float(g64[i] @ g64[j])))
    p = pd.DataFrame(pairs, columns=["i", "j", "hamming", "corr"])
    p["id_i"], p["id_j"] = df.id_code.values[p.i], df.id_code.values[p.j]
    p["grade_i"], p["grade_j"] = df.diagnosis.values[p.i], df.diagnosis.values[p.j]
    p.sort_values(["corr"], ascending=False).head(20000).to_csv(f"{OUT}/dup_pairs.csv", index=False)

    parent = list(range(len(h)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ver = p[p["corr"] >= CORR_MIN]
    for i, j in zip(ver.i, ver.j):
        parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(len(h))])
    _, group = np.unique(roots, return_inverse=True)
    g = pd.DataFrame({"id_code": df.id_code, "diagnosis": df.diagnosis, "dup_group": group})
    g["group_size"] = g.groupby("dup_group").id_code.transform("size")
    g.to_csv(f"{OUT}/dup_groups.csv", index=False)
    multi = g[g.group_size > 1]
    conflicts = multi.groupby("dup_group").diagnosis.nunique()
    res = {
        "phash_nearest_neighbor_hist(0..12)": np.bincount(np.minimum(nn, 12), minlength=13).tolist(),
        "candidate_pairs_by_hamming": p.hamming.value_counts().sort_index().to_dict(),
        "verified_pairs(corr>=%.2f)" % CORR_MIN: int(len(ver)),
        "corr_percentiles_of_candidates_10_50_90": (np.percentile(p["corr"], [10, 50, 90]).round(3).tolist()
                                                     if len(p) else []),
        "groups_with_size>1": int(multi.dup_group.nunique()),
        "images_in_groups": int(len(multi)),
        "groups_with_label_conflict": int((conflicts > 1).sum()),
        "max_group_size": int(g.group_size.max()),
    }
    R["near_duplicates"] = res
    MD.append("## Near-duplicates (pHash + correlation)\n```\n" + json.dumps(res, indent=1) + "\n```")

    try:  # visual check, stays on Kaggle
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = ver.sort_values("corr").head(6)          # weakest verified pairs = most informative
        if len(top):
            fig, ax = plt.subplots(len(top), 2, figsize=(5, 2.6 * len(top)))
            ax = np.atleast_2d(ax)
            for r, (_, row) in enumerate(top.iterrows()):
                for c, k in enumerate((row.i, row.j)):
                    ax[r, c].imshow(u8[int(k)]); ax[r, c].axis("off")
                    ax[r, c].set_title(f"{df.id_code.values[int(k)]} g={df.diagnosis.values[int(k)]} "
                                       f"h={row.hamming} c={row['corr']:.3f}", fontsize=7)
            plt.tight_layout(); plt.savefig(f"{OUT}/dup_examples.png", dpi=80); plt.close()
    except Exception as e:
        print("plot failed:", e)


# ----------------------------------------------------------------------------- 6. EyePACS mirror
EYE_RE = re.compile(r"^(\d+)_(left|right)\.(jpe?g|png|tiff?)$", re.I)


@section("eyepacs")
def eyepacs():
    root = find_dir(lambda dp, dns, fns: "diabetic-retinopathy-2015-data-colored-resized" in dp)
    files, csvs = [], []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            (csvs if fn.lower().endswith(".csv") else files).append(os.path.join(dp, fn))
    parsed = [EYE_RE.match(os.path.basename(f)) for f in files]
    by_dir = pd.Series([os.path.relpath(os.path.dirname(f), root) for f in files]).value_counts().head(20).to_dict()
    rng = np.random.default_rng(0)
    shapes = [list(cv2.imread(files[i]).shape) for i in rng.choice(len(files), min(30, len(files)), replace=False)]
    patients = {m.group(1) for m in parsed if m}
    res = {"root": root, "n_files": len(files), "parse_rate": float(np.mean([m is not None for m in parsed])),
           "n_patients": len(patients), "files_per_dir": by_dir, "csv_files": csvs[:10],
           "sample_shapes": shapes[:10], "unique_shapes": len({tuple(s) for s in shapes}),
           "example_names": [os.path.basename(f) for f in files[:5]]}
    for c in csvs[:3]:
        res[f"csv:{os.path.basename(c)}"] = {"columns": list(pd.read_csv(c, nrows=3).columns),
                                             "rows": int(sum(1 for _ in open(c)) - 1)}
    R["eyepacs"] = res
    MD.append("## EyePACS mirror\n```\n" + json.dumps(res, indent=1, default=str) + "\n```")
    return {os.path.splitext(os.path.basename(f))[0]: f for f in files}


# ----------------------------------------------------------------------------- 7. IDRiD
@section("idrid")
def idrid():
    root = find_dir(lambda dp, dns, fns: os.path.basename(dp.rstrip("/" + os.sep)) == "indian-diabetic-retinopathy-image-dataset")
    tree = {}
    for dp, dns, fns in os.walk(root):
        rel = os.path.relpath(dp, root)
        if rel.count(os.sep) <= 4:
            exts = pd.Series([os.path.splitext(f)[1].lower() for f in fns]).value_counts().to_dict()
            tree[rel] = {"n_files": len(fns), "exts": exts, "examples": sorted(fns)[:3]}
    csvs = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".csv"):
                c = pd.read_csv(os.path.join(dp, fn))
                c = c.loc[:, ~c.columns.astype(str).str.startswith("Unnamed")]
                csvs[os.path.relpath(os.path.join(dp, fn), root)] = {
                    "rows": len(c), "columns": list(c.columns),
                    "value_counts_col2": c.iloc[:, 1].value_counts().sort_index().to_dict() if c.shape[1] > 1 else {}}
    R["idrid"] = {"root": root, "tree": tree, "csvs": csvs}
    MD.append("## IDRiD\n```\n" + json.dumps(R["idrid"], indent=1, default=str)[:6000] + "\n```")


# ----------------------------------------------------------------------------- 8. EyeQ labels -> mirror
@section("eyeq")
def eyeq(mirror_index):
    import urllib.request
    base = "https://raw.githubusercontent.com/HzFu/EyeQ/master/data/"
    frames = []
    for name in ("Label_EyeQ_train.csv", "Label_EyeQ_test.csv"):
        path = f"{OUT}/_{name}"
        urllib.request.urlretrieve(base + name, path)
        f = pd.read_csv(path)
        f = f.loc[:, ~f.columns.astype(str).str.startswith("Unnamed")]
        f["eyeq_split"] = "train" if "train" in name else "test"
        frames.append(f)
        os.remove(path)
    q = pd.concat(frames, ignore_index=True)
    img_col = next(c for c in q.columns if "image" in c.lower())
    q["stem"] = q[img_col].astype(str).str.replace(r"\.[A-Za-z]+$", "", regex=True)
    q["mirror_path"] = q.stem.map(mirror_index or {})
    q["mirror_class_dir"] = q.mirror_path.map(lambda p: os.path.basename(os.path.dirname(p)) if isinstance(p, str) else None)
    q.drop(columns=["stem"]).to_csv(f"{OUT}/eyeq_mapping.csv", index=False)
    qual_col = next((c for c in q.columns if "quality" in c.lower()), None)
    res = {"rows": len(q), "columns": list(q.columns),
           "quality_counts": q[qual_col].value_counts().sort_index().to_dict() if qual_col else None,
           "matched_to_mirror": int(q.mirror_path.notna().sum()),
           "match_rate": float(q.mirror_path.notna().mean())}
    dr_col = next((c for c in q.columns if c.lower().startswith("dr")), None)
    if dr_col:
        res["dr_grade_vs_mirror_class"] = pd.crosstab(q[dr_col], q.mirror_class_dir).to_dict()
    res["verdict_for_T04"] = ("OK - EyeQ labels map onto the mirror by filename"
                              if res["match_rate"] > 0.95 else
                              "PARTIAL - check missing files / use another mirror")
    R["eyeq"] = res
    MD.append("## EyeQ -> EyePACS mirror\n```\n" + json.dumps(res, indent=1, default=str) + "\n```")


# ----------------------------------------------------------------------------- run
if __name__ == "__main__":
    env()
    arr_dir = arrays()
    raw = aptos_raw()
    if raw is not None:
        df, u8, paths = raw
        near_duplicates(df, u8)
        if arr_dir:
            best = legacy_variant_search(arr_dir, df, paths, u8)
            if best is not None:
                provenance(arr_dir, df, best[1])
        del u8
    idx = eyepacs()
    idrid()
    eyeq(idx)
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    print("\n".join(MD))
