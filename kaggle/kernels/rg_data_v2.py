"""
rg-data-v2 (Kaggle, CPU, ~0.5 h). Raw APTOS -> preprocess_rgb (frozen contract) -> uint8 memmap + meta,
with the v2 split design. Also preprocesses IDRiD grading images as an external test set.

Inputs : APTOS competition, rg-recon output (xval_map.csv, dup_groups.csv), IDRiD, retinograde-code.
Outputs (stay on Kaggle, consumed by rg-train via kernel_sources):
  aptos_v2_u8.npy, aptos_v2_meta.csv            (~0.55 GB)
  idrid_u8.npy, idrid_meta.csv                  (516 grading imgs, external test; never used for training)
  data_v2_summary.json                          (small -> downloaded)

Split design (docs/experiments.md):
  test    = the 733 old-val ids (only images v1 never trained on) -> touched once, Phase 3
  dropped = train images in a near-duplicate group that also has a test member (leak guard)
  train   = the rest, 5-fold StratifiedGroupKFold (groups = dup_group, seed 42)
"""
import glob
import hashlib
import json
import os
import re
import sys
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

CONFIG = {}  # <- injected by make_kernel.py

OUT, INPUT = "/kaggle/working", "/kaggle/input"
SEED, N_FOLDS = 42, 5
IMG = int(CONFIG.get("img", 224))
# The 384px build must reuse exactly the 224px split (same recon inputs + same seed -> same meta file).
META_SHA_224 = "a690105d5107f282e8c28c8f45e4a96a1fd202539d75642cf72aa557ec3a918d"


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
from retinograde_inference import load_rgb, preprocess_rgb  # noqa: E402
from data import aptos_v2 as D  # noqa: E402


def _prep(path):
    return np.round(preprocess_rgb(load_rgb(path), IMG) * 255).astype(np.uint8)


def preprocess_to_memmap(paths, out_path):
    X = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(len(paths), IMG, IMG, 3))
    with Pool(os.cpu_count()) as p:
        for i, img in enumerate(p.imap(_prep, paths, chunksize=8)):
            X[i] = img
    X.flush()
    return X


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def find_dir(pred):
    for dp, dns, fns in os.walk(INPUT):
        if pred(dp, dns, fns):
            return dp
    raise FileNotFoundError("directory not found")


def build_aptos(summary):
    from sklearn.model_selection import StratifiedGroupKFold

    d = find_dir(lambda dp, dns, fns: "train.csv" in fns and "train_images" in dns)
    df = pd.read_csv(f"{d}/train.csv").rename(columns={"diagnosis": "grade"})

    # recon v2: one-to-one assignment of legacy X_val rows to APTOS ids (legacy preprocessing reproduced)
    xval = pd.read_csv(D.find_file("xval_map.csv"))
    assert len(xval) == 733 and xval.id_code.is_unique, "recon X_val mapping incomplete"
    # recon v2: 83.5% pixel-exact; the rest differ by <=1.2/255 on average (a few border pixels), while a
    # wrong image would differ by tens of levels -> require every row to be a near-exact match
    exact = float((xval.max_abs_diff <= 1).mean())
    assert (xval.mean_abs_diff < 2).all(), "some X_val rows are not near-exact matches - mapping unreliable"
    summary["xval_mapping"] = {"exact_rate": exact,
                               "label_mismatch_rows(csv label used)": int((xval.grade_csv != xval.y_val).sum())}

    dup = pd.read_csv(D.find_file("dup_groups.csv"))[["id_code", "dup_group"]]
    df = df.merge(dup, on="id_code", how="left", validate="1:1")
    assert df.dup_group.notna().all()

    test_ids = set(xval.id_code)
    df["split"] = np.where(df.id_code.isin(test_ids), "test", "train")
    test_groups = set(df.dup_group[df.split == "test"])
    leak = (df.split == "train") & df.dup_group.isin(test_groups)
    df.loc[leak, "split"] = "dropped"

    df["fold"] = -1
    tr = df.index[df.split == "train"]
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, va) in enumerate(sgkf.split(tr, df.grade[tr], groups=df.dup_group[tr])):
        df.loc[tr[va], "fold"] = k
    D.check_splits(df)

    df[["id_code", "grade", "dup_group", "split", "fold"]].to_csv(f"{OUT}/{D.META_FILE}", index=False)
    if IMG != 224:      # fail fast (before the slow preprocessing) if the split would differ from 224px
        assert file_sha(f"{OUT}/{D.META_FILE}") == META_SHA_224, "meta differs from the 224px build - folds would not match"
    X = preprocess_to_memmap([f"{d}/train_images/{i}.png" for i in df.id_code], f"{OUT}/{D.IMAGES_FILE}")

    summary["aptos"] = {
        "n": len(df), "shape": list(X.shape),
        "split_counts": df.split.value_counts().to_dict(),
        "dropped_for_dup_leak": int(leak.sum()),
        "grade_by_split": {s: np.bincount(g.grade, minlength=5).tolist() for s, g in df.groupby("split")},
        "grade_by_fold": {int(f): np.bincount(g.grade, minlength=5).tolist()
                          for f, g in df[df.split == "train"].groupby("fold")},
        "meta_sha256": file_sha(f"{OUT}/{D.META_FILE}"),
        "images_sha256": file_sha(f"{OUT}/{D.IMAGES_FILE}"),
    }


def build_idrid(summary):
    """IDRiD disease-grading images (train 413 + test 103), external test only."""
    root = next(dp for dp, _, _ in os.walk(INPUT)
                if os.path.basename(dp.rstrip("/" + os.sep)) == "indian-diabetic-retinopathy-image-dataset")
    images = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if re.match(r"^IDRiD_\d+\.(jpe?g|png|tif)$", fn, re.I) and "segment" not in dp.lower():
                images.setdefault(os.path.splitext(fn)[0], []).append(os.path.join(dp, fn))
    rows = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.lower().endswith(".csv") or "grad" not in (dp + fn).lower():
                continue
            c = pd.read_csv(os.path.join(dp, fn))
            c = c.loc[:, ~c.columns.astype(str).str.startswith("Unnamed")]
            name_col = next(col for col in c.columns if "image" in col.lower())
            grade_col = next(col for col in c.columns if "retinopathy" in col.lower())
            dme_col = next((col for col in c.columns if "edema" in col.lower() or "dme" in col.lower()), None)
            split = "test" if "test" in (dp + fn).lower() else "train"
            for _, r in c.dropna(subset=[name_col]).iterrows():
                name = str(r[name_col]).strip()
                cands = [p for p in images.get(name, []) if split in p.lower()] or images.get(name, [])
                cands = sorted(cands, key=lambda p: "grading" not in p.lower())   # prefer B. Disease Grading copy
                if cands:
                    rows.append({"image": name, "grade": int(r[grade_col]), "idrid_split": split,
                                 "dme": int(r[dme_col]) if dme_col else -1, "path": cands[0]})
    meta = pd.DataFrame(rows).drop_duplicates(["image", "idrid_split"]).reset_index(drop=True)
    assert len(meta) > 0, "no IDRiD grading images matched - check recon.json idrid tree"
    preprocess_to_memmap(meta.path.tolist(), f"{OUT}/idrid_u8.npy")
    meta["path"] = meta.path.str.replace(INPUT, "", regex=False)
    meta.to_csv(f"{OUT}/idrid_meta.csv", index=False)
    summary["idrid"] = {"n": len(meta), "by_split": meta.idrid_split.value_counts().to_dict(),
                        "grades": {s: np.bincount(g.grade, minlength=5).tolist()
                                   for s, g in meta.groupby("idrid_split")}}


if __name__ == "__main__":
    summary = {"code_version": CODE_VERSION, "seed": SEED, "n_folds": N_FOLDS, "img": IMG}
    build_aptos(summary)
    try:
        build_idrid(summary)
    except Exception as e:                       # APTOS data must not be blocked by IDRiD layout issues
        summary["idrid_error"] = repr(e)
    with open(f"{OUT}/data_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=int)
    print(json.dumps(summary, indent=1, default=int))
