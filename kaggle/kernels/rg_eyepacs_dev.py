"""
rg-eyepacs-dev (Kaggle, CPU only, ~40 min). Headroom check item 4: prediction-only domain-gap check.

EyePACS 2015 train set (35,126 images, 17,563 patients) -> patient-level split, fixed forever:
    u = sha256("eyepacs-split-v1:<patient>") as a number in [0, 1)
    dev  : u < 0.10          (scored here)
    test : 0.10 <= u < 0.20  (listed by id only - images are NEVER opened; final evaluation, once)
    rest : unused (reserved, e.g. for future EyePACS pretraining)
Both eyes of a patient are always in the same split.

Scoring: the 5 fold models of B seed 42 (224px), and the 2 fold models of the 384px pilot, each
loaded as saved (src/model_io.py; verified to match GPU predictions on CPU). Frozen preprocess_rgb contract.
Two mirrors of the same images are scored at 224px to separate "population gap" from "mirror gap":
    hires : tanlikesmath/diabetic-retinopathy-resized  (resized_train, ~1024px wide, aspect kept)
    lowres: sovitrath/...-colored-resized              (already squeezed to 224x224 by the mirror)
EyeQ quality labels (Good/Usable/Reject) are attached where available (rg-recon eyeq_mapping.csv).

Outputs (small): eyepacs_split.csv (patient -> split; ids only), eyepacs_dev_preds.npz,
                 eyepacs_dev_summary.json
"""
import glob
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

CONFIG = {}  # <- injected by make_kernel.py

OUT, INPUT = "/kaggle/working", "/kaggle/input"
DEV_FRAC, TEST_FRAC = 0.10, 0.10


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


def split_of(patient: str) -> str:
    u = int(hashlib.sha256(f"eyepacs-split-v1:{patient}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return "dev" if u < DEV_FRAC else ("test" if u < DEV_FRAC + TEST_FRAC else "rest")


def index_images(root_pred):
    """{stem: path} for EyePACS-style files (<patient>_<left|right>.<ext>) under matching roots."""
    out = {}
    for dp, _, fns in os.walk(INPUT):
        if not root_pred(dp):
            continue
        for fn in fns:
            m = re.match(r"^(\d+_(left|right))\.(jpe?g|png|tiff?)$", fn, re.I)
            if m:
                out.setdefault(m.group(1), os.path.join(dp, fn))
    return out


def _prep(args):
    path, size = args
    try:
        return np.round(preprocess_rgb(load_rgb(path), size) * 255).astype(np.uint8)
    except Exception:
        return None


def prep_all(paths, size):
    with Pool(os.cpu_count()) as p:
        res = p.map(_prep, [(q, size) for q in paths], chunksize=8)
    ok = np.array([r is not None for r in res])
    X = np.stack([r if r is not None else np.zeros((size, size, 3), np.uint8) for r in res])
    return X, ok


def predict(model, X, bs=64):
    out = []
    for i in range(0, len(X), bs):
        out.append(np.asarray(model.predict_on_batch(X[i:i + bs].astype(np.float32)), np.float64))
    return np.concatenate(out)


if __name__ == "__main__":
    import tensorflow as tf
    from model_io import load_model

    t0 = time.time()
    summary = {"code_version": CODE_VERSION, "rule": "sha256('eyepacs-split-v1:<patient>') < 0.10 -> dev; < 0.20 -> test"}
    lab = pd.read_csv(next(p for p in glob.glob(f"{INPUT}/**/trainLabels.csv", recursive=True)))
    lab["stem"] = lab.image.astype(str).str.replace(r"\.\w+$", "", regex=True)
    lab["patient"] = lab.stem.str.split("_").str[0]
    lab["split"] = lab.patient.map(split_of)
    lab[["patient", "split"]].drop_duplicates().to_csv(f"{OUT}/eyepacs_split.csv", index=False)
    summary["patients_by_split"] = lab.drop_duplicates("patient").split.value_counts().to_dict()
    summary["images_by_split"] = lab.split.value_counts().to_dict()
    summary["grades_by_split"] = {s: np.bincount(g.level, minlength=5).tolist() for s, g in lab.groupby("split")}
    dev = lab[lab.split == "dev"].reset_index(drop=True)          # ONLY dev images are opened below

    hires = index_images(lambda dp: "resized_train" in dp and "cropped" not in dp)
    lowres = index_images(lambda dp: "colored_images" in dp)
    summary["mirror_files"] = {"hires": len(hires), "lowres": len(lowres)}
    if not hires or not lowres:
        tops = sorted({os.path.relpath(dp, INPUT) for dp, _, _ in os.walk(INPUT) if dp.count(os.sep) <= 5})
        raise SystemExit(f"mirror layout not recognised {summary['mirror_files']}; dirs: {tops[:80]}")
    dev = dev[dev.stem.isin(hires) & dev.stem.isin(lowres)].reset_index(drop=True)
    summary["dev_images_in_both_mirrors"] = len(dev)

    eyeq = D.find_file("eyeq_mapping.csv", required=False)
    if eyeq:
        q = pd.read_csv(eyeq)
        q["stem"] = q.image.astype(str).str.replace(r"\.\w+$", "", regex=True)
        dev = dev.merge(q[["stem", "quality"]].drop_duplicates("stem"), on="stem", how="left")
    dev["quality"] = dev.get("quality", pd.Series(np.nan, index=dev.index)).fillna(-1).astype(int)

    from PIL import Image
    summary["hires_sample_sizes"] = [Image.open(hires[s]).size for s in dev.stem[:5]]
    summary["lowres_sample_sizes"] = [Image.open(lowres[s]).size for s in dev.stem[:5]]

    arrays = {}
    for name, idx, size in (("hires224", hires, 224), ("lowres224", lowres, 224), ("hires384", hires, 384)):
        X, ok = prep_all([idx[s] for s in dev.stem], size)
        arrays[name] = (X, ok)
        summary[f"preprocess_failures_{name}"] = int((~ok).sum())
        print(name, X.shape, "failures", int((~ok).sum()), f"{time.time() - t0:.0f}s", flush=True)

    preds = {"stem": dev.stem.to_numpy().astype(str), "patient": dev.patient.to_numpy().astype(str),
             "y": dev.level.to_numpy().astype(np.int64), "quality": dev.quality.to_numpy().astype(np.int64)}
    jobs = [(f"B_f{f}_s42_224", "hires224") for f in range(5)] + \
           [(f"B_f{f}_s42_224", "lowres224") for f in range(5)] + \
           [(f"B_f{f}_s42_384", "hires384") for f in range(2)]
    for rid, arr in jobs:
        model = load_model(D.find_file(f"{rid}_model.keras"))
        X, ok = arrays[arr]
        P = predict(model, X)
        P[~ok] = np.nan
        preds[f"{rid}|{arr}"] = P
        print(rid, arr, f"{time.time() - t0:.0f}s", flush=True)
        del model
        tf.keras.backend.clear_session()
    np.savez_compressed(f"{OUT}/eyepacs_dev_preds.npz", **preds)
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    with open(f"{OUT}/eyepacs_dev_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=1, default=str))
