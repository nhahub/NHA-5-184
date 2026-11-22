"""
rg-final-test (Kaggle, GPU T4, ~20 min). THE ONE-SHOT TEST EVALUATION (Phase 3). Run once, after the
configuration is frozen in docs/experiments.md. Nothing here is used to choose anything.

Test sets:
  aptos   : the 733 held-out APTOS images (split == "test" in aptos_v2_meta.csv)
  idrid   : IDRiD disease-grading images, all 516 (official train 413 + test 103; never used before)
  eyepacs : EyePACS test patients = sha256("eyepacs-split-v1:<patient>") in [0.10, 0.20)
            (1,714 patients / 3,428 images) - opened here for the first time

All images go through the frozen preprocess_rgb contract at CONFIG["img"].
Every model listed in CONFIG["models"] (file <run_id>_model.keras from the attached kernels) scores
every test image; ensembles / thresholds are applied locally (src/evaluation/final_test.py).
Output: final_test_preds.npz, final_test_summary.json (small; labels + probabilities only).
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
IMG = int(CONFIG.get("img", 384))


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


def split_of(patient: str) -> str:          # identical rule to rg_eyepacs_dev.py / rg_eyepacs_data.py
    x = int(hashlib.sha256(f"eyepacs-split-v1:{patient}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return "dev" if x < 0.10 else ("test" if x < 0.20 else "rest")


def _prep(path):
    try:
        return np.round(preprocess_rgb(load_rgb(path), IMG) * 255).astype(np.uint8)
    except Exception:
        return None


def eyepacs_test():
    lab = pd.read_csv(next(p for p in glob.glob(f"{INPUT}/**/trainLabels.csv", recursive=True)))
    lab["stem"] = lab.image.astype(str).str.replace(r"\.\w+$", "", regex=True)
    lab["patient"] = lab.stem.str.split("_").str[0]
    lab["split"] = lab.patient.map(split_of)
    prev = D.find_file("eyepacs_split.csv", required=False)          # written by rg-eyepacs-dev
    if prev:
        p = pd.read_csv(prev, dtype={"patient": str})
        chk = lab.drop_duplicates("patient")[["patient", "split"]].merge(p, on="patient", suffixes=("", "_prev"))
        assert (chk.split == chk.split_prev).all(), "EyePACS split rule changed"
    test = lab[lab.split == "test"].sort_values("stem").reset_index(drop=True)
    hires, wanted = {}, set(test.stem)
    for dp, _, fns in os.walk(INPUT):
        if "resized_train" in dp and "cropped" not in dp:
            for fn in fns:
                m = re.match(r"^(\d+_(left|right))\.(jpe?g|png)$", fn, re.I)
                if m and m.group(1) in wanted:
                    hires[m.group(1)] = os.path.join(dp, fn)
    test = test[test.stem.isin(hires)].reset_index(drop=True)
    with Pool(os.cpu_count()) as pool:
        imgs = pool.map(_prep, [hires[s] for s in test.stem], chunksize=16)
    ok = np.array([i is not None for i in imgs])
    X = np.stack([i if i is not None else np.zeros((IMG, IMG, 3), np.uint8) for i in imgs])
    q = D.find_file("eyeq_mapping.csv", required=False)
    quality = np.full(len(test), -1)
    if q:
        eq = pd.read_csv(q)
        eq["stem"] = eq.image.astype(str).str.replace(r"\.\w+$", "", regex=True)
        quality = test.stem.map(dict(zip(eq.stem, eq.quality))).fillna(-1).astype(int).to_numpy()
    return X[ok], {"id": test.stem.to_numpy().astype(str)[ok], "patient": test.patient.to_numpy().astype(str)[ok],
                   "y": test.level.to_numpy().astype(np.int64)[ok], "quality": quality[ok]}, int((~ok).sum())


if __name__ == "__main__":
    import tensorflow as tf
    from model_io import load_model

    t0 = time.time()
    assert CONFIG.get("models"), "CONFIG['models'] (run ids) required"
    summary = {"code_version": CODE_VERSION, "img": IMG, "models": CONFIG["models"], "tta": bool(CONFIG.get("tta"))}

    X, meta = D.load()
    D.check_splits(meta)
    assert X.shape[1] == IMG
    te = D.test_indices(meta)
    assert len(te) == 733
    sets = {"aptos": (X, te, {"id": meta.id_code.to_numpy().astype(str)[te],
                              "y": meta.grade.to_numpy().astype(np.int64)[te]})}
    Xi = np.load(D.find_file("idrid_u8.npy"), mmap_mode="r")
    mi = pd.read_csv(D.find_file("idrid_meta.csv"))
    assert Xi.shape[1] == IMG and len(mi) == len(Xi)
    sets["idrid"] = (Xi, np.arange(len(Xi)), {"id": mi.image.to_numpy().astype(str), "y": mi.grade.to_numpy().astype(np.int64),
                                              "idrid_split": mi.idrid_split.to_numpy().astype(str)})
    Xe, info_e, fail_e = eyepacs_test()
    sets["eyepacs"] = (Xe, np.arange(len(Xe)), info_e)
    summary["n"] = {k: int(len(v[1])) for k, v in sets.items()}
    summary["eyepacs_preprocess_failures"] = fail_e
    print(summary, f"{time.time() - t0:.0f}s", flush=True)

    out = {}
    for name, (_, _, info) in sets.items():
        for k, v in info.items():
            out[f"{name}|{k}"] = v
    for rid in CONFIG["models"]:
        model = load_model(D.find_file(f"{rid}_model.keras"))
        for name, (Xs, idx, _) in sets.items():
            out[f"{name}|{rid}"] = D.predict_memmap(model, Xs, idx, tta=bool(CONFIG.get("tta")))
        print(rid, f"{time.time() - t0:.0f}s", flush=True)
        del model
        tf.keras.backend.clear_session()
    np.savez_compressed(f"{OUT}/final_test_preds.npz", **out)
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    with open(f"{OUT}/final_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=1))
