"""
rg-quality-data (Kaggle, CPU only, ~15 min). Arrays for the image-quality gate.

Images: EyePACS hi-res mirror, EyeQ-labelled names of DEV and REST patients only. The EyePACS held-out TEST
patients are excluded before any file is opened.
Split (patient level, fixed):  dev patients -> qtest;  rest patients -> qval if
    sha256("eyeq-split-v1:<patient>") < 0.15 else qtrain
Preparation: src/quality/quality.py prepare_quality (crop, pad, 224px, NO CLAHE), identical to the app.
Outputs: quality_u8.npy, quality_meta.csv (stem, patient, quality, dr_grade, split), quality_data_summary.json
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
QVAL_FRAC = 0.15


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
from quality.quality import QUALITY_IMG_SIZE, prepare_quality  # noqa: E402
from retinograde_inference import load_rgb  # noqa: E402


def u(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16) / 16 ** 15


def eyepacs_split(patient):                      # same rule as rg_eyepacs_dev / rg_eyepacs_data / rg_final_test
    x = u(f"eyepacs-split-v1:{patient}")
    return "dev" if x < 0.10 else ("test" if x < 0.20 else "rest")


def _prep(path):
    return prepare_quality(load_rgb(path))


if __name__ == "__main__":
    t0 = time.time()
    q = pd.read_csv(next(p for p in glob.glob(f"{INPUT}/**/eyeq_mapping.csv", recursive=True)))
    q = q[q.eyeq_split == "train"].copy()
    q["stem"] = q.image.str.replace(r"\.\w+$", "", regex=True)
    q["patient"] = q.stem.str.split("_").str[0]
    es = q.patient.map(eyepacs_split)
    n_test_excluded = int((es == "test").sum())
    q, es = q[es != "test"], es[es != "test"]
    q["split"] = np.where(es == "dev", "qtest",
                          np.where(q.patient.map(lambda p: u(f"eyeq-split-v1:{p}") < QVAL_FRAC), "qval", "qtrain"))
    q = q.sort_values(["split", "stem"]).reset_index(drop=True)

    wanted, files = set(q.stem), {}
    for dp, _, fns in os.walk(INPUT):
        if "resized_train" in dp and "cropped" not in dp:
            for fn in fns:
                m = re.match(r"^(\d+_(left|right))\.(jpe?g|png)$", fn, re.I)
                if m and m.group(1) in wanted:
                    files[m.group(1)] = os.path.join(dp, fn)
    assert len(files) == len(q), (len(files), len(q))

    S = QUALITY_IMG_SIZE
    X = np.lib.format.open_memmap(f"{OUT}/quality_u8.npy", mode="w+", dtype=np.uint8, shape=(len(q), S, S, 3))
    with Pool(os.cpu_count()) as p:
        for i, im in enumerate(p.imap(_prep, [files[s] for s in q.stem], chunksize=16)):
            X[i] = im
    X.flush()
    meta = q.rename(columns={"DR_grade": "dr_grade"})[["stem", "patient", "quality", "dr_grade", "split"]]
    meta.to_csv(f"{OUT}/quality_meta.csv", index=False)
    s = {"code_version": CODE_VERSION, "n": len(meta), "test_patient_images_excluded": n_test_excluded,
         "by_split": {k: {"n": int(len(g)), "patients": int(g.patient.nunique()),
                          "good_usable_reject": np.bincount(g.quality, minlength=3).tolist()} for k, g in meta.groupby("split")},
         "minutes": round((time.time() - t0) / 60, 1)}
    with open(f"{OUT}/quality_data_summary.json", "w") as f:
        json.dump(s, f, indent=1)
    print(json.dumps(s, indent=1))
