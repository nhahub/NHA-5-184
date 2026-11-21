"""
rg-eyepacs-data (Kaggle, CPU only, ~1 h). EyePACS images for MIXED training (APTOS + EyePACS).

Patient-level split (identical rule to rg-eyepacs-dev, fixed forever):
    u = sha256("eyepacs-split-v1:<patient>")  -> dev (<0.10) / test (0.10-0.20) / rest (>=0.20)
Within REST (only these may be used for training or model selection):
    u2 = sha256("eyepacs-rest-v1:<patient>") < 0.06  -> "rval"   (epoch selection slice; all its images)
    otherwise                                          -> "rtrain" (training pool)
    rtrain keeps every DR image (grade >= 1) and the grade-0 images whose
    sha256("eyepacs-g0-v1:<image>") < 0.30 (a fixed 30% sample - keeps disk and CPU time down)
TEST patients are never opened. DEV patients are preprocessed only to be SCORED (never used to
select anything) so the pilot can be compared with the APTOS-only models on the same images.

Images: hi-res mirror (tanlikesmath, ~1024px, aspect kept) -> frozen preprocess_rgb at IMG (384).
Outputs (private kernel output, consumed via kernel_sources):
    eyepacs_rest_u8.npy + eyepacs_rest_meta.csv   (stem, patient, grade, part, quality)
    eyepacs_dev_u8.npy  + eyepacs_dev_meta.csv    (stem, patient, grade, quality)
    eyepacs_data_summary.json (small)
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
RVAL_FRAC, G0_KEEP = 0.06, 0.30


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


def u(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16) / 16 ** 15


def split_of(patient: str) -> str:          # MUST match kaggle/kernels/rg_eyepacs_dev.py
    x = u(f"eyepacs-split-v1:{patient}")
    return "dev" if x < 0.10 else ("test" if x < 0.20 else "rest")


def _prep(path):
    try:
        return np.round(preprocess_rgb(load_rgb(path), IMG) * 255).astype(np.uint8)
    except Exception:
        return None


def to_memmap(paths, out_path):
    X = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(len(paths), IMG, IMG, 3))
    ok = np.ones(len(paths), bool)
    with Pool(os.cpu_count()) as p:
        for i, img in enumerate(p.imap(_prep, paths, chunksize=16)):
            if img is None:
                ok[i] = False
            else:
                X[i] = img
    X.flush()
    return ok


if __name__ == "__main__":
    t0 = time.time()
    lab = pd.read_csv(next(p for p in glob.glob(f"{INPUT}/**/trainLabels.csv", recursive=True)))
    lab["stem"] = lab.image.astype(str).str.replace(r"\.\w+$", "", regex=True)
    lab["patient"] = lab.stem.str.split("_").str[0]
    lab["grade"] = lab.level.astype(int)
    lab["split"] = lab.patient.map(split_of)

    # cross-check with the split file written by rg-eyepacs-dev (same rule -> must be identical)
    prev = D.find_file("eyepacs_split.csv", required=False)
    if prev:
        p = pd.read_csv(prev, dtype={"patient": str})
        chk = lab.drop_duplicates("patient")[["patient", "split"]].merge(p, on="patient", suffixes=("", "_prev"))
        assert (chk.split == chk.split_prev).all() and len(chk) == lab.patient.nunique(), "split rule changed"

    hires = {}
    for dp, _, fns in os.walk(INPUT):
        if "resized_train" in dp and "cropped" not in dp:
            for fn in fns:
                m = re.match(r"^(\d+_(left|right))\.(jpe?g|png)$", fn, re.I)
                if m:
                    hires.setdefault(m.group(1), os.path.join(dp, fn))
    lab = lab[lab.stem.isin(hires)].reset_index(drop=True)

    q = D.find_file("eyeq_mapping.csv", required=False)
    if q:
        eq = pd.read_csv(q)
        eq["stem"] = eq.image.astype(str).str.replace(r"\.\w+$", "", regex=True)
        lab = lab.merge(eq[["stem", "quality"]].drop_duplicates("stem"), on="stem", how="left")
    lab["quality"] = lab.get("quality", pd.Series(np.nan, index=lab.index)).fillna(-1).astype(int)

    rest = lab[lab.split == "rest"].copy()
    rest["part"] = np.where(rest.patient.map(lambda s: u(f"eyepacs-rest-v1:{s}") < RVAL_FRAC), "rval", "rtrain")
    keep = (rest.part == "rval") | (rest.grade >= 1) | rest.stem.map(lambda s: u(f"eyepacs-g0-v1:{s}") < G0_KEEP)
    rest = rest[keep].sort_values(["part", "stem"]).reset_index(drop=True)
    dev = lab[lab.split == "dev"].sort_values("stem").reset_index(drop=True)
    assert set(rest.patient).isdisjoint(set(dev.patient))
    assert not (lab[lab.split == "test"].stem.isin(set(rest.stem) | set(dev.stem))).any()

    summary = {"code_version": CODE_VERSION, "img": IMG, "rule": __doc__.split("Patient-level split")[1].split("Images:")[0]}
    for name, df in (("rest", rest), ("dev", dev)):
        ok = to_memmap([hires[s] for s in df.stem], f"{OUT}/eyepacs_{name}_u8.npy")
        df = df.assign(ok=ok)
        cols = ["stem", "patient", "grade", "quality", "ok"] + (["part"] if name == "rest" else [])
        df[cols].to_csv(f"{OUT}/eyepacs_{name}_meta.csv", index=False)
        summary[name] = {"n": len(df), "failures": int((~ok).sum()), "patients": int(df.patient.nunique()),
                         "grades": np.bincount(df.grade, minlength=5).tolist()}
        if name == "rest":
            summary[name]["by_part"] = {p_: np.bincount(g.grade, minlength=5).tolist() for p_, g in df.groupby("part")}
        print(name, summary[name], f"{time.time() - t0:.0f}s", flush=True)
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    with open(f"{OUT}/eyepacs_data_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=int)
    print(json.dumps(summary, indent=1, default=int))
