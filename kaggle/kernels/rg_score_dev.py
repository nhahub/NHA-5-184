"""
rg-score-dev (Kaggle, CPU only). Score already-trained models on the EyePACS DEV patients
(prediction only; dev is never used to select anything for these models).
Images: eyepacs_dev_u8.npy from rg-eyepacs-data (frozen preprocess_rgb at the models' size).
CONFIG: {"models": [run ids]} - files <run_id>_model.keras from the attached kernels.
Output: eyepacs_dev_scores.npz (stem, patient, y, and one probability array per model).
"""
import glob
import os
import sys
import zipfile

import numpy as np

CONFIG = {}  # <- injected by make_kernel.py

OUT, INPUT = "/kaggle/working", "/kaggle/input"


def bootstrap_code():
    for root, _, files in os.walk(INPUT):
        if "retinograde_inference.py" in files and os.path.basename(root) == "src":
            break
    else:
        z = glob.glob(f"{INPUT}/**/src.zip", recursive=True)[0]
        zipfile.ZipFile(z).extractall(f"{OUT}/_code")
        root = f"{OUT}/_code/src" if os.path.isdir(f"{OUT}/_code/src") else f"{OUT}/_code"
    sys.path[:0] = [root, os.path.join(root, "training")]


bootstrap_code()
from data import aptos_v2 as D  # noqa: E402
from model_io import load_model  # noqa: E402

if __name__ == "__main__":
    import tensorflow as tf
    Xd, md = D.load_eyepacs("dev")
    ok = np.where(md.ok.to_numpy().astype(bool))[0]
    out = {"stem": md.stem.to_numpy().astype(str)[ok], "patient": md.patient.to_numpy().astype(str)[ok],
           "y": md.grade.to_numpy().astype(np.int64)[ok]}
    for rid in CONFIG["models"]:
        m = load_model(D.find_file(f"{rid}_model.keras"))
        out[rid] = D.predict_memmap(m, Xd, ok)
        print(rid, "done", flush=True)
        del m
        tf.keras.backend.clear_session()
    np.savez_compressed(f"{OUT}/eyepacs_dev_scores.npz", **out)
