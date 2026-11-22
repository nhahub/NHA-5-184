"""
rg-quality-train (Kaggle, GPU T4, ~0.5 h). Trains the image-quality gate, 3 seeds (src/training/train_quality.py).
Inputs: retinograde-code + rg-quality-data. Outputs: runs/quality_s<seed>/{quality_run.json, qval_preds.npz,
qtest_preds.npz, quality_b0_224_s<seed>.keras}. Private.
"""
import glob
import os
import sys
import zipfile

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
import train_quality  # noqa: E402

if __name__ == "__main__":
    for r in train_quality.main(CONFIG):
        print(r["run_id"], r["best_qval_macro_f1"], r["duration_s"])
