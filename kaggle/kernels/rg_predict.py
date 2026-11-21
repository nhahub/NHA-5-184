"""
rg-predict (Kaggle, GPU T4). Re-scores saved fold models (TTA on OOF, or the one-shot test run).
Config injected by `python kaggle/make_kernel.py build rg-predict --suffix <name> --config '<json>'`.
Inputs: retinograde-code + rg-data-v2 output + the rg-train-* kernels that hold the .keras files.
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
import predict  # noqa: E402

if __name__ == "__main__":
    for r in predict.main(CONFIG):
        print(r["run_id"], r["duration_s"], "s")
