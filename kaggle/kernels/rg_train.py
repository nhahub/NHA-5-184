"""
rg-train (Kaggle, GPU T4). Generic trainer: the config is injected by
`python kaggle/make_kernel.py build rg-train --suffix <name> --config '<json object or list>'`.
Inputs: rg-data-v2 output (kernel source) + retinograde-code dataset.
Outputs: /kaggle/working/runs/<run_id>/{run.json, oof_preds.npz, *.keras}
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
import train  # noqa: E402

if __name__ == "__main__":
    configs = CONFIG if isinstance(CONFIG, list) else [CONFIG]   # a list = several runs in one session
    records = [r for cfg in configs for r in train.main(cfg)]
    for r in records:
        m = r["metrics"]
        if m is None:
            print(f"{r['run_id']}: full-pool model | {r['duration_s'] / 60:.1f} min")
            continue
        print(f"{r['run_id']}: QWK {m['qwk']:.4f} | ref sens/spec {m['referable_sens']:.3f}/"
              f"{m['referable_spec']:.3f} | {r['duration_s'] / 60:.1f} min")
