"""
rg-latency (Kaggle, CPU only, ~10 min). Per-image CPU latency of the deployed path:
read PNG -> preprocess_rgb (frozen contract) -> model.predict, batch size 1, at 224 vs 384.

Each (img size, thread count, mode) runs in a fresh subprocess so TensorFlow's thread settings apply.
The model is loaded as saved (src/model_io.load_model). Version 3 of this kernel also timed a
"float32 copy"; it gave wrong outputs and was dropped (see src/model_io.py). Every run reports the
difference to the stored GPU out-of-fold predictions as a correctness check.
Inputs: retinograde-code, APTOS competition (33 fold-0 validation images), the saved fold-0 models of
rg-train-b-s42 (224) and rg-train-384-b-pilot (384). Output: latency.json (small).
"""
import glob
import json
import os
import subprocess
import sys
import time
import zipfile

CONFIG = {}  # <- injected by make_kernel.py

OUT, INPUT = "/kaggle/working", "/kaggle/input"
N_IMAGES, WARMUP = 30, 3


def code_root():
    for root, _, files in os.walk(INPUT):
        if "retinograde_inference.py" in files and os.path.basename(root) == "src":
            return root
    z = glob.glob(f"{INPUT}/**/src.zip", recursive=True)[0]
    zipfile.ZipFile(z).extractall(f"{OUT}/_code")
    return f"{OUT}/_code/src" if os.path.isdir(f"{OUT}/_code/src") else f"{OUT}/_code"


def worker(img: int, threads: int, mode: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    sys.path[:0] = [code_root()]
    from retinograde_inference import load_rgb, preprocess_rgb
    from data import aptos_v2 as D

    from model_io import load_model
    model = load_model(D.find_file(f"B_f0_s42_{img}_model.keras"))
    meta = pd.read_csv(D.find_file("aptos_v2_meta.csv"))
    # fold-0 validation images of the train pool (test split untouched): the fold-0 model never trained on
    # them and its GPU predictions are stored in oof_preds.npz -> ground truth for the CPU outputs
    ids = meta.id_code[(meta.split == "train") & (meta.fold == 0)].iloc[:N_IMAGES + WARMUP].tolist()
    oof = np.load(glob.glob(f"{INPUT}/**/B_f0_s42_{img}/oof_preds.npz", recursive=True)[0])
    gpu = dict(zip(oof["id_code"].astype(str), oof["probs5"]))
    tdir = os.path.dirname(D.find_file(f"{ids[0]}.png"))
    pre, inf, outs = [], [], []
    for i, id_ in enumerate(ids):
        t0 = time.perf_counter()
        x = preprocess_rgb(load_rgb(f"{tdir}/{id_}.png"), img) * 255.0
        t1 = time.perf_counter()
        outs.append(np.asarray(model(np.asarray(x, np.float32)[None], training=False), np.float64)[0])
        t2 = time.perf_counter()
        if i >= WARMUP:
            pre.append(t1 - t0)
            inf.append(t2 - t1)
    q = lambda v, p: round(float(np.percentile(v, p)) * 1000, 1)  # noqa: E731
    tot = np.array(pre) + np.array(inf)
    np.save(f"{OUT}/_probs_{img}_{mode}.npy", np.array(outs))
    ref = np.array([gpu[i] for i in ids])
    vs_gpu = {"max_abs_prob_diff_vs_gpu_oof": round(float(np.abs(np.array(outs) - ref).max()), 4),
              "argmax_agree_vs_gpu_oof": float((np.array(outs).argmax(1) == ref.argmax(1)).mean())}
    print(json.dumps({"img": img, "threads": threads, "mode": mode, "n": len(inf),
                      "preprocess_ms_median": q(pre, 50), "model_ms_median": q(inf, 50),
                      "total_ms_median": q(tot, 50), "total_ms_p90": q(tot, 90),
                      "params": int(model.count_params()), "cpus": os.cpu_count(), "tf": tf.__version__, **vs_gpu}))


if __name__ == "__main__":
    if len(sys.argv) == 4:
        worker(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
        sys.exit(0)
    results = []
    for img in (224, 384):
        for mode in ("saved",):
            for threads in (1, 2, 4):
                p = subprocess.run([sys.executable, __file__, str(img), str(threads), mode], capture_output=True, text=True)
                line = [l for l in p.stdout.splitlines() if l.startswith("{")]
                results.append(json.loads(line[-1]) if line else
                               {"img": img, "threads": threads, "mode": mode, "error": p.stderr[-2000:]})
                print(results[-1], flush=True)
    with open(f"{OUT}/latency.json", "w") as f:
        json.dump(results, f, indent=2)
