"""
rg-infer-check (Kaggle, CPU only, ~10 min). End-to-end check of src/retinograde_inference.py (v2)
with the REAL frozen model on REAL images. Reads the train pool and EyePACS dev only - no test split.

  1. preprocessing: raw APTOS train-pool files -> RetinoGrade.prepare  vs  the stored training arrays
     (aptos_v2_u8.npy at 384px)                                      -> must be bit-exact
  2. predictions:   raw EyePACS dev files -> RetinoGrade.predict_batch vs the probabilities the final
     model produced on Colab right after training (eyepacs_dev_preds.npz)
  3. latency:       one image at a time (file -> result), 1 / 2 / 4 threads, fresh process each
Output: infer_check_summary.json (small).
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd

CONFIG = {}  # <- injected by make_kernel.py

OUT, INPUT = "/kaggle/working", "/kaggle/input"
N_PRE, N_PRED, N_LAT, WARMUP = 120, 300, 30, 3


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
import retinograde_inference as R  # noqa: E402
from data import aptos_v2 as D  # noqa: E402


def aptos_train_files(n, seed=0):
    d = next(dp for dp, dns, fns in os.walk(INPUT) if "train.csv" in fns and "train_images" in dns)
    meta = pd.read_csv(D.find_file(D.META_FILE))
    rows = meta.index[meta.split == "train"].to_numpy()
    rows = np.sort(np.random.default_rng(seed).choice(rows, n, replace=False))
    return [(int(r), f"{d}/train_images/{meta.id_code[r]}.png") for r in rows]


def latency_worker(threads):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    t0 = time.perf_counter()
    m = R.RetinoGrade(D.find_file(os.path.basename(R.DEFAULT_MODEL_PATH)))
    load_s = time.perf_counter() - t0
    files = [p for _, p in aptos_train_files(N_LAT + WARMUP, seed=1)]
    pre, net = [], []
    for i, p in enumerate(files):
        a = time.perf_counter()
        x = m.prepare(p)
        b = time.perf_counter()
        R.v2_result(m.probabilities(x[None])[0])
        c = time.perf_counter()
        if i >= WARMUP:
            pre.append(b - a)
            net.append(c - b)
    tot = np.array(pre) + np.array(net)
    ms = lambda v, q: round(float(np.percentile(v, q)) * 1000, 1)  # noqa: E731
    print("LAT" + json.dumps({"threads": threads, "n": len(tot), "load_model_s": round(load_s, 1),
                              "preprocess_ms_median": ms(pre, 50), "model_ms_median": ms(net, 50),
                              "total_ms_median": ms(tot, 50), "total_ms_p90": ms(tot, 90), "cpus": os.cpu_count()}))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].isdigit():
        latency_worker(int(sys.argv[1]))
        sys.exit(0)

    summary = {"code_version": CODE_VERSION, "img_size": R.IMG_SIZE, "model_name": R.MODEL_NAME}
    import tensorflow as tf
    m = R.RetinoGrade(D.find_file(os.path.basename(R.DEFAULT_MODEL_PATH)))
    summary["tf"] = tf.__version__

    # 1. preprocessing parity
    X = np.load(D.find_file(D.IMAGES_FILE), mmap_mode="r")
    exact = 0
    worst = 0.0
    for row, path in aptos_train_files(N_PRE):
        d = np.abs(m.prepare(path) - np.asarray(X[row], np.float32))
        exact += int(d.max() == 0)
        worst = max(worst, float(d.max()))
    summary["preprocessing"] = {"n": N_PRE, "bit_exact": exact, "max_abs_diff_levels": worst}
    print(summary["preprocessing"], flush=True)

    # 2. predictions vs stored EyePACS dev predictions of the same model
    z = np.load(D.find_file("eyepacs_dev_preds.npz"))
    pick = np.sort(np.random.default_rng(0).choice(len(z["stem"]), N_PRED, replace=False))
    wanted, paths = set(z["stem"][pick]), {}
    for dp, _, fns in os.walk(INPUT):
        if "resized_train" in dp and "cropped" not in dp:
            for fn in fns:
                k = re.match(r"^(\d+_(left|right))\.(jpe?g|png)$", fn, re.I)
                if k and k.group(1) in wanted:
                    paths[k.group(1)] = os.path.join(dp, fn)
    stems = [s for s in z["stem"][pick] if s in paths]
    res = m.predict_batch([paths[s] for s in stems])
    P = np.array([[r["probabilities"][c] for c in R.CLASS_NAMES] for r in res])
    ref = np.array([z["probs5"][list(z["stem"]).index(s)] for s in stems], dtype=np.float64)
    d = np.abs(P - ref).max(1)
    summary["predictions_vs_stored"] = {
        "n": len(stems), "max_abs_prob_diff": float(d.max()), "median_abs_prob_diff": float(np.median(d)),
        "p99_abs_prob_diff": float(np.percentile(d, 99)), "same_grade": int((P.argmax(1) == ref.argmax(1)).sum()),
        "same_referable_flag": int(((P[:, 2:].sum(1) >= R.REFERABLE_THRESHOLD) == (ref[:, 2:].sum(1) >= R.REFERABLE_THRESHOLD)).sum()),
        "same_sight_threatening_flag": int(((P[:, 3:].sum(1) >= R.SIGHT_THREATENING_THRESHOLD)
                                            == (ref[:, 3:].sum(1) >= R.SIGHT_THREATENING_THRESHOLD)).sum())}
    print(summary["predictions_vs_stored"], flush=True)
    summary["output_keys"] = sorted(res[0])

    # 3. latency
    summary["latency"] = []
    for th in (1, 2, 4):
        p = subprocess.run([sys.executable, os.path.abspath(sys.argv[0]), str(th)], capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("LAT")]
        summary["latency"].append(json.loads(line[-1][3:]) if line else {"threads": th, "error": p.stderr[-800:]})
        print(summary["latency"][-1], flush=True)
    with open(f"{OUT}/infer_check_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=1))
