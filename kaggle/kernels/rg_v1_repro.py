"""
rg-v1-repro (Kaggle, GPU T4, ~0.3 GPU-h). Re-score the frozen v1 cascade on the old APTOS val (733).

Inputs : aptos2019-preprocessed-dataset (X_val/y_val), aptos-stage1-models
         (best_binary_dr_model.keras == stage1_v1, best_severity_model.keras), retinograde-code.
Outputs: val_probs_v1.npz  p1, p2, y_true, threshold, stage2_model   (teammates' T11/T15 input, no GPU needed)
         v1_repro.json     metrics via src/evaluation/metrics.py + model sha256 (compare with the team's earlier v1 results)
"""
import glob
import hashlib
import json
import os
import sys
import time
import zipfile

import numpy as np

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
    v = glob.glob(f"{INPUT}/**/CODE_VERSION.txt", recursive=True)
    return open(v[0]).read().strip() if v else "unknown"


CODE_VERSION = bootstrap_code()
import tensorflow as tf  # noqa: E402
from data.aptos_v2 import find_file  # noqa: E402
from evaluation import metrics as M  # noqa: E402

THR = 0.3
NATURAL_PRIOR = np.array([370, 999, 193, 295], dtype=np.float64)
NATURAL_PRIOR /= NATURAL_PRIOR.sum()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def predict(model, X, bs=64):
    out = []
    for i in range(0, len(X), bs):
        out.append(model.predict_on_batch(np.asarray(X[i:i + bs], dtype=np.float32) * 255.0))
    return np.concatenate(out)


def main():
    t0 = time.time()
    X = np.load(find_file("X_val.npy"), mmap_mode="r")
    y = np.load(find_file("y_val.npy")).astype(np.int64)
    s1_path = find_file("stage1_v1.keras", required=False) or find_file("best_binary_dr_model.keras")
    s2_path = find_file("best_severity_model.keras")
    print("stage1:", s1_path, "\nstage2:", s2_path, "\nGPU:", tf.config.list_physical_devices("GPU"))

    p1 = predict(tf.keras.models.load_model(s1_path), X).ravel().astype(np.float64)
    p2 = predict(tf.keras.models.load_model(s2_path), X).astype(np.float64)
    np.savez_compressed(f"{OUT}/val_probs_v1.npz", p1=p1, p2=p2, y_true=y, threshold=THR,
                        stage2_model=os.path.basename(s2_path))

    probs5 = M.cascade_probs5(p1, p2)
    table = []
    for thr in (0.5, 0.4, 0.3, 0.2):
        pred = M.cascade_predict(p1, p2, thr)
        m = M.point_metrics(y, pred)
        m.update(threshold=thr, dr_recall=float(np.mean(p1[y > 0] >= thr)),
                 healthy_flagged=float(np.mean(p1[y == 0] >= thr)))
        table.append(m)

    pred = M.cascade_predict(p1, p2, THR)
    report = M.full_report(y, pred, score_dr=p1, score_ref=probs5[:, 2:].sum(1), n_boot=1000)

    dr = y > 0
    s2_pred = p2[dr].argmax(1)
    stage2_alone = {"n": int(dr.sum()), "qwk": M.qwk(y[dr] - 1, s2_pred, n_classes=4),
                    "accuracy": M.accuracy(y[dr] - 1, s2_pred), "within1": M.within1(y[dr] - 1, s2_pred),
                    "per_class": M.per_class_pr(y[dr] - 1, s2_pred, n_classes=4)}
    prior_sweep = []
    for alpha in (0.0, 0.5, 1.0):
        pa = M.cascade_predict(p1, p2, THR, NATURAL_PRIOR, alpha)
        mm = M.point_metrics(y, pa)
        prior_sweep.append({"alpha": alpha, "qwk": mm["qwk"], "sight_threat_sens": mm["sight_threat_sens"]})
    missed = [{"index": int(i), "grade": int(y[i]), "p1": round(float(p1[i]), 4)}
              for i in np.where((y >= 2) & (p1 < THR))[0]]

    rec = {"candidate": "v1-cascade", "threshold": THR, "code_version": CODE_VERSION,
           "metrics": report, "threshold_table": table, "stage2_alone": stage2_alone,
           "prior_sweep": prior_sweep, "missed_grade2plus_at_thr": missed,
           "stage1_any_dr_auc": M.binary_auc(dr, p1),
           "models": {os.path.basename(p): sha256(p) for p in (s1_path, s2_path)},
           "env": {"tf": tf.__version__}, "duration_s": round(time.time() - t0, 1),
           "expected_from_team_v1_results": {"cascade_qwk": [0.831, 0.840], "accuracy": [0.764, 0.783],
                                          "referable_sens": 0.85, "referable_spec": 0.96,
                                          "sight_threat_sens": [0.60, 0.62], "stage2_qwk": [0.49, 0.52],
                                          "misses_at_0.3": "6, all grade 2"}}
    with open(f"{OUT}/v1_repro.json", "w") as f:
        json.dump(rec, f, indent=2, default=float)
    print(json.dumps({k: rec[k] for k in ("threshold_table", "stage2_alone", "prior_sweep",
                                          "missed_grade2plus_at_thr", "models")}, indent=1, default=float))
    print("cascade @0.3:", {k: round(report[k], 4) for k in ("qwk", "accuracy", "referable_sens",
                                                             "referable_spec", "sight_threat_sens",
                                                             "sight_threat_spec")})


if __name__ == "__main__":
    main()
