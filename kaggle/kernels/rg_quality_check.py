"""
rg-quality-check (Kaggle, CPU only, ~10 min). The deployed quality gate on REAL photos, end to end.
No EyePACS test patient is opened (qtest = dev patients).

  1. raw dev photos -> QualityGate.predict   vs   the stored qtest predictions of the same model (training job)
  2. RetinoPipeline (gate + frozen grading model) on 24 raw photos: the combined output and its keys
  3. time per image of the gate alone (CPU)
Output: quality_check_summary.json
"""
import glob
import json
import os
import re
import sys
import time
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
    v = glob.glob(f"{INPUT}/**/CODE_VERSION.txt", recursive=True)
    return open(v[0]).read().strip() if v else "unknown"


CODE_VERSION = bootstrap_code()
import retinograde_inference as R  # noqa: E402
from data import aptos_v2 as D  # noqa: E402
from pipeline import RetinoPipeline  # noqa: E402
from quality import quality as Q  # noqa: E402

if __name__ == "__main__":
    s = {"code_version": CODE_VERSION}
    gate = Q.QualityGate(D.find_file("quality_b0_224_s42.keras"))
    z = np.load(D.find_file("qtest_preds.npz"))          # seed 42's own folder comes first in the walk? pin it below
    runs = glob.glob(f"{INPUT}/**/quality_s42/qtest_preds.npz", recursive=True)
    z = np.load(runs[0])
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(z["stem"]), 300, replace=False))
    wanted, paths = set(z["stem"][pick]), {}
    for dp, _, fns in os.walk(INPUT):
        if "resized_train" in dp and "cropped" not in dp:
            for fn in fns:
                m = re.match(r"^(\d+_(left|right))\.(jpe?g|png)$", fn, re.I)
                if m and m.group(1) in wanted:
                    paths[m.group(1)] = os.path.join(dp, fn)
    diffs, same = [], 0
    for i in pick:
        st = str(z["stem"][i])
        p = np.array([gate.predict(paths[st])["probabilities"][n] for n in Q.QUALITY_NAMES])
        diffs.append(float(np.abs(p - z["probs3"][i]).max()))
        same += int(p.argmax() == z["probs3"][i].argmax())
    s["gate_vs_stored"] = {"n": len(diffs), "max_abs_prob_diff": max(diffs), "median": float(np.median(diffs)), "same_label": same}
    print(s["gate_vs_stored"], flush=True)

    m = RetinoPipeline(D.find_file(os.path.basename(R.DEFAULT_MODEL_PATH)), D.find_file("quality_b0_224_s42.keras"))
    rows = []
    for st in list(paths)[:24]:
        r = m.predict(paths[st])
        rows.append({"quality": r["quality"]["label"], "grade": r["grade"], "referable": r["referable"]})
    s["pipeline_24"] = {"keys": sorted(m.predict(paths[list(paths)[0]]).keys()),
                        "labels": {k: sum(x["quality"] == k for x in rows) for k in Q.QUALITY_NAMES}}

    t = []
    for k, i in enumerate(pick[:33]):
        img = R.load_rgb(paths[str(z["stem"][i])])
        a = time.perf_counter()
        gate.probabilities(Q.prepare_quality(img)[None])
        if k >= 3:
            t.append(time.perf_counter() - a)
    s["gate_latency_ms"] = {"median": float(np.median(t) * 1000), "p90": float(np.percentile(t, 90) * 1000)}
    print(s)
    with open(f"{OUT}/quality_check_summary.json", "w") as f:
        json.dump(s, f, indent=1)
