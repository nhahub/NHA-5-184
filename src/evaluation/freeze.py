"""
Freeze the final configuration BEFORE the one-shot test (local, CPU).

  python src/evaluation/freeze.py epochs                 # fine-tune epochs for the full-pool model
  python src/evaluation/freeze.py write --full B_full_s42_384_mix   # thresholds + docs/frozen_config.json

epochs: for every mixed CV run (B 384 _mix), find the epoch with the best selection score; the full-pool
        model fine-tunes for the median of those epochs (same 15-epoch learning-rate schedule).
write : referral thresholds are set on the FULL model's predictions for the EyePACS DEV patients
        (never used in training or epoch selection of that model), at a fixed target specificity:
            P(grade >= 2) threshold -> specificity 0.90 for "referable"
            P(grade >= 3) threshold -> specificity 0.90 for "sight-threatening"
        EyePACS dev is used (not APTOS OOF) because it is a screening population: real prevalence,
        mixed image quality, and no source shortcut. The same thresholds are reported on APTOS OOF.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.operating_points import thr_at_spec  # noqa: E402

TARGET_SPEC = {"referable": 0.90, "sight_threatening": 0.90}


def best_ft_epochs(runs):
    out = {}
    for p in glob.glob(os.path.join(runs, "**", "B_f*_s*_384_mix", "run.json"), recursive=True):
        r = json.load(open(p))
        log = r["history"]["model"]["epochs"]
        best = max(range(len(log)), key=lambda i: log[i]["val_score"])
        out[r["run_id"]] = log[best]["epoch"] + 1 if log[best]["phase"] == "finetune" else 0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["epochs", "write"])
    ap.add_argument("--runs", default="kaggle/runs")
    ap.add_argument("--full", default="B_full_s42_384_mix")
    ap.add_argument("--out", default="docs/frozen_config.json")
    a = ap.parse_args(argv)

    ep = best_ft_epochs(a.runs)
    med = int(max(1, round(float(np.median(list(ep.values()))))))
    if a.action == "epochs":
        print(json.dumps(ep, indent=1))
        print(f"median best fine-tune epoch over {len(ep)} mixed CV runs: {med}")
        return

    p = glob.glob(os.path.join(a.runs, "**", a.full, "eyepacs_dev_preds.npz"), recursive=True)
    assert len(p) == 1, f"full model dev predictions not found: {p}"
    z = np.load(p[0])
    y, P = z["y"], z["probs5"]
    thr = {}
    for name, k in (("referable", 2), ("sight_threatening", 3)):
        s = P[:, k:].sum(1)
        thr[name] = round(thr_at_spec(s[y < k], TARGET_SPEC[name]), 4)
    ensemble = sorted(os.path.basename(os.path.dirname(q)) for q in
                      glob.glob(os.path.join(a.runs, "**", "B_f*_s42_384_mix", "run.json"), recursive=True))
    assert len(ensemble) == 5, ensemble
    cfg = {
        "name": "B384-mix-v2",
        "frozen_on": "2026-11-21",
        "model": "EfficientNetB0 5-class softmax (candidate B), 384x384, frozen preprocess_rgb contract",
        "training_data": "APTOS v2 train pool (2,879) + 8,000 EyePACS 'rest' images (class-stratified)",
        "primary": a.full,
        "primary_ft_epochs": med,
        "ensemble": [e for e in ensemble],
        "grade_rule": "argmax of the 5 class probabilities",
        "thresholds": thr,
        "threshold_rule": f"set on {a.full} EyePACS-dev predictions at specificity {TARGET_SPEC}",
        "tta": False,
        "test_sets": ["APTOS held-out 733", "IDRiD grading 516", "EyePACS test patients (1,714 / 3,428)"],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
