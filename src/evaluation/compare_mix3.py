"""
Mixed training vs APTOS-only, B 384, 3 seeds x folds 0-1 (project rule: >= 3 seeds for any claim).

    python src/evaluation/compare_mix3.py --runs kaggle/runs

Per seed, the fold-0 and fold-1 models are compared on
  * APTOS fold-val (1,142 images; each image scored by the model that did not train on it)
  * EyePACS dev patients (3,532 images; the two fold models' probabilities averaged)
Reported: per-seed values, mean +/- std over seeds, and the paired bootstrap of the seed-averaged
difference (EyePACS: patient-level).
Sources of EyePACS dev probabilities:
  mixed            : <run>/eyepacs_dev_preds.npz written by the training kernel
  APTOS-only s42   : rg-eyepacs-dev  (key "B_f{f}_s42_384|hires384")
  APTOS-only s7/1234: rg-score-dev-b384 (eyepacs_dev_scores.npz)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.compare_384 import load  # noqa: E402
from evaluation.eyepacs_gap import KEYS, point  # noqa: E402
from evaluation import metrics as M  # noqa: E402

SEEDS, FOLDS = (7, 42, 1234), (0, 1)
APTOS_KEYS = ["qwk", "auc2", "auc3"]


def _one(pattern):
    p = glob.glob(pattern, recursive=True)
    assert len(p) == 1, (pattern, p)
    return dict(np.load(p[0]))


def dev_probs(runs):
    ref = _one(os.path.join(runs, "**", "rg-eyepacs-dev", "eyepacs_dev_preds.npz"))
    sc = _one(os.path.join(runs, "**", "rg-score-dev-b384", "eyepacs_dev_scores.npz"))
    order = np.argsort(sc["stem"])
    stems, y, patient = sc["stem"][order], sc["y"][order], sc["patient"][order]
    pos_ref = {s: i for i, s in enumerate(ref["stem"])}
    ir = np.array([pos_ref[s] for s in stems])
    base, mix = {}, {}
    for s in SEEDS:
        if s == 42:
            base[s] = np.mean([ref[f"B_f{f}_s42_384|hires384"][ir] for f in FOLDS], 0)
        else:
            base[s] = np.mean([sc[f"B_f{f}_s{s}_384"][order] for f in FOLDS], 0)
        ms = []
        for f in FOLDS:
            m = _one(os.path.join(runs, "**", f"B_f{f}_s{s}_384_mix", "eyepacs_dev_preds.npz"))
            pos = {st: i for i, st in enumerate(m["stem"])}
            ms.append(m["probs5"][[pos[st] for st in stems]])
        mix[s] = np.mean(ms, 0)
    return y, patient, base, mix


def aptos_metrics(y, p):
    return {"qwk": M.qwk(y, p.argmax(1)), "auc2": M.binary_auc((y >= 2).astype(int), p[:, 2:].sum(1)),
            "auc3": M.binary_auc((y >= 3).astype(int), p[:, 3:].sum(1))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="kaggle/runs")
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(0)

    # APTOS fold-val
    A = {s: load(a.runs, s, 384, list(FOLDS)) for s in SEEDS}
    X = {s: load(a.runs, s, 384, list(FOLDS), "_mix") for s in SEEDS}
    y = A[42]["y"]
    print("### Mixed vs APTOS-only, B 384, 3 seeds × folds 0–1\n\n**APTOS fold-val** (n=%d)\n" % len(y))
    print("| seed | " + " | ".join(f"{k} only → mixed" for k in APTOS_KEYS) + " |\n|---|" + "---|" * len(APTOS_KEYS))
    per = {s: (aptos_metrics(y, A[s]["p"]), aptos_metrics(y, X[s]["p"])) for s in SEEDS}
    for s in SEEDS:
        print(f"| {s} | " + " | ".join(f"{per[s][0][k]:.3f} → {per[s][1][k]:.3f}" for k in APTOS_KEYS) + " |")
    boots = []
    for _ in range(a.n_boot):
        ix = rng.integers(0, len(y), len(y))
        boots.append([np.mean([aptos_metrics(y[ix], X[s]["p"][ix])[k] - aptos_metrics(y[ix], A[s]["p"][ix])[k]
                               for s in SEEDS]) for k in APTOS_KEYS])
    boots = np.array(boots)
    print("| Δ mean of seeds [CI] | " + " | ".join(
        f"{np.mean([per[s][1][k] - per[s][0][k] for s in SEEDS]):+.3f} "
        f"[{np.percentile(boots[:, j], 2.5):+.3f}, {np.percentile(boots[:, j], 97.5):+.3f}]"
        for j, k in enumerate(APTOS_KEYS)) + " |")

    # EyePACS dev
    yd, pat, base, mix = dev_probs(a.runs)
    print(f"\n**EyePACS dev patients** (n={len(yd)}, {len(np.unique(pat))} patients)\n")
    print("| seed | " + " | ".join(f"{k} only → mixed" for k in KEYS) + " |\n|---|" + "---|" * len(KEYS))
    pd_ = {s: (point(yd, base[s]), point(yd, mix[s])) for s in SEEDS}
    for s in SEEDS:
        print(f"| {s} | " + " | ".join(f"{pd_[s][0][k]:.3f} → {pd_[s][1][k]:.3f}" for k in KEYS) + " |")
    _, inv = np.unique(pat, return_inverse=True)
    rows = [np.where(inv == g)[0] for g in range(inv.max() + 1)]
    boots = []
    for _ in range(a.n_boot):
        ix = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        boots.append([np.mean([point(yd[ix], mix[s][ix])[k] - point(yd[ix], base[s][ix])[k] for s in SEEDS]) for k in KEYS])
    boots = np.array(boots)
    print("| Δ mean of seeds [CI] | " + " | ".join(
        f"{np.mean([pd_[s][1][k] - pd_[s][0][k] for s in SEEDS]):+.3f} "
        f"[{np.nanpercentile(boots[:, j], 2.5):+.3f}, {np.nanpercentile(boots[:, j], 97.5):+.3f}]"
        for j, k in enumerate(KEYS)) + " |")


if __name__ == "__main__":
    main()
