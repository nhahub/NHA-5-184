"""
E pilot: B at 384px vs B at 224px on the same folds (local, CPU, OOF only).

    python src/evaluation/compare_384.py --runs kaggle/runs --folds 0 1

Paired on the same images (the 384 build reuses the 224 split, checked by meta sha256):
  * main  : 384 seed 42 vs 224 seed 42 (same seed, same folds)
  * noise : 224 seed 7 / 1234 vs 224 seed 42 -> how big a difference a seed change alone produces
Metrics: QWK (argmax), AUC and sensitivity at fixed specificity for >=2 and >=3 (B score = P(grade>=k)).
Sensitivities use the threshold fitted on the other pilot fold(s) (cross-fit); the bootstrap uses
the ROC read-off (threshold refit per resample) so both sides are at exactly the same specificity.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import metrics as M  # noqa: E402
from evaluation.operating_points import crossfit_sens_at_spec, oracle_sens_at_spec  # noqa: E402

COLS = ["qwk", "auc2", "sens2@0.95", "sens2@0.9", "auc3", "sens3@0.95", "sens3@0.9"]


def load(runs, seed, img, folds, suffix=""):
    parts = []
    for f in folds:
        ps = glob.glob(os.path.join(runs, "**", f"B_f{f}_s{seed}_{img}{suffix}", "oof_preds.npz"), recursive=True)
        assert len(ps) == 1, (seed, img, f, ps)
        with np.load(ps[0], allow_pickle=False) as z:
            parts.append((f, z["id_code"], z["y_true"], z["probs5"]))
    ids = np.concatenate([p[1] for p in parts])
    o = np.argsort(ids)
    return {"id": ids[o], "y": np.concatenate([p[2] for p in parts])[o],
            "fold": np.concatenate([np.full(len(p[1]), p[0]) for p in parts])[o],
            "p": np.concatenate([p[3] for p in parts])[o].astype(np.float64)}


def full(y, p, fold):
    out = {"qwk": M.qwk(y, p.argmax(1))}
    for k in (2, 3):
        yb, s = y >= k, p[:, k:].sum(1)
        out[f"auc{k}"] = M.binary_auc(yb.astype(int), s)
        for sp in (0.95, 0.9):
            out[f"sens{k}@{sp}"] = crossfit_sens_at_spec(yb, s, fold, sp)[0]
    return out


def light(y, p):
    v = [M.qwk(y, p.argmax(1))]
    for k in (2, 3):
        yb, s = y >= k, p[:, k:].sum(1)
        v.append(M.binary_auc(yb.astype(int), s))
        v += [oracle_sens_at_spec(yb, s, sp)[0] for sp in (0.95, 0.9)]
    return np.array(v)


def boot_diff(y, p_new, p_ref, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        d.append(light(y[ix], p_new[ix]) - light(y[ix], p_ref[ix]))
    d = np.array(d)
    return [(d[:, j].mean(), np.percentile(d[:, j], 2.5), np.percentile(d[:, j], 97.5)) for j in range(d.shape[1])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="kaggle/runs")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)

    ref = load(a.runs, 42, 224, a.folds)
    new = load(a.runs, 42, 384, a.folds)
    others = {f"224 seed {s}": load(a.runs, s, 224, a.folds) for s in (7, 1234)}
    for d in [new, *others.values()]:
        assert np.array_equal(d["id"], ref["id"]) and np.array_equal(d["y"], ref["y"]) and np.array_equal(d["fold"], ref["fold"])
    y, fold = ref["y"], ref["fold"]

    print(f"### E pilot: B 384px vs 224px (folds {a.folds}, n={len(y)}, grades {np.bincount(y, minlength=5).tolist()})\n")
    print("| model | QWK | AUC ≥2 | sens ≥2 @ .95 | sens ≥2 @ .90 | AUC ≥3 | sens ≥3 @ .95 | sens ≥3 @ .90 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, d in [("224 seed 42", ref), *others.items(), ("**384 seed 42**", new)]:
        m = full(y, d["p"], fold)
        print(f"| {name} | " + " | ".join(f"{m[c]:.3f}" for c in COLS) + " |")

    print("\nPaired bootstrap differences vs 224 seed 42 (95% CI):\n")
    print("| comparison | Δ QWK | Δ AUC ≥2 | Δ sens ≥2 @ .95 | Δ sens ≥2 @ .90 | Δ AUC ≥3 | Δ sens ≥3 @ .95 | Δ sens ≥3 @ .90 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, d in [("**384 s42 − 224 s42**", new), *[(f"{k} − 224 s42 (seed noise)", v) for k, v in others.items()]]:
        r = boot_diff(y, d["p"], ref["p"], a.n_boot)
        print(f"| {name} | " + " | ".join(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]" for m, lo, hi in r) + " |")

    print("\nPer-fold QWK (fold-val):")
    for f in a.folds:
        m = fold == f
        row = {n: M.qwk(y[m], d["p"][m].argmax(1)) for n, d in [("224 s42", ref), *others.items(), ("384 s42", new)]}
        print(f"  fold {f}: " + ", ".join(f"{k} {v:.3f}" for k, v in row.items()))


if __name__ == "__main__":
    main()
