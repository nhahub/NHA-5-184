"""
"Any DR" (grade >= 1) detection: A's Stage-1 probability vs B's 1 - P(grade 0). Local, CPU, OOF only.

    python src/evaluation/any_dr.py --runs kaggle/runs

A exists for seed 42 only, so the primary comparison is paired on seed 42 (same images, same folds,
same seed). B's other seeds and B with TTA / 3-seed averaging are shown to judge whether B can close
any gap without retraining. Sensitivities use thresholds cross-fitted on the other 4 folds.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import metrics as M  # noqa: E402
from evaluation.operating_points import (SEEDS, crossfit_sens_at_spec, load_oof,  # noqa: E402
                                         oracle_sens_at_spec)

SPECS = (0.95, 0.90)


def metrics(yb, s, fold):
    out = {"auc": M.binary_auc(yb.astype(int), s)}
    for sp in SPECS:
        out[f"sens@{sp}"], out[f"spec@{sp}"] = crossfit_sens_at_spec(yb, s, fold, sp)
    return out


def light(yb, s):
    return np.array([M.binary_auc(yb.astype(int), s)] + [oracle_sens_at_spec(yb, s, sp)[0] for sp in SPECS])


def paired_boot(yb, s_new, s_ref, n_boot=2000, seed=0):
    """Mean and 95% CI of (new - ref) for AUC and sens@spec (threshold refit inside each resample).
    s_new / s_ref may be lists of score vectors (averaged metric, e.g. over seeds)."""
    s_new = s_new if isinstance(s_new, list) else [s_new]
    s_ref = s_ref if isinstance(s_ref, list) else [s_ref]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(yb), len(yb))
        d.append(np.mean([light(yb[ix], s[ix]) for s in s_new], 0) - np.mean([light(yb[ix], s[ix]) for s in s_ref], 0))
    d = np.array(d)
    return [(d[:, j].mean(), np.percentile(d[:, j], 2.5), np.percentile(d[:, j], 97.5)) for j in range(d.shape[1])]


def per_fold_auc_diff(yb, fold, s_new, s_ref):
    diffs = [M.binary_auc(yb[fold == f].astype(int), s_new[fold == f]) - M.binary_auc(yb[fold == f].astype(int), s_ref[fold == f])
             for f in np.unique(fold)]
    return np.array(diffs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="kaggle/runs")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)

    A = load_oof(a.runs, "A", 42)
    B = {s: load_oof(a.runs, "B", s) for s in SEEDS}
    try:
        Bt = {s: load_oof(a.runs, "B", s, "_tta") for s in SEEDS}
    except AssertionError:
        Bt = None
    for d in [*B.values(), *(Bt or {}).values()]:
        assert np.array_equal(d["id"], A["id"]) and np.array_equal(d["fold"], A["fold"]) and np.array_equal(d["y"], A["y"])
    y, fold = A["y"], A["fold"]
    yb = y >= 1

    b = lambda d: 1.0 - d["probs5"][:, 0]                 # noqa: E731  B's P(any DR)
    rows = {"A Stage 1, seed 42": A["p1"],
            "B 1-P(0), seed 42": b(B[42])}
    for s in SEEDS:
        if s != 42:
            rows[f"B 1-P(0), seed {s}"] = b(B[s])
    rows["B 1-P(0), 3-seed avg"] = np.mean([b(B[s]) for s in SEEDS], 0)
    if Bt:
        rows["B 1-P(0), TTA seed 42"] = b(Bt[42])
        rows["B 1-P(0), TTA + 3-seed avg"] = np.mean([b(Bt[s]) for s in SEEDS], 0)

    print(f"### Any DR (grade >= 1): A vs B (OOF, n={len(y)}, positives {int(yb.sum())}; "
          f"sens at fixed spec, thresholds cross-fitted)\n")
    print("| score | AUC | sens @ spec .95 (achieved spec) | sens @ spec .90 (achieved spec) |\n|---|---|---|---|")
    for name, s in rows.items():
        m = metrics(yb, s, fold)
        print(f"| {name} | {m['auc']:.4f} | {m['sens@0.95']:.3f} ({m['spec@0.95']:.3f}) | "
              f"{m['sens@0.9']:.3f} ({m['spec@0.9']:.3f}) |")

    print("\nPaired bootstrap, B − A (95% CI; same images and folds):\n")
    print("| comparison | Δ AUC | Δ sens @ .95 | Δ sens @ .90 | per-fold Δ AUC (B wins) |\n|---|---|---|---|---|")
    comps = {"B seed 42 − A seed 42 (same seed)": (rows["B 1-P(0), seed 42"], A["p1"]),
             "B mean of 3 seeds − A": ([b(B[s]) for s in SEEDS], A["p1"]),
             "B 3-seed avg − A": (rows["B 1-P(0), 3-seed avg"], A["p1"])}
    if Bt:
        comps["B TTA + 3-seed avg − A"] = (rows["B 1-P(0), TTA + 3-seed avg"], A["p1"])
    for name, (sn, sr) in comps.items():
        r = paired_boot(yb, sn, sr, a.n_boot)
        pf = per_fold_auc_diff(yb, fold, sn if not isinstance(sn, list) else np.mean(sn, 0), sr)
        cells = " | ".join(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]" for m, lo, hi in r)
        print(f"| {name} | {cells} | {pf.mean():+.4f} ({int((pf > 0).sum())}/{len(pf)}) |")


if __name__ == "__main__":
    main()
