"""
Headroom item 3 (local, CPU): is B still improving when it gets more training data?

    python src/evaluation/learning_curve.py --runs kaggle/runs --folds 0 1

B 224px, seed 42, folds 0-1, trained on 25 / 50 / 75 / 100 % of each fold's training images
(nested, class-stratified; the validation fold is always complete, so all points are scored on the
same images). The 100 % point is the existing seed-42 run; seeds 7 and 1234 give the seed-noise band.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.compare_384 import COLS, boot_diff, full, load  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="kaggle/runs")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)

    pts = {f"{p}%": load(a.runs, 42, 224, a.folds, f"_p{p}") for p in (25, 50, 75)}
    pts["100%"] = load(a.runs, 42, 224, a.folds)
    noise = {f"100% seed {s}": load(a.runs, s, 224, a.folds) for s in (7, 1234)}
    ref = pts["100%"]
    for d in [*pts.values(), *noise.values()]:
        assert np.array_equal(d["id"], ref["id"]) and np.array_equal(d["y"], ref["y"])
    y, fold = ref["y"], ref["fold"]

    print(f"### Learning curve: B 224px seed 42, folds {a.folds} (n={len(y)} validation images)\n")
    print("| training data | QWK | AUC ≥2 | sens ≥2 @ .95 | sens ≥2 @ .90 | AUC ≥3 | sens ≥3 @ .95 | sens ≥3 @ .90 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, d in [*pts.items(), *noise.items()]:
        m = full(y, d["p"], fold)
        print(f"| {name} | " + " | ".join(f"{m[c]:.3f}" for c in COLS) + " |")

    print("\nPaired bootstrap steps (95% CI):\n")
    print("| step | Δ QWK | Δ AUC ≥2 | Δ sens ≥2 @ .95 | Δ sens ≥2 @ .90 | Δ AUC ≥3 | Δ sens ≥3 @ .95 | Δ sens ≥3 @ .90 |")
    print("|---|---|---|---|---|---|---|---|")
    for lo, hi in (("25%", "50%"), ("50%", "75%"), ("75%", "100%"), ("50%", "100%")):
        r = boot_diff(y, pts[hi]["p"], pts[lo]["p"], a.n_boot)
        print(f"| {lo} → {hi} | " + " | ".join(f"{m:+.3f} [{l:+.3f}, {h:+.3f}]" for m, l, h in r) + " |")
    seed_mean = np.mean([full(y, d["p"], fold)["qwk"] for d in [ref, *noise.values()]])
    print(f"\n100% QWK over 3 seeds: {seed_mean:.3f}; seed-42 vs seeds 7/1234: "
          + ", ".join(f"{full(y, d['p'], fold)['qwk'] - full(y, ref['p'], fold)['qwk']:+.3f}" for d in noise.values()))


if __name__ == "__main__":
    main()
