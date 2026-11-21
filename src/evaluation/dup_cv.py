"""
Headroom item 1 (local, CPU): do near-duplicates that sit in DIFFERENT CV folds inflate the OOF numbers?

    python src/evaluation/dup_cv.py --pairs kaggle/runs/rg-headroom/dup_candidates.csv --cos 0.97 --corr 0.95

A candidate pair (from kaggle/kernels/rg_headroom.py) counts as a near-duplicate when
    corr64 >= --corr   (64x64 grey correlation, best over flips/rotations)   OR
    emb_cos >= --cos   (ImageNet embedding cosine, best over {original, h-flip})
Thresholds are chosen by looking at the pair gallery (docs/figures_private/rg-headroom/).

Reported:
  * number of cross-fold duplicate pairs / images, and how many carry different labels
  * OOF metrics of B and C (3 seeds) on all images vs. with every image of a cross-fold pair removed
    (removing the images is the conservative choice: if the model "remembered" the twin from
    training, both copies are suspect)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import metrics as M  # noqa: E402
from evaluation.operating_points import SEEDS, crossfit_sens_at_spec, load_oof  # noqa: E402


def dup_pairs(pairs: pd.DataFrame, cos: float, corr: float) -> pd.DataFrame:
    d = pairs[(pairs.corr64 >= corr) | (pairs.emb_cos >= cos)].copy()
    d["label_conflict"] = d.grade_a != d.grade_b
    d["grade_gap"] = (d.grade_a - d.grade_b).abs()
    return d


def metrics_for(cand, d, keep):
    y, fold = d["y"][keep], d["fold"][keep]
    if cand == "B":
        p = d["probs5"][keep]
        pred, s2, s3 = p.argmax(1), p[:, 2:].sum(1), p[:, 3:].sum(1)
    else:
        s = d["s"][keep]
        pred, s2, s3 = M.apply_thresholds(s), s, s
    return {"qwk": M.qwk(y, pred), "acc": float((pred == y).mean()),
            "auc2": M.binary_auc((y >= 2).astype(int), s2), "auc3": M.binary_auc((y >= 3).astype(int), s3),
            "sens2@0.95": crossfit_sens_at_spec(y >= 2, s2, fold, 0.95)[0],
            "sens3@0.95": crossfit_sens_at_spec(y >= 3, s3, fold, 0.95)[0]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="kaggle/runs")
    ap.add_argument("--pairs", default="kaggle/runs/rg-headroom/dup_candidates.csv")
    ap.add_argument("--cos", type=float, default=0.97)
    ap.add_argument("--corr", type=float, default=0.95)
    a = ap.parse_args(argv)

    pairs = pd.read_csv(a.pairs)
    d = dup_pairs(pairs, a.cos, a.corr)
    cross = d[d.fold_a != d.fold_b]
    within = d[d.fold_a == d.fold_b]
    bad = set(cross.id_a) | set(cross.id_b)
    print(f"### Near-duplicates (rule: corr64 >= {a.corr} or emb_cos >= {a.cos})\n")
    print(f"- candidate pairs scanned: {len(pairs)}; duplicate pairs: {len(d)} "
          f"(within one fold: {len(within)}, across folds: {len(cross)})")
    print(f"- cross-fold pairs with different labels: {int(cross.label_conflict.sum())} "
          f"(grade gap >= 2: {int((cross.grade_gap >= 2).sum())})")
    print(f"- images involved in cross-fold pairs: {len(bad)}")
    print(f"- all duplicate pairs with label conflict: {int(d.label_conflict.sum())} of {len(d)}\n")

    print("| model | images | QWK | acc | AUC ≥2 | AUC ≥3 | sens ≥2 @ .95 | sens ≥3 @ .95 |\n|---|---|---|---|---|---|---|---|")
    for cand in ("B", "C"):
        ds = [load_oof(a.runs, cand, s) for s in SEEDS]
        keep_all = np.ones(len(ds[0]["y"]), bool)
        keep_clean = ~np.isin(ds[0]["id"], list(bad))
        for name, keep in (("all", keep_all), ("without cross-fold dups", keep_clean)):
            ms = [metrics_for(cand, d_, keep) for d_ in ds]
            cells = " | ".join(f"{np.mean([m[k] for m in ms]):.3f} ± {np.std([m[k] for m in ms], ddof=1):.3f}"
                               for k in ("qwk", "acc", "auc2", "auc3", "sens2@0.95", "sens3@0.95"))
            print(f"| {cand} ({name}) | {int(keep.sum())} | {cells} |")


if __name__ == "__main__":
    main()
