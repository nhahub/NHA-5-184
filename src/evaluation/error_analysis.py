"""
Headroom item 2 (local, CPU): where does model B go wrong?

    python src/evaluation/error_analysis.py --stats kaggle/runs/rg-headroom/image_stats.csv

Uses the OOF probabilities of B averaged over the 3 seeds (one prediction per image).
  * confusion matrix; adjacent (|error| = 1) vs far (|error| >= 2) errors
  * confident-wrong: wrong with averaged max probability >= 0.7 and all 3 seeds agreeing
  * error rate by image statistics (quartiles of raw brightness, contrast, sharpness, resolution)
    from kaggle/kernels/rg_headroom.py, with a bootstrap CI for the worst-vs-best quartile gap
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import metrics as M  # noqa: E402
from evaluation.operating_points import SEEDS, load_oof  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="kaggle/runs")
    ap.add_argument("--stats", default="kaggle/runs/rg-headroom/image_stats.csv")
    a = ap.parse_args(argv)

    ds = [load_oof(a.runs, "B", s) for s in SEEDS]
    y = ds[0]["y"]
    P = np.mean([d["probs5"] for d in ds], 0)
    pred, conf = P.argmax(1), P.max(1)
    agree = np.all([d["probs5"].argmax(1) == ds[0]["probs5"].argmax(1) for d in ds], 0)
    err = pred - y
    wrong = err != 0

    print(f"### B error analysis (OOF, 3-seed average, n={len(y)})\n")
    print(f"- QWK {M.qwk(y, pred):.3f}, accuracy {1 - wrong.mean():.3f}")
    print(f"- errors: {wrong.sum()} = adjacent (|d|=1) {(np.abs(err) == 1).sum()} "
          f"({(np.abs(err) == 1).sum() / wrong.sum():.0%}) + far (|d|>=2) {(np.abs(err) >= 2).sum()}")
    print(f"- over-calls (pred > true) {(err > 0).sum()}, under-calls {(err < 0).sum()}")
    cw = wrong & (conf >= 0.7) & agree
    print(f"- confident-wrong (p>=0.7, 3 seeds agree): {cw.sum()}  "
          f"(of these far: {(cw & (np.abs(err) >= 2)).sum()})")
    print(f"- missed referable (true>=2, pred<2): {((y >= 2) & (pred < 2)).sum()}; "
          f"missed sight-threatening (true>=3, pred<3): {((y >= 3) & (pred < 3)).sum()}\n")

    print("Confusion matrix (rows = true grade, cols = predicted):\n")
    C = M.confusion(y, pred, 5) if "confusion" in dir(M) else pd.crosstab(y, pred).to_numpy()
    print("| true \\ pred | 0 | 1 | 2 | 3 | 4 | recall |\n|---|---|---|---|---|---|---|")
    for g in range(5):
        print(f"| {g} | " + " | ".join(str(int(v)) for v in C[g]) + f" | {C[g, g] / C[g].sum():.2f} |")

    print("\nTop confusions:", ", ".join(f"{t}→{p}: {int(C[t, p])}" for t, p in
                                         sorted(((t, p) for t in range(5) for p in range(5) if t != p),
                                                key=lambda tp: -C[tp])[:5]))

    if os.path.exists(a.stats):
        st = pd.read_csv(a.stats).set_index("id_code").loc[ds[0]["id"]]
        st["raw_pixels"] = st.raw_w * st.raw_h
        print("\nError rate by image-statistic quartile (Q1 = lowest):\n")
        print("| statistic | Q1 | Q2 | Q3 | Q4 | Q4 − Q1 [95% CI] | grade mix Q1 → Q4 (% grade ≥2) |\n|---|---|---|---|---|---|---|")
        rng = np.random.default_rng(0)
        for col in ("raw_brightness", "raw_contrast", "raw_sharpness", "raw_pixels", "retina_frac"):
            q = pd.qcut(st[col].rank(method="first"), 4, labels=False).to_numpy()
            rates = [wrong[q == k].mean() for k in range(4)]
            boots = []
            for _ in range(1000):
                ix = rng.integers(0, len(y), len(y))
                w, qq = wrong[ix], q[ix]
                boots.append(w[qq == 3].mean() - w[qq == 0].mean())
            mix = [(y[q == k] >= 2).mean() for k in (0, 3)]
            print(f"| {col} | " + " | ".join(f"{r:.3f}" for r in rates) +
                  f" | {rates[3] - rates[0]:+.3f} [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]"
                  f" | {mix[0]:.0%} → {mix[1]:.0%} |")
        print("\n(The grade mix column matters: a quartile with more diseased eyes has more errors"
              " regardless of image quality, because grades 1-4 are harder than grade 0.)")

        # grade-standardised error rate: per quartile, average the per-grade error rates with the
        # OVERALL grade proportions as weights -> removes the grade-mix confounder
        w = np.bincount(y, minlength=5) / len(y)

        def std_rate(mask, wr):
            r = 0.0
            for g in range(5):
                m = mask & (y == g)
                r += w[g] * (wr[m].mean() if m.any() else np.nan)
            return r

        print("\nGrade-standardised error rate by quartile (same grade mix in every quartile):\n")
        print("| statistic | Q1 | Q2 | Q3 | Q4 | Q4 − Q1 [95% CI] | DR images only: Q1 → Q4 |\n|---|---|---|---|---|---|---|")
        for col in ("raw_brightness", "raw_contrast", "raw_sharpness", "raw_pixels", "retina_frac"):
            q = pd.qcut(st[col].rank(method="first"), 4, labels=False).to_numpy()
            rates = [std_rate(q == k, wrong) for k in range(4)]
            boots = []
            for _ in range(500):
                ix = rng.integers(0, len(y), len(y))
                yy, ww, qq = y[ix], wrong[ix], q[ix]
                rr = []
                for k in (0, 3):
                    rr.append(sum(w[g] * ww[(qq == k) & (yy == g)].mean() for g in range(5)
                                  if ((qq == k) & (yy == g)).any()))
                boots.append(rr[1] - rr[0])
            dr = y >= 1
            print(f"| {col} | " + " | ".join(f"{r:.3f}" for r in rates) +
                  f" | {rates[3] - rates[0]:+.3f} [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]"
                  f" | {wrong[dr & (q == 0)].mean():.3f} → {wrong[dr & (q == 3)].mean():.3f} |")


if __name__ == "__main__":
    main()
