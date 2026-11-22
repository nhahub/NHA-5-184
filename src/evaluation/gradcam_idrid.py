"""
IDRiD Grad-CAM validation: tables from the per-image CSV of kaggle/kernels/rg_gradcam.py (local, CPU).

    python src/evaluation/gradcam_idrid.py --csv kaggle/runs/rg-gradcam/gradcam_idrid.csv

Strict numbers use undilated lesions (tol=0); tol=16 px forgives the 12x12 Grad-CAM grid.
CIs: paired bootstrap over images (2,000 resamples). IDRiD is analysis only: nothing is chosen from it.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explain import lesion_overlap as L  # noqa: E402

MAPS = ["gradcam_referable", "gradcam_pred", "random_weights", "centre", "optic_disc", "random"]
COLS = ["pointing", "energy", "precision@10", "recall@20", "enrichment@10"]


def table(d, tol):
    t = d[d.tol == tol]
    g = t.groupby("map")[COLS + ["lesion_fraction"]].mean()
    L_ = [f"**tol = {tol} px** (n = {t.image.nunique()} images; mean over images)", "",
          "| heatmap | pointing game | energy | precision@10% | recall@20% | enrichment@10% |", "|---|---|---|---|---|---|"]
    for m in MAPS:
        L_.append(f"| {m} | " + " | ".join(f"{g.loc[m, c]:.3f}" for c in COLS) + " |")
    L_.append(f"\nLesions cover {g['lesion_fraction'].mean():.1%} of the retina on average (this is what a random map scores).")
    return "\n".join(L_)


def diffs(d, tol):
    t = d[d.tol == tol].pivot_table(index="image", columns="map", values=COLS)
    L_ = ["| difference (Grad-CAM referral − baseline) | " + " | ".join(COLS) + " |", "|---|" + "---|" * len(COLS)]
    for b in ("random", "random_weights", "centre", "optic_disc"):
        cells = []
        for c in COLS:
            m, lo, hi, n = L.bootstrap_mean_diff(t[c]["gradcam_referable"], t[c][b])
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        L_.append(f"| vs {b} | " + " | ".join(cells) + " |")
    return "\n".join(L_)


def subgroups(d, tol=0):
    a = d[(d.tol == tol) & (d["map"] == "gradcam_referable")].copy()
    a["size"] = pd.qcut(a.lesion_fraction, 3, labels=["small lesions", "medium", "large"])
    out = ["**By lesion load (tol 0)**", "", "| group | n | lesions / retina | pointing | enrichment@10% |", "|---|---|---|---|---|"]
    for g, x in a.groupby("size", observed=True):
        out.append(f"| {g} | {len(x)} | {x.lesion_fraction.mean():.1%} | {x.pointing.mean():.2f} | {x['enrichment@10'].mean():.2f} |")
    out += ["", "**By model decision (tol 0)**", "", "| group | n | pointing | enrichment@10% |", "|---|---|---|---|"]
    for name, m in (("referable flag on", a.referable_flag), ("flag off", ~a.referable_flag.astype(bool))):
        x = a[m.astype(bool)]
        out.append(f"| {name} | {len(x)} | {x.pointing.mean():.2f} | {x['enrichment@10'].mean():.2f} |")
    out += ["", "**By lesion type** (share of the type's pixels inside the hottest 20%): " +
            ", ".join(f"{t} {a[f'recall20_{t}'].mean():.2f}" for t in L.TYPES if f"recall20_{t}" in a)]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="kaggle/runs/rg-gradcam/gradcam_idrid.csv")
    a = ap.parse_args(argv)
    d = pd.read_csv(a.csv)
    for tol in (0, 16):
        print(table(d, tol) + "\n\n" + diffs(d, tol) + "\n")
    print(subgroups(d))


if __name__ == "__main__":
    main()
