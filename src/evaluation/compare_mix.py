"""
Mixed-training pilot: B 384 trained on APTOS + EyePACS vs B 384 trained on APTOS only.
Same seed (42), same folds (0, 1), same APTOS validation images, same EyePACS dev images.

    python src/evaluation/compare_mix.py --runs kaggle/runs

Decision rule (fixed BEFORE the pilot, 2026-11-20):
    adopt mixed training  iff  EyePACS dev AUC>=2 improves by >= +0.05
                          and  APTOS fold-val QWK drops by <= 0.02
Both differences are paired (same images); CIs from a paired bootstrap (EyePACS: by patient).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import metrics as M  # noqa: E402
from evaluation.compare_384 import COLS, boot_diff, full, load  # noqa: E402
from evaluation.eyepacs_gap import KEYS, patient_boot, point  # noqa: E402

FOLDS = (0, 1)


def dev_preds(runs):
    """EyePACS dev probabilities: APTOS-only 384 (from rg-eyepacs-dev) and mixed (from the training runs)."""
    z = dict(np.load(glob.glob(os.path.join(runs, "**", "rg-eyepacs-dev", "eyepacs_dev_preds.npz"), recursive=True)[0]))
    base_stem = z["stem"]
    base = np.mean([z[f"B_f{f}_s42_384|hires384"] for f in FOLDS], 0)
    mix, stems = [], None
    for f in FOLDS:
        p = glob.glob(os.path.join(runs, "**", f"B_f{f}_s42_384_mix", "eyepacs_dev_preds.npz"), recursive=True)
        assert len(p) == 1, (f, p)
        m = dict(np.load(p[0]))
        o = np.argsort(m["stem"])
        stems = m["stem"][o] if stems is None else stems
        assert np.array_equal(stems, m["stem"][o])
        mix.append(m["probs5"][o])
        y, patient = m["y"][o], m["patient"][o]
    pos = {s: i for i, s in enumerate(base_stem)}
    ix = np.array([pos[s] for s in stems])
    assert np.array_equal(z["y"][ix], y), "labels differ between the two dev files"
    return y, patient, base[ix], np.mean(mix, 0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="kaggle/runs")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)

    base = load(a.runs, 42, 384, list(FOLDS))
    mix = load(a.runs, 42, 384, list(FOLDS), "_mix")
    assert np.array_equal(base["id"], mix["id"])
    y, fold = base["y"], base["fold"]
    print(f"### Mixed-training pilot (B 384, seed 42, folds {list(FOLDS)})\n")
    print(f"**APTOS fold-val** (n={len(y)})\n")
    print("| model | " + " | ".join(COLS) + " |\n|---|" + "---|" * len(COLS))
    mb, mm = full(y, base["p"], fold), full(y, mix["p"], fold)
    print("| APTOS only | " + " | ".join(f"{mb[c]:.3f}" for c in COLS) + " |")
    print("| APTOS + EyePACS | " + " | ".join(f"{mm[c]:.3f}" for c in COLS) + " |")
    r = boot_diff(y, mix["p"], base["p"], a.n_boot)
    print("| Δ [95% CI] | " + " | ".join(f"{m:+.3f} [{l:+.3f}, {h:+.3f}]" for m, l, h in r) + " |")

    yd, pat, pb, pm = dev_preds(a.runs)
    print(f"\n**EyePACS dev patients** (n={len(yd)}, {len(np.unique(pat))} patients; fold 0+1 models averaged)\n")
    print("| model | " + " | ".join(KEYS) + " |\n|---|" + "---|" * len(KEYS))
    db, dm = point(yd, pb), point(yd, pm)
    print("| APTOS only | " + " | ".join(f"{db[k]:.3f}" for k in KEYS) + " |")
    print("| APTOS + EyePACS | " + " | ".join(f"{dm[k]:.3f}" for k in KEYS) + " |")
    ci = patient_boot(yd, pat, pm, pb, n_boot=a.n_boot)
    print("| Δ [95% CI] | " + " | ".join(f"{dm[k] - db[k]:+.3f} [{ci[k][0]:+.3f}, {ci[k][1]:+.3f}]" for k in KEYS) + " |")
    for k in (1, 2):
        m = (yd == 0) | (yd == k)
        print(f"- EyePACS grade 0 vs {k} AUC: {M.binary_auc((yd[m] > 0).astype(int), 1 - pb[m, 0]):.3f} → "
              f"{M.binary_auc((yd[m] > 0).astype(int), 1 - pm[m, 0]):.3f}")

    d_auc2, d_qwk = dm["auc2"] - db["auc2"], mm["qwk"] - mb["qwk"]
    verdict = "ADOPT mixed training" if (d_auc2 >= 0.05 and d_qwk >= -0.02) else "KEEP APTOS-only"
    print(f"\nDecision rule: EyePACS ΔAUC≥2 {d_auc2:+.3f} (need ≥ +0.05), APTOS ΔQWK {d_qwk:+.3f} (need ≥ −0.02) → **{verdict}**")


if __name__ == "__main__":
    main()
