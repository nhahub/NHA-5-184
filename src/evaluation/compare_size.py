"""
Final image-size decision for B: 384px vs 224px, full 5-fold OOF x 3 seeds (local, CPU).

    python src/evaluation/compare_size.py --runs kaggle/runs

Decision rule (fixed BEFORE the full run, 2026-11-20):
    adopt 384  iff  mean OOF QWK(384) - mean OOF QWK(224) >= +0.01   (3 seeds each)
               and  mean AUC>=3 (384) is not lower than at 224 (difference CI does not lie below 0)
    otherwise freeze B at 224.

Uncertainty: paired image bootstrap of the seed-averaged difference (same images, same folds);
plus the per-seed and per-(seed, fold) differences.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import metrics as M  # noqa: E402
from evaluation.operating_points import SEEDS, crossfit_sens_at_spec, load_oof, oracle_sens_at_spec  # noqa: E402

KEYS = ["qwk", "acc", "auc2", "auc3", "sens2@0.95", "sens3@0.95", "sens3@0.9"]


def full(y, p, fold):
    pred = p.argmax(1)
    out = {"qwk": M.qwk(y, pred), "acc": float((pred == y).mean())}
    for k in (2, 3):
        out[f"auc{k}"] = M.binary_auc((y >= k).astype(int), p[:, k:].sum(1))
    out["sens2@0.95"] = crossfit_sens_at_spec(y >= 2, p[:, 2:].sum(1), fold, 0.95)[0]
    out["sens3@0.95"] = crossfit_sens_at_spec(y >= 3, p[:, 3:].sum(1), fold, 0.95)[0]
    out["sens3@0.9"] = crossfit_sens_at_spec(y >= 3, p[:, 3:].sum(1), fold, 0.90)[0]
    return out


def light(y, p):
    v = [M.qwk(y, p.argmax(1)), float((p.argmax(1) == y).mean())]
    v += [M.binary_auc((y >= k).astype(int), p[:, k:].sum(1)) for k in (2, 3)]
    v += [oracle_sens_at_spec(y >= 2, p[:, 2:].sum(1), 0.95)[0], oracle_sens_at_spec(y >= 3, p[:, 3:].sum(1), 0.95)[0],
          oracle_sens_at_spec(y >= 3, p[:, 3:].sum(1), 0.90)[0]]
    return np.array(v)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="kaggle/runs")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)

    big = {s: load_oof(a.runs, "B", s, img=384) for s in SEEDS}
    small = {s: load_oof(a.runs, "B", s, img=224) for s in SEEDS}
    for s in SEEDS:
        for d in (big[s], small[s]):
            assert np.array_equal(d["id"], small[42]["id"]) and np.array_equal(d["fold"], small[42]["fold"])
    y, fold = small[42]["y"], small[42]["fold"]

    per = {name: {s: full(y, d[s]["probs5"], fold) for s in SEEDS} for name, d in (("224", small), ("384", big))}
    print(f"### B: 384px vs 224px — 5-fold OOF, 3 seeds (n={len(y)})\n")
    print("| size | " + " | ".join(KEYS) + " |\n|---|" + "---|" * len(KEYS))
    for name in ("224", "384"):
        print(f"| {name} | " + " | ".join(
            f"{np.mean([per[name][s][k] for s in SEEDS]):.3f} ± {np.std([per[name][s][k] for s in SEEDS], ddof=1):.3f}"
            for k in KEYS) + " |")
    print("| Δ per seed (7, 42, 1234) | " + " | ".join(
        ", ".join(f"{per['384'][s][k] - per['224'][s][k]:+.3f}" for s in SEEDS) for k in KEYS) + " |")

    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(a.n_boot):
        ix = rng.integers(0, len(y), len(y))
        diffs.append(np.mean([light(y[ix], big[s]["probs5"][ix]) for s in SEEDS], 0)
                     - np.mean([light(y[ix], small[s]["probs5"][ix]) for s in SEEDS], 0))
    diffs = np.array(diffs)
    ci = {k: (np.percentile(diffs[:, j], 2.5), np.percentile(diffs[:, j], 97.5)) for j, k in enumerate(KEYS)}
    print("| Δ seed-averaged, paired bootstrap 95% CI | " +
          " | ".join(f"[{ci[k][0]:+.3f}, {ci[k][1]:+.3f}]" for k in KEYS) + " |")

    fd = [M.qwk(y[fold == f], big[s]["probs5"][fold == f].argmax(1)) - M.qwk(y[fold == f], small[s]["probs5"][fold == f].argmax(1))
          for s in SEEDS for f in range(5)]
    print(f"\nPer (seed, fold) QWK difference: mean {np.mean(fd):+.3f}, 384 better in {int(np.sum(np.array(fd) > 0))}/15")

    dq = np.mean([per["384"][s]["qwk"] - per["224"][s]["qwk"] for s in SEEDS])
    auc3_ok = ci["auc3"][0] >= 0 or np.mean([per["384"][s]["auc3"] - per["224"][s]["auc3"] for s in SEEDS]) >= 0
    verdict = "ADOPT 384" if (dq >= 0.01 and auc3_ok) else "FREEZE 224"
    print(f"\nDecision rule: ΔQWK {dq:+.3f} (need ≥ +0.010), AUC≥3 not worse: {auc3_ok}  →  **{verdict}**")


if __name__ == "__main__":
    main()
