"""
B vs C at matched operating points (local, CPU, OOF predictions only — never the test split).

    python src/evaluation/operating_points.py --runs kaggle/runs

Question answered: is C's higher referable / sight-threatening sensitivity a better *ranking* of
patients, or only a different decision threshold?  We compare threshold-free AUC and sensitivity at
the SAME specificity for both models.

Continuous scores (one per binary task, higher = sicker):
  C : raw regression output s                      (same score for >=2 and >=3)
  B : P(grade >= k) = sum(probs5[:, k:])           (main)  and expected grade E = sum(i * p_i) (alt.)

Specificity-matched thresholds, two flavours:
  oracle      : threshold set on the same pooled OOF   -> pure ranking comparison (ROC read-off)
  cross-fit   : threshold set on the other 4 folds, applied to the 5th -> honest operating point

Uncertainty:
  * paired per-(seed, fold) differences C - B  (15 pairs), mean +/- std
  * paired image bootstrap (same resampled images for B and C, averaged over the 3 seeds), 95% CI
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import metrics as M  # noqa: E402

SEEDS = (7, 42, 1234)
TASKS = (2, 3)                       # referable (>=2), sight-threatening (>=3)
SPECS = (0.95, 0.90)


# ----------------------------------------------------------------------------- data
def load_oof(runs_dir: str, cand: str, seed: int, suffix: str = "", img: int = 224) -> dict:
    """Stitch the 5 fold-val files of one (candidate, seed) into OOF arrays sorted by id_code.
    suffix="_tta" loads the re-scored TTA predictions written by src/training/predict.py."""
    parts = []
    for f in range(5):
        ps = glob.glob(os.path.join(runs_dir, "**", f"{cand}_f{f}_s{seed}_{img}{suffix}", "oof_preds.npz"),
                       recursive=True)
        assert len(ps) == 1, (cand, seed, f, ps)
        with np.load(ps[0], allow_pickle=False) as z:
            parts.append({k: z[k] for k in z.files})
    ids = np.concatenate([p["id_code"] for p in parts])
    order = np.argsort(ids)
    out = {"id": ids[order], "y": np.concatenate([p["y_true"] for p in parts])[order],
           "fold": np.concatenate([np.full(len(p["y_true"]), f) for f, p in enumerate(parts)])[order]}
    if cand == "C":
        out["s"] = np.concatenate([p["score"] for p in parts])[order]
    else:
        probs = np.concatenate([p["probs5"] for p in parts])[order]
        out["probs5"] = probs
        out["E"] = probs @ np.arange(5)
        if "p1" in parts[0]:                                   # cascade (A): Stage-1 P(any DR)
            out["p1"] = np.concatenate([p["p1"] for p in parts]).astype(np.float64)[order]
    return out


def score_for(d: dict, k: int, b_score: str = "pk") -> np.ndarray:
    if "s" in d:
        return d["s"]
    return d["probs5"][:, k:].sum(1) if b_score == "pk" else d["E"]


# ----------------------------------------------------------------------------- operating points
def thr_at_spec(score_neg: np.ndarray, spec: float) -> float:
    """Smallest threshold t with  mean(score_neg < t) >= spec  (positive call = score >= t)."""
    s = np.sort(score_neg)
    i = int(np.ceil(spec * len(s))) - 1
    i = min(max(i, 0), len(s) - 1)
    t = s[i]
    return float(np.nextafter(t, np.inf))       # strictly above that negative -> spec >= target


def sens_spec(y_bin, score, t):
    pos = score >= t
    return float(pos[y_bin].mean()), float((~pos[~y_bin]).mean())


def oracle_sens_at_spec(y_bin, score, spec):
    return sens_spec(y_bin, score, thr_at_spec(score[~y_bin], spec))


def crossfit_sens_at_spec(y_bin, score, fold, spec):
    pos = np.zeros(len(y_bin), bool)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        t = thr_at_spec(score[tr & ~y_bin], spec)
        pos[te] = score[te] >= t
    return float(pos[y_bin].mean()), float((~pos[~y_bin]).mean())


def task_metrics(y, score, fold, k):
    yb = y >= k
    out = {"auc": M.binary_auc(yb.astype(int), score)}
    for sp in SPECS:
        out[f"sens@{sp}"], _ = oracle_sens_at_spec(yb, score, sp)
        out[f"cf_sens@{sp}"], out[f"cf_spec@{sp}"] = crossfit_sens_at_spec(yb, score, fold, sp)
    return out


# ----------------------------------------------------------------------------- analysis
def analyse(runs_dir: str, n_boot: int = 2000, seed: int = 0, b_score: str = "pk") -> dict:
    B = {s: load_oof(runs_dir, "B", s) for s in SEEDS}
    C = {s: load_oof(runs_dir, "C", s) for s in SEEDS}
    for s in SEEDS:     # identical images, labels and folds -> paired comparison is valid
        assert np.array_equal(B[s]["id"], C[s]["id"]) and np.array_equal(B[s]["y"], C[s]["y"])
        assert np.array_equal(B[s]["fold"], C[s]["fold"]) and np.array_equal(B[s]["id"], B[SEEDS[0]]["id"])
    y, fold = B[SEEDS[0]]["y"], B[SEEDS[0]]["fold"]
    keys = (["auc"] + [f"sens@{sp}" for sp in SPECS] + [f"cf_sens@{sp}" for sp in SPECS]
            + [f"cf_spec@{sp}" for sp in SPECS])

    res = {"n": int(len(y)), "b_score": b_score, "tasks": {}}
    rng = np.random.default_rng(seed)
    boot_idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]

    for k in TASKS:
        per_seed = {m: {} for m in ("B", "C")}
        for s in SEEDS:
            per_seed["B"][s] = task_metrics(y, score_for(B[s], k, b_score), fold, k)
            per_seed["C"][s] = task_metrics(y, score_for(C[s], k), fold, k)

        # per-(seed, fold) paired differences (AUC and oracle sensitivity inside the fold)
        fold_diffs = {"auc": [], "sens@0.95": [], "sens@0.9": []}
        for s in SEEDS:
            for f in range(5):
                m = fold == f
                yb = y[m] >= k
                sb, sc = score_for(B[s], k, b_score)[m], score_for(C[s], k)[m]
                fold_diffs["auc"].append(M.binary_auc(yb.astype(int), sc) - M.binary_auc(yb.astype(int), sb))
                for sp in SPECS:
                    fold_diffs[f"sens@{sp}"].append(oracle_sens_at_spec(yb, sc, sp)[0] - oracle_sens_at_spec(yb, sb, sp)[0])

        # paired image bootstrap of the seed-averaged difference (oracle metrics; threshold refit per resample)
        boots = {k2: [] for k2 in ("auc", "sens@0.95", "sens@0.9")}
        for idx in boot_idx:
            yb = y[idx] >= k
            if yb.all() or (~yb).all():
                continue
            d = {k2: [] for k2 in boots}
            for s in SEEDS:
                sb, sc = score_for(B[s], k, b_score)[idx], score_for(C[s], k)[idx]
                d["auc"].append(M.binary_auc(yb.astype(int), sc) - M.binary_auc(yb.astype(int), sb))
                for sp in SPECS:
                    d[f"sens@{sp}"].append(oracle_sens_at_spec(yb, sc, sp)[0] - oracle_sens_at_spec(yb, sb, sp)[0])
            for k2 in boots:
                boots[k2].append(np.mean(d[k2]))

        t = {"per_seed": per_seed,
             "mean": {m: {kk: float(np.mean([per_seed[m][s][kk] for s in SEEDS])) for kk in keys} for m in ("B", "C")},
             "std": {m: {kk: float(np.std([per_seed[m][s][kk] for s in SEEDS], ddof=1)) for kk in keys} for m in ("B", "C")},
             "fold_diff": {kk: [float(np.mean(v)), float(np.std(v, ddof=1)), int(np.sum(np.array(v) > 0)), len(v)]
                           for kk, v in fold_diffs.items()},
             "boot_diff": {kk: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                           for kk, v in boots.items()}}
        res["tasks"][k] = t
    return res


def render(res: dict) -> str:
    L = [f"### B vs C at matched operating points (OOF, n={res['n']}, 3 seeds; B score = "
         f"{'P(grade>=k)' if res['b_score'] == 'pk' else 'expected grade'})", ""]
    names = {2: "referable (>=2)", 3: "sight-threatening (>=3)"}
    for k, t in res["tasks"].items():
        k = int(k)
        L += [f"**{names[k]}**", "",
              "| metric | B (mean ± std) | C (mean ± std) | C − B, paired bootstrap [95% CI] | C − B per fold (mean ± std, C wins) |",
              "|---|---|---|---|---|"]
        for kk, lab in [("auc", "AUC"), ("sens@0.95", "sens @ spec 0.95 (ROC)"), ("sens@0.9", "sens @ spec 0.90 (ROC)"),
                        ("cf_sens@0.95", "sens @ spec 0.95 (cross-fit thr)"), ("cf_sens@0.9", "sens @ spec 0.90 (cross-fit thr)")]:
            b = f"{t['mean']['B'][kk]:.3f} ± {t['std']['B'][kk]:.3f}"
            c = f"{t['mean']['C'][kk]:.3f} ± {t['std']['C'][kk]:.3f}"
            bd = t["boot_diff"].get(kk)
            fd = t["fold_diff"].get(kk)
            bd_s = f"{bd[0]:+.3f} [{bd[1]:+.3f}, {bd[2]:+.3f}]" if bd else "—"
            fd_s = f"{fd[0]:+.3f} ± {fd[1]:.3f} ({fd[2]}/{fd[3]})" if fd else "—"
            L.append(f"| {lab} | {b} | {c} | {bd_s} | {fd_s} |")
        cf_spec = " · ".join(f"spec {sp}: B {t['mean']['B'][f'cf_spec@{sp}']:.3f}, C {t['mean']['C'][f'cf_spec@{sp}']:.3f}"
                             for sp in SPECS)
        L += ["", f"Achieved specificity with cross-fitted thresholds — {cf_spec}", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="kaggle/runs")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--b-score", choices=["pk", "E"], default="pk")
    ap.add_argument("--json", help="also write the raw numbers here")
    a = ap.parse_args(argv)
    res = analyse(a.runs, a.n_boot, b_score=a.b_score)
    print(render(res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
