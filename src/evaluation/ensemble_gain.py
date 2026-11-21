"""
How much do TTA and model averaging add to B? (local, CPU, OOF only)

    python src/evaluation/ensemble_gain.py --runs kaggle/runs [--cand B]

Why a *seed* ensemble: the deployable model is the average of the 5 fold models, but every training
image was seen by 4 of them, so a 5-fold ensemble has no honest OOF score; it is measured once on the
test split. Averaging the 3 seed models trained on the SAME fold is a valid OOF stand-in for the gain
from averaging several models.

Rows: single model (mean over 3 seeds) vs 3-seed average, each without / with 4-view flip TTA
(TTA rows appear once the rg-predict outputs are pulled). Deltas use a paired image bootstrap.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import metrics as M  # noqa: E402
from evaluation.operating_points import SEEDS, crossfit_sens_at_spec, load_oof  # noqa: E402


def probs_metrics(y, probs, fold) -> dict:
    """B: grade = argmax; referable / sight-threatening scores = P(grade >= k)."""
    pred = probs.argmax(1)
    out = {"qwk": M.qwk(y, pred), "acc": float((pred == y).mean())}
    for k in (2, 3):
        s, yb = probs[:, k:].sum(1), y >= k
        out[f"auc{k}"] = M.binary_auc(yb.astype(int), s)
        out[f"sens{k}@0.95"] = crossfit_sens_at_spec(yb, s, fold, 0.95)[0]
        out[f"sens{k}@0.9"] = crossfit_sens_at_spec(yb, s, fold, 0.90)[0]
    return out


def score_metrics(y, score, fold) -> dict:
    """C: grade = default cuts; one continuous score for both binary tasks."""
    pred = M.apply_thresholds(score)
    out = {"qwk": M.qwk(y, pred), "acc": float((pred == y).mean())}
    for k in (2, 3):
        yb = y >= k
        out[f"auc{k}"] = M.binary_auc(yb.astype(int), score)
        out[f"sens{k}@0.95"] = crossfit_sens_at_spec(yb, score, fold, 0.95)[0]
        out[f"sens{k}@0.9"] = crossfit_sens_at_spec(yb, score, fold, 0.90)[0]
    return out


def analyse(runs_dir, cand="B", n_boot=500, seed=0):
    rows, raw = {}, {}
    for tag, suffix in (("no TTA", ""), ("TTA", "_tta")):
        try:
            d = {s: load_oof(runs_dir, cand, s, suffix) for s in SEEDS}
        except AssertionError:
            continue                                    # TTA outputs not pulled yet
        key = "probs5" if cand == "B" else "s"
        fn = probs_metrics if cand == "B" else score_metrics
        y, fold = d[SEEDS[0]]["y"], d[SEEDS[0]]["fold"]
        singles = [d[s][key] for s in SEEDS]
        ens = np.mean(singles, axis=0)
        raw[tag] = (y, fold, singles, ens)
        per = [fn(y, p, fold) for p in singles]
        rows[f"single, {tag}"] = {k: (float(np.mean([m[k] for m in per])), float(np.std([m[k] for m in per], ddof=1)))
                                  for k in per[0]}
        rows[f"3-seed avg, {tag}"] = {k: (v, 0.0) for k, v in fn(y, ens, fold).items()}

    # paired bootstrap deltas vs "single, no TTA" (QWK and AUCs; seed-averaged single-model metric)
    fn = probs_metrics if cand == "B" else score_metrics
    deltas = {}
    if "no TTA" in raw:
        y, fold, base_singles, _ = raw["no TTA"]
        rng = np.random.default_rng(seed)
        idxs = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]

        def light(yy, p):
            pred = p.argmax(1) if p.ndim == 2 else M.apply_thresholds(p)
            sc = (lambda k: p[:, k:].sum(1)) if p.ndim == 2 else (lambda k: p)
            return np.array([M.qwk(yy, pred)] + [M.binary_auc((yy >= k).astype(int), sc(k)) for k in (2, 3)])

        for tag, (_, _, singles, ens) in raw.items():
            for label, preds in ((f"single, {tag}", singles), (f"3-seed avg, {tag}", [ens])):
                if label == "single, no TTA":
                    continue
                ds = []
                for ix in idxs:
                    base = np.mean([light(y[ix], p[ix]) for p in base_singles], axis=0)
                    new = np.mean([light(y[ix], p[ix]) for p in preds], axis=0)
                    ds.append(new - base)
                ds = np.array(ds)
                deltas[label] = [(float(ds[:, j].mean()), float(np.percentile(ds[:, j], 2.5)),
                                  float(np.percentile(ds[:, j], 97.5))) for j in range(3)]
    return rows, deltas


def render(rows, deltas, cand):
    cols = ["qwk", "acc", "auc2", "sens2@0.95", "sens2@0.9", "auc3", "sens3@0.95", "sens3@0.9"]
    L = [f"### {cand}: TTA and model-averaging gains (OOF, 3 seeds; sens at fixed spec, cross-fitted thresholds)", "",
         "| setting | QWK | acc | AUC ≥2 | sens ≥2 @ .95 | sens ≥2 @ .90 | AUC ≥3 | sens ≥3 @ .95 | sens ≥3 @ .90 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, m in rows.items():
        cells = [f"{m[c][0]:.3f}" + (f" ± {m[c][1]:.3f}" if m[c][1] else "") for c in cols]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    if deltas:
        L += ["", "Δ vs single model without TTA (paired image bootstrap, 95% CI):", "",
              "| setting | Δ QWK | Δ AUC ≥2 | Δ AUC ≥3 |", "|---|---|---|---|"]
        for name, d in deltas.items():
            L.append(f"| {name} | " + " | ".join(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]" for m, lo, hi in d) + " |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="kaggle/runs")
    ap.add_argument("--cand", default="B", choices=["B", "C"])
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args(argv)
    rows, deltas = analyse(a.runs, a.cand, a.n_boot)
    print(render(rows, deltas, a.cand))


if __name__ == "__main__":
    main()
