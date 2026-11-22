"""
Phase 3: score the ONE-SHOT test predictions with the FROZEN configuration (local, CPU).

    python src/evaluation/final_test.py --preds kaggle/runs/rg-final-test/final_test_preds.npz \
                                        --frozen docs/frozen_config.json

The frozen config (written BEFORE the test kernel ran) fixes everything that is applied here:
  primary          : run id of the deployable model
  ensemble         : run ids averaged for the secondary "ensemble" row
  thresholds       : {"referable": t2, "sight_threatening": t3} applied to P(grade>=2), P(grade>=3)
  grade rule       : argmax of the 5 probabilities
Nothing is tuned on the test data. CIs: 2,000 bootstrap resamples (EyePACS: by patient).
v1 comparison: v1's stored predictions on the same 733 APTOS images (rg-v1-repro), paired.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import metrics as M  # noqa: E402

KEYS = ["qwk", "acc", "within1", "auc1", "auc2", "auc3", "ref_sens", "ref_spec", "st_sens", "st_spec"]
LABEL = {"qwk": "QWK", "acc": "accuracy", "within1": "within 1 grade", "auc1": "AUC any DR (≥1)",
         "auc2": "AUC referable (≥2)", "auc3": "AUC sight-threat. (≥3)", "ref_sens": "referable sensitivity",
         "ref_spec": "referable specificity", "st_sens": "sight-threat. sensitivity", "st_spec": "sight-threat. specificity"}


def metrics(y, P, thr):
    pred = P.argmax(1)
    s2, s3 = P[:, 2:].sum(1), P[:, 3:].sum(1)
    out = {"qwk": M.qwk(y, pred), "acc": float((pred == y).mean()), "within1": float((np.abs(pred - y) <= 1).mean())}
    for k in (1, 2, 3):
        yb = y >= k
        out[f"auc{k}"] = M.binary_auc(yb.astype(int), P[:, k:].sum(1)) if 0 < yb.sum() < len(yb) else np.nan
    for key, s, k, t in (("ref", s2, 2, thr["referable"]), ("st", s3, 3, thr["sight_threatening"])):
        yb, pos = y >= k, s >= t
        out[f"{key}_sens"] = float(pos[yb].mean()) if yb.any() else np.nan
        out[f"{key}_spec"] = float((~pos[~yb]).mean()) if (~yb).any() else np.nan
    return out


def boot(y, P, thr, groups=None, n_boot=2000, seed=0, P_ref=None):
    """95% CI of metrics(P) (or of metrics(P) - metrics(P_ref)); resampling groups (patients) if given."""
    rng = np.random.default_rng(seed)
    if groups is None:
        rows = [np.array([i]) for i in range(len(y))]
    else:
        _, inv = np.unique(groups, return_inverse=True)
        rows = [np.where(inv == g)[0] for g in range(inv.max() + 1)]
    vals = []
    for _ in range(n_boot):
        ix = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        a = metrics(y[ix], P[ix], thr)
        b = metrics(y[ix], P_ref[ix], thr) if P_ref is not None else {k: 0.0 for k in KEYS}
        vals.append([a[k] - b[k] for k in KEYS])
    v = np.array(vals)
    return {k: (float(np.nanpercentile(v[:, j], 2.5)), float(np.nanpercentile(v[:, j], 97.5))) for j, k in enumerate(KEYS)}


def table(rows, cis):
    names = list(rows)
    L = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in KEYS:
        cells = []
        for n in names:
            v, ci = rows[n][k], cis.get(n, {}).get(k)
            cells.append("—" if v is None or (isinstance(v, float) and np.isnan(v)) else
                         f"{v:.3f}" + (f" [{ci[0]:.3f}–{ci[1]:.3f}]" if ci else ""))
        L.append(f"| {LABEL[k]} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def v1_probs(runs):
    """v1 cascade on the old val set (= the 733 test ids) as 5-class probabilities, keyed by id_code."""
    z = np.load(os.path.join(runs, "rg-v1-repro", "val_probs_v1.npz"))
    xmap = pd.read_csv(os.path.join(runs, "rg-recon", "xval_map.csv")).sort_values("row")
    P = M.cascade_probs5(z["p1"], z["p2"])
    pred = M.cascade_predict(z["p1"], z["p2"], float(z["threshold"]))
    return dict(zip(xmap.id_code, P)), dict(zip(xmap.id_code, pred))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="kaggle/runs")
    ap.add_argument("--preds", default="kaggle/runs/rg-final-test/final_test_preds.npz")
    ap.add_argument("--frozen", default="docs/frozen_config.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json", default="kaggle/runs/rg-final-test/final_test_report.json")
    a = ap.parse_args(argv)

    cfg = json.load(open(a.frozen))
    thr = cfg["thresholds"]
    z = dict(np.load(a.preds, allow_pickle=False))
    report = {"frozen": cfg}
    print(f"## One-shot test evaluation — frozen config `{cfg['name']}`\n")
    print(f"Primary model: `{cfg['primary']}` · ensemble: {len(cfg['ensemble'])} models · thresholds "
          f"P(≥2) ≥ {thr['referable']:.3f}, P(≥3) ≥ {thr['sight_threatening']:.3f} · grade = argmax\n")

    for ds in ("aptos", "idrid", "eyepacs"):
        y = z[f"{ds}|y"]
        groups = z.get(f"{ds}|patient")
        rows, cis = {}, {}
        P1 = z[f"{ds}|{cfg['primary']}"]
        PE = np.mean([z[f"{ds}|{r}"] for r in cfg["ensemble"]], 0)
        rows["primary model"], cis["primary model"] = metrics(y, P1, thr), boot(y, P1, thr, groups, a.n_boot)
        rows["ensemble"], cis["ensemble"] = metrics(y, PE, thr), boot(y, PE, thr, groups, a.n_boot)
        if ds == "aptos":
            v1p, v1pred = v1_probs(a.runs)
            ids = z["aptos|id"]
            Pv1 = np.stack([v1p[i] for i in ids])
            m = metrics(y, Pv1, thr)
            m["qwk"] = M.qwk(y, np.array([v1pred[i] for i in ids]))            # v1's own decision rule
            m["acc"] = float((np.array([v1pred[i] for i in ids]) == y).mean())
            rows["v1 cascade (thr 0.3)"] = m
            d = boot(y, P1, thr, None, a.n_boot, P_ref=Pv1)
            rows["Δ primary − v1 [CI]"] = {k: rows["primary model"][k] - m[k] for k in KEYS}
            cis["Δ primary − v1 [CI]"] = d
        print(f"### {ds.upper()} test (n={len(y)}" + (f", {len(np.unique(groups))} patients" if groups is not None else "")
              + f"; grades {np.bincount(y, minlength=5).tolist()})\n")
        print(table(rows, cis))
        C = M.confusion(y, P1.argmax(1), 5)
        print("\nPrimary model confusion matrix (rows = true, cols = predicted):\n")
        print("| true \\ pred | 0 | 1 | 2 | 3 | 4 | recall |\n|---|---|---|---|---|---|---|")
        for g in range(5):
            print(f"| {g} | " + " | ".join(str(int(v)) for v in C[g]) +
                  f" | {C[g, g] / C[g].sum() if C[g].sum() else float('nan'):.2f} |")
        print()
        report[ds] = {"n": int(len(y)), "grades": np.bincount(y, minlength=5).tolist(), "rows": rows, "ci": cis,
                      "confusion_primary": C.tolist()}
        if ds == "eyepacs" and f"{ds}|quality" in z:
            q = z[f"{ds}|quality"]
            print("EyePACS by EyeQ quality (primary model):\n")
            print("| quality | n | QWK | AUC ≥2 | AUC ≥3 | ref. sens/spec |\n|---|---|---|---|---|---|")
            for qq, name in ((0, "Good"), (1, "Usable"), (2, "Reject"), (-1, "no label")):
                mm = q == qq
                if mm.sum() >= 30:
                    r = metrics(y[mm], P1[mm], thr)
                    print(f"| {name} | {int(mm.sum())} | {r['qwk']:.3f} | {r['auc2']:.3f} | {r['auc3']:.3f} | "
                          f"{r['ref_sens']:.3f}/{r['ref_spec']:.3f} |")
            print()
    with open(a.json, "w") as f:
        json.dump(report, f, indent=1, default=float)


if __name__ == "__main__":
    main()
