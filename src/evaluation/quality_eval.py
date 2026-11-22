"""
Evaluation of the quality gate (local, CPU, saved predictions only).

    python src/evaluation/quality_eval.py --runs kaggle/runs

  * qtest = 1,199 EyePACS DEV-patient images with EyeQ labels; patients never seen in training or epoch selection
  * CIs: bootstrap over patients (both eyes together)
  * grading accuracy by image quality: the frozen grading model's stored predictions on the same dev images,
    split by the EyeQ label and by the gate's own label (no test split is read)
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
from evaluation import quality_metrics as Q  # noqa: E402

DEPLOY_SEED = 42            # fixed before training


def boot(fn, patient, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    pats, inv = np.unique(patient, return_inverse=True)
    rows = [np.where(inv == i)[0] for i in range(len(pats))]
    v = [fn(np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])) for _ in range(n_boot)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def grading_by_quality(dev, quality, name):
    """Frozen grading model on dev images, split by a quality label array (0/1/2, -1 = unknown)."""
    out = []
    for k, n in enumerate(Q.NAMES):
        m = quality == k
        if m.sum() < 30:
            continue
        y, p = dev["y"][m], dev["probs5"][m]
        pred = p.argmax(1)
        rows = {"group": n, "n": int(m.sum()), "grade_accuracy": float((pred == y).mean()), "qwk": float(M.qwk(y, pred)),
                "far_errors": float((np.abs(pred - y) >= 2).mean())}
        yb = y >= 2
        s = p[:, 2:].sum(1)
        rows["auc_referable"] = float(M.binary_auc(yb.astype(int), s)) if 0 < yb.sum() < len(yb) else float("nan")
        rows["referable_sens"] = float((s[yb] >= 0.3269).mean()) if yb.any() else float("nan")
        rows["referable_spec"] = float((s[~yb] < 0.3269).mean()) if (~yb).any() else float("nan")
        out.append(rows)
    return {"by": name, "rows": out}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="kaggle/runs")
    ap.add_argument("--json", default="kaggle/runs/quality_eval.json")
    a = ap.parse_args(argv)

    seeds = {}
    for p in sorted(glob.glob(os.path.join(a.runs, "rg-quality-train", "runs", "quality_s*", "qtest_preds.npz"))):
        z = np.load(p)
        seeds[int(np.load(p)["y"].shape[0]) and int(os.path.basename(os.path.dirname(p)).split("_s")[1])] = {k: z[k] for k in z.files}
    d = seeds[DEPLOY_SEED]
    rep = {"n": int(len(d["y"])), "patients": int(len(np.unique(d["patient"]))), "deploy_seed": DEPLOY_SEED,
           "per_seed": {str(s): Q.report(v["y"], v["probs3"]) for s, v in seeds.items()}}
    main_rep = rep["per_seed"][str(DEPLOY_SEED)]
    y, pr = d["y"], d["probs3"]
    rep["ci"] = {"accuracy": boot(lambda ix: (pr[ix].argmax(1) == y[ix]).mean(), d["patient"]),
                 "macro_f1": boot(lambda ix: Q.macro_f1(y[ix], pr[ix].argmax(1)), d["patient"]),
                 "reject_recall": boot(lambda ix: (pr[ix][y[ix] == 2].argmax(1) == 2).mean() if (y[ix] == 2).any() else np.nan, d["patient"])}
    rep["seed_mean_std"] = {k: [float(np.mean([r[k] for r in rep["per_seed"].values()])), float(np.std([r[k] for r in rep["per_seed"].values()], ddof=1))]
                            for k in ("accuracy", "macro_f1", "reject_sensitivity", "reject_specificity")}
    rep["reject_tradeoff"] = Q.reject_tradeoff(y, pr)

    # grading accuracy by image quality, same 1,199 dev images (stored predictions of the frozen full model)
    gp = glob.glob(os.path.join(a.runs, "rg-final-model", "B_full_s42_384_mix", "eyepacs_dev_preds.npz"))
    z = np.load(gp[0])
    stems = {s: i for i, s in enumerate(z["stem"])}
    ix = np.array([stems[s] for s in d["stem"]])
    dev = {"y": z["y"].astype(int)[ix], "probs5": z["probs5"][ix]}
    rep["grading_by_quality"] = [grading_by_quality(dev, y, "EyeQ label (ground truth)"), grading_by_quality(dev, pr.argmax(1), "quality gate (predicted label)")]
    rep["grading_gate_vs_label"] = {"n": int(len(y)), "agreement": float((pr.argmax(1) == y).mean())}

    json.dump(rep, open(a.json, "w"), indent=1)
    print(f"deploy seed {DEPLOY_SEED}: acc {main_rep['accuracy']:.4f} {rep['ci']['accuracy']} | macro-F1 {main_rep['macro_f1']:.4f} {rep['ci']['macro_f1']} | Reject recall {main_rep['reject_sensitivity']:.3f} {rep['ci']['reject_recall']}")
    print("seeds mean ± std:", {k: f"{m:.3f} ± {s:.3f}" for k, (m, s) in rep["seed_mean_std"].items()})
    print("confusion (rows true Good/Usable/Reject):", np.array(main_rep["confusion"]).tolist())
    for n, v in main_rep["per_class"].items():
        print(f"  {n}: precision {v['precision']:.3f} recall {v['recall']:.3f} f1 {v['f1']:.3f} (n={v['support']})")
    print("Reject trade-off:")
    for r in rep["reject_tradeoff"]:
        print(f"  P(Reject) >= {r['threshold']:.1f}: stops {r['reject_stopped']:.3f} of Rejects, refuses {r['gradable_refused']:.3f} of gradable photos (Good {r['good_refused']:.3f}, Usable {r['usable_refused']:.3f})")
    for g in rep["grading_by_quality"]:
        print("grading by", g["by"])
        for r in g["rows"]:
            print(f"  {r['group']:8s} n={r['n']:4d} accuracy {r['grade_accuracy']:.3f} QWK {r['qwk']:.3f} far errors {r['far_errors']:.3f} referable AUC {r['auc_referable']:.3f} sens {r['referable_sens']:.3f} spec {r['referable_spec']:.3f}")


if __name__ == "__main__":
    main()
