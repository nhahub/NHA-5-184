"""
Headroom item 4 (local, CPU): domain gap APTOS -> EyePACS dev patients (prediction only).

    python src/evaluation/eyepacs_gap.py --preds kaggle/runs/rg-eyepacs-dev/eyepacs_dev_preds.npz

Input: kaggle/kernels/rg_eyepacs_dev.py output (B seed-42 fold models, float32, frozen preprocessing).
  * per model metrics on EyePACS dev, mean +/- std over the 5 fold models, and the 5-model average
  * the same images from two mirrors (hi-res ~1024px vs the pre-squeezed 224px mirror)
  * 384px pilot models vs the same two fold models at 224px (paired)
  * APTOS-derived referral thresholds (set at APTOS OOF specificity 0.95) applied unchanged
  * by EyeQ quality (Good / Usable / Reject) where available
CIs: patient-level bootstrap (both eyes of a patient are resampled together).
APTOS reference = B seed-42 OOF (same models' out-of-fold predictions on APTOS).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import metrics as M  # noqa: E402
from evaluation.operating_points import load_oof, oracle_sens_at_spec, thr_at_spec  # noqa: E402

QUAL = {0: "Good", 1: "Usable", 2: "Reject", -1: "no EyeQ label"}


def point(y, P):
    pred = P.argmax(1)
    out = {"qwk": M.qwk(y, pred), "acc": float((pred == y).mean())}
    for k in (1, 2, 3):
        yb, s = y >= k, P[:, k:].sum(1)
        out[f"auc{k}"] = M.binary_auc(yb.astype(int), s) if 0 < yb.sum() < len(yb) else np.nan
    for k in (2, 3):
        yb, s = y >= k, P[:, k:].sum(1)
        out[f"sens{k}@0.95"] = oracle_sens_at_spec(yb, s, 0.95)[0]
    return out


KEYS = ["qwk", "acc", "auc1", "auc2", "auc3", "sens2@0.95", "sens3@0.95"]


def patient_boot(y, patient, P_new, P_ref=None, n_boot=1000, seed=0):
    """CI of point(P_new) (or of point(P_new) - point(P_ref)), resampling patients."""
    rng = np.random.default_rng(seed)
    pats, inv = np.unique(patient, return_inverse=True)
    rows = [np.where(inv == i)[0] for i in range(len(pats))]
    vals = []
    for _ in range(n_boot):
        ix = np.concatenate([rows[i] for i in rng.integers(0, len(pats), len(pats))])
        a = point(y[ix], P_new[ix])
        b = point(y[ix], P_ref[ix]) if P_ref is not None else {k: 0.0 for k in KEYS}
        vals.append([a[k] - b[k] for k in KEYS])
    v = np.array(vals)
    return {k: (np.nanpercentile(v[:, j], 2.5), np.nanpercentile(v[:, j], 97.5)) for j, k in enumerate(KEYS)}


def fmt(m, ci=None):
    return " | ".join(f"{m[k]:.3f}" + (f" [{ci[k][0]:.3f}–{ci[k][1]:.3f}]" if ci else "") for k in KEYS)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="kaggle/runs")
    ap.add_argument("--preds", default="kaggle/runs/rg-eyepacs-dev/eyepacs_dev_preds.npz")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)

    z = dict(np.load(a.preds, allow_pickle=False))
    y, patient, quality = z["y"], z["patient"], z["quality"]
    models = {k: v for k, v in z.items() if "|" in k}
    ok = np.all([~np.isnan(v).any(1) for v in models.values()], 0)       # drop preprocessing failures
    y, patient, quality = y[ok], patient[ok], quality[ok]
    models = {k: v[ok] for k, v in models.items()}

    # APTOS reference: B seed-42 OOF (each image scored by the fold model that did not train on it)
    ap_ = load_oof(a.runs, "B", 42)
    print(f"### Domain gap: APTOS CV → EyePACS dev (n={len(y)} images, {len(np.unique(patient))} patients; "
          f"grades {np.bincount(y, minlength=5).tolist()}; dropped {int((~ok).sum())} unreadable)\n")
    print("| data / model | " + " | ".join(KEYS) + " |\n|---|" + "---|" * len(KEYS))
    print(f"| APTOS OOF, B s42 (reference) | {fmt(point(ap_['y'], ap_['probs5']))} |")

    for mirror in ("hires224", "lowres224"):
        per = [point(y, models[f"B_f{f}_s42_224|{mirror}"]) for f in range(5)]
        mean = {k: float(np.nanmean([m[k] for m in per])) for k in KEYS}
        std = {k: float(np.nanstd([m[k] for m in per], ddof=1)) for k in KEYS}
        print(f"| EyePACS {mirror}, single fold model (mean ± std of 5) | " +
              " | ".join(f"{mean[k]:.3f} ± {std[k]:.3f}" for k in KEYS) + " |")
        ens = np.mean([models[f"B_f{f}_s42_224|{mirror}"] for f in range(5)], 0)
        print(f"| EyePACS {mirror}, 5-model average [patient-bootstrap CI] | "
              f"{fmt(point(y, ens), patient_boot(y, patient, ens, n_boot=a.n_boot))} |")

    # mirror effect, paired (5-model averages)
    e_hi = np.mean([models[f"B_f{f}_s42_224|hires224"] for f in range(5)], 0)
    e_lo = np.mean([models[f"B_f{f}_s42_224|lowres224"] for f in range(5)], 0)
    d = patient_boot(y, patient, e_hi, e_lo, n_boot=a.n_boot)
    dm = {k: point(y, e_hi)[k] - point(y, e_lo)[k] for k in KEYS}
    print("\nMirror effect (hi-res − pre-squeezed mirror, same images, 5-model average): " +
          ", ".join(f"{k} {dm[k]:+.3f} [{d[k][0]:+.3f}, {d[k][1]:+.3f}]" for k in KEYS))

    # 384 pilot vs the same fold models at 224 (hi-res mirror), paired
    if all(f"B_f{f}_s42_384|hires384" in models for f in (0, 1)):
        p384 = np.mean([models[f"B_f{f}_s42_384|hires384"] for f in (0, 1)], 0)
        p224 = np.mean([models[f"B_f{f}_s42_224|hires224"] for f in (0, 1)], 0)
        d = patient_boot(y, patient, p384, p224, n_boot=a.n_boot)
        m384, m224 = point(y, p384), point(y, p224)
        print("\n384 vs 224 on EyePACS dev (fold 0+1 models averaged, hi-res mirror):\n")
        print("| model | " + " | ".join(KEYS) + " |\n|---|" + "---|" * len(KEYS))
        print(f"| 224 (f0+f1) | {fmt(m224)} |\n| 384 (f0+f1) | {fmt(m384)} |")
        print("| Δ 384 − 224 [CI] | " + " | ".join(f"{m384[k] - m224[k]:+.3f} [{d[k][0]:+.3f}, {d[k][1]:+.3f}]"
                                                for k in KEYS) + " |")

    # APTOS-tuned referral thresholds applied unchanged
    print("\nAPTOS thresholds (set at APTOS OOF specificity 0.95) applied to EyePACS dev (5-model average, hi-res):\n")
    print("| flag | threshold | APTOS sens/spec | EyePACS sens/spec |\n|---|---|---|---|")
    for k in (2, 3):
        s_ap, yb_ap = ap_["probs5"][:, k:].sum(1), ap_["y"] >= k
        t = thr_at_spec(s_ap[~yb_ap], 0.95)
        s, yb = e_hi[:, k:].sum(1), y >= k
        print(f"| grade ≥{k} | {t:.3f} | {(s_ap[yb_ap] >= t).mean():.3f}/{(s_ap[~yb_ap] < t).mean():.3f} | "
              f"{(s[yb] >= t).mean():.3f}/{(s[~yb] < t).mean():.3f} |")

    print("\nBy EyeQ quality (5-model average, hi-res mirror):\n")
    print("| quality | n | " + " | ".join(KEYS) + " |\n|---|---|" + "---|" * len(KEYS))
    for q in (0, 1, 2, -1):
        m = quality == q
        if m.sum() >= 30:
            print(f"| {QUAL[q]} | {int(m.sum())} | {fmt(point(y[m], e_hi[m]))} |")


if __name__ == "__main__":
    main()
