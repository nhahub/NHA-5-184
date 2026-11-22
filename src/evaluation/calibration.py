"""
Step 4: confidence calibration, the sight-threatening alarm, and the "needs human review" rule.
Local, CPU, predictions only. NOT a test split: the fitted data are the APTOS out-of-fold (OOF) CV
predictions, and EyePACS DEV patients are used as an external check (dev is where the frozen
thresholds were set, never test data).

    python src/evaluation/calibration.py --runs kaggle/runs

Data
  fit   : APTOS OOF, mixed 384px, seed 42, 5 folds (each image scored by the fold model that did not train on it)
  check : EyePACS dev, the deployed full-pool model's own predictions (kaggle/runs/rg-final-model)

Temperature scaling: p_T(k) is proportional to p(k)^(1/T). It never changes the predicted grade.
The referral flags use the raw probabilities (frozen thresholds): calibration only changes the displayed confidence.
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
from evaluation.operating_points import load_oof  # noqa: E402

EPS = 1e-8
N_BINS = 15
ST_FROZEN, REF_FROZEN = 0.0277, 0.3269


# ----------------------------------------------------------------------------- calibration
def apply_temperature(p, T):
    q = np.clip(np.asarray(p, np.float64), EPS, 1) ** (1.0 / T)
    return q / q.sum(axis=-1, keepdims=True)


def nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))))


def fit_temperature(p, y, grid=np.exp(np.linspace(np.log(0.3), np.log(6), 400))):
    losses = [nll(apply_temperature(p, t), y) for t in grid]
    i = int(np.argmin(losses))
    lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    fine = np.linspace(lo, hi, 201)
    return float(fine[int(np.argmin([nll(apply_temperature(p, t), y) for t in fine]))])


def ece(p, y, bins=N_BINS, adaptive=False):
    conf, pred = p.max(1), p.argmax(1)
    ok = (pred == y).astype(float)
    edges = np.quantile(conf, np.linspace(0, 1, bins + 1)) if adaptive else np.linspace(0, 1, bins + 1)
    edges = np.unique(edges)
    out = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b) if a > edges[0] else (conf >= a) & (conf <= b)
        if m.any():
            out += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(out)


def reliability_points(p, y, bins=10):
    conf, ok = p.max(1), (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    return [(float(conf[m].mean()), float(ok[m].mean()), int(m.sum()))
            for a, b in zip(edges[:-1], edges[1:]) if (m := (conf > a) & (conf <= b)).any()]


def brier(p, y):
    return float(np.mean(((p - np.eye(p.shape[1])[y]) ** 2).sum(1)))


def crossfit_temperature(p, y, fold):
    """Probabilities calibrated with the temperature of the OTHER folds."""
    out, Ts = np.zeros_like(p), {}
    for f in np.unique(fold):
        T = fit_temperature(p[fold != f], y[fold != f])
        out[fold == f], Ts[int(f)] = apply_temperature(p[fold == f], T), T
    return out, Ts


def crossfit_isotonic(p, y, fold):
    """Isotonic map of the top-class confidence (fitted on the other folds); the predicted grade is unchanged."""
    from sklearn.isotonic import IsotonicRegression
    conf, ok, out = p.max(1), (p.argmax(1) == y).astype(float), np.zeros(len(y))
    for f in np.unique(fold):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(conf[fold != f], ok[fold != f])
        out[fold == f] = ir.predict(conf[fold == f])
    return out


def boot_ci(fn, n, n_boot=1000, seed=0, groups=None):
    rng = np.random.default_rng(seed)
    if groups is None:
        rows = [np.array([i]) for i in range(n)]
    else:
        _, inv = np.unique(groups, return_inverse=True)
        rows = [np.where(inv == g)[0] for g in range(inv.max() + 1)]
    v = [fn(np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])) for _ in range(n_boot)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ----------------------------------------------------------------------------- referral alarms
def alarm_row(y, score, t, k):
    yb, pos = y >= k, score >= t
    tp, fp = int((pos & yb).sum()), int((pos & ~yb).sum())
    return {"threshold": float(t), "sensitivity": tp / max(1, int(yb.sum())), "specificity": float((~pos[~yb]).mean()),
            "ppv": tp / max(1, int(pos.sum())), "flagged_pct": float(pos.mean()), "false_alarms_per_1000": 1000 * fp / len(y),
            "missed": int(yb.sum()) - tp, "positives": int(yb.sum())}


def threshold_at_spec(score_neg, spec):
    s = np.sort(score_neg)
    return float(np.nextafter(s[min(int(np.ceil(spec * len(s))) - 1, len(s) - 1)], np.inf))


# ----------------------------------------------------------------------------- "needs human review"
def review_scores(p, p_ref_threshold=REF_FROZEN):
    """Higher = more reason to ask a human. Each is a simple, explainable quantity."""
    conf = np.sort(p, axis=1)
    ent = -(np.clip(p, EPS, 1) * np.log(np.clip(p, EPS, 1))).sum(1) / np.log(p.shape[1])
    ref = p[:, 2:].sum(1)
    return {"low top-class probability": 1 - conf[:, -1], "small margin (top1 - top2)": 1 - (conf[:, -1] - conf[:, -2]),
            "entropy": ent, "near the referral boundary": -np.abs(ref - p_ref_threshold)}


def coverage_table(score, y, p, p_ref_threshold=REF_FROZEN, st_threshold=ST_FROZEN, coverages=(1.0, 0.9, 0.8, 0.7)):
    """Send the (1 - coverage) share with the highest review score to a human; what is left is automatic."""
    pred, ref, st = p.argmax(1), p[:, 2:].sum(1) >= p_ref_threshold, p[:, 3:].sum(1) >= st_threshold
    order, rows = np.argsort(score, kind="stable"), []
    for c in coverages:
        auto = np.zeros(len(y), bool)
        auto[order[: int(round(c * len(y)))]] = True
        a = auto
        rows.append({"coverage": c, "n_review": int((~a).sum()),
                     "grade_error_auto": float((pred[a] != y[a]).mean()),
                     "far_error_auto": float((np.abs(pred[a] - y[a]) >= 2).mean()),
                     "referable_total": int((y >= 2).sum()),
                     "referable_missed_auto": int(((y >= 2) & ~ref & a).sum()), "referable_caught_by_review": int(((y >= 2) & ~ref & ~a).sum()),
                     "st_total": int((y >= 3).sum()), "st_missed_auto": int(((y >= 3) & ~st & a).sum()),
                     "st_caught_by_review": int(((y >= 3) & ~st & ~a).sum()),
                     "errors_caught_by_review": float((pred != y)[~a].sum() / max(1, (pred != y).sum()))})
    return rows


# ----------------------------------------------------------------------------- data + report
def load_data(runs):
    a = load_oof(runs, "B", 42, suffix="_mix", img=384)
    z = np.load(glob.glob(os.path.join(runs, "rg-final-model", "B_full_s42_384_mix", "eyepacs_dev_preds.npz"))[0])
    return {"y": a["y"], "p": a["probs5"], "fold": a["fold"]}, \
           {"y": z["y"].astype(int), "p": z["probs5"].astype(np.float64), "patient": z["patient"], "stem": z["stem"]}


def write_figures(rep, folder):
    """Reliability plots + Markdown tables (docs/figures/calibration). Contains no images of patients."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(folder, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, ds, title in ((axes[0], "oof", "APTOS out-of-fold"), (axes[1], "dev", "EyePACS dev")):
        ax.plot([0, 1], [0, 1], "--", color="0.6", label="perfect")
        for key, lab, col in (("raw", "before", "#c62828"), ("cal", "after temperature", "#1565c0")):
            pts = rep["reliability"][f"{ds}_{key}"]
            ax.plot([x for x, _, _ in pts], [y for _, y, _ in pts], "o-", color=col, label=lab)
        ax.set(xlabel="confidence", ylabel="accuracy", title=title, xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left")
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "reliability.png"), dpi=110)
    plt.close(fig)

    L = ["# Calibration and review tables (generated by src/evaluation/calibration.py)", ""]
    t = rep["temperature"]
    L += [f"Temperature fitted on APTOS OOF: **{t['T']:.3f}** (per fold {', '.join(f'{v:.2f}' for v in t['T_per_fold'].values())}); fitted on EyePACS dev it would be {t['T_if_fitted_on_dev']:.3f}.", "",
          "## ECE", "", "| data | method | ECE | 95% CI | NLL | Brier |", "|---|---|---|---|---|---|"]
    for name, r in rep["oof"].items():
        L.append(f"| APTOS OOF (cross-fitted where fitted) | {name} | {r['ece']:.3f} | " + (f"{r['ece_ci'][0]:.3f}–{r['ece_ci'][1]:.3f}" if "ece_ci" in r else "–") + f" | {r.get('nll', float('nan')):.3f} | {r.get('brier', float('nan')):.3f} |")
    for name, r in rep["dev"].items():
        L.append(f"| EyePACS dev | {name} | {r['ece']:.3f} | – | {r.get('nll', float('nan')):.3f} | {r.get('brier', float('nan')):.3f} |")
    L += ["", "## Alarms", ""]
    for key, rows in rep["alarms"].items():
        ds, kname, pname = key.split(" | ")
        if pname != "raw":
            continue
        L += [f"**{ds}: {kname} (raw probabilities)**", "", "| operating point | threshold | sensitivity | specificity | PPV | false alarms / 1000 screened | missed |", "|---|---|---|---|---|---|---|"]
        for n, r in rows.items():
            L.append(f"| {n} | {r['threshold']:.4f} | {r['sensitivity']:.3f} | {r['specificity']:.3f} | {r['ppv']:.3f} | {r['false_alarms_per_1000']:.0f} | {r['missed']}/{r['positives']} |")
        L.append("")
    L += ["## Review rule candidates (share of decisions kept automatic)", ""]
    for ds, rules in rep["review"].items():
        for r, rows in rules.items():
            L += [f"**{ds}: {r}**", "", "| automatic share | grade error | far errors | missed referable left automatic | caught by review |", "|---|---|---|---|---|"]
            for x in rows:
                L.append(f"| {x['coverage']:.1f} | {x['grade_error_auto']:.3f} | {x['far_error_auto']:.3f} | {x['referable_missed_auto']}/{x['referable_total']} | {x['referable_caught_by_review']} |")
            L.append("")
    with open(os.path.join(folder, "tables.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="kaggle/runs")
    ap.add_argument("--json", default="kaggle/runs/calibration.json")
    ap.add_argument("--figures", help="write reliability plots and Markdown tables into this folder")
    a = ap.parse_args(argv)
    oof, dev = load_data(a.runs)
    rep = {"n_oof": int(len(oof["y"])), "n_dev": int(len(dev["y"]))}

    # 1. temperature
    T = fit_temperature(oof["p"], oof["y"])
    cal, Ts = crossfit_temperature(oof["p"], oof["y"], oof["fold"])
    iso = crossfit_isotonic(oof["p"], oof["y"], oof["fold"])
    rep["temperature"] = {"fitted_on": "APTOS OOF", "T": T, "T_per_fold": Ts,
                          "T_if_fitted_on_dev": fit_temperature(dev["p"], dev["y"])}
    tab = {}
    n = len(oof["y"])
    for name, p in (("raw", oof["p"]), ("temperature (cross-fitted)", cal)):
        tab[name] = {"ece": ece(p, oof["y"]), "ece_adaptive": ece(p, oof["y"], adaptive=True), "nll": nll(p, oof["y"]), "brier": brier(p, oof["y"])}
        tab[name]["ece_ci"] = boot_ci(lambda ix, p=p: ece(p[ix], oof["y"][ix]), n)
    ok = (oof["p"].argmax(1) == oof["y"]).astype(float)
    conf_iso = iso
    tab["isotonic on confidence (cross-fitted)"] = {"ece": float(np.sum([np.abs(ok[m].mean() - conf_iso[m].mean()) * m.mean()
        for a_, b_ in zip(np.linspace(0, 1, N_BINS + 1)[:-1], np.linspace(0, 1, N_BINS + 1)[1:]) if (m := (conf_iso > a_) & (conf_iso <= b_)).any()]))}
    rep["oof"] = tab
    dev_cal = apply_temperature(dev["p"], T)
    rep["dev"] = {"raw": {"ece": ece(dev["p"], dev["y"]), "nll": nll(dev["p"], dev["y"]), "brier": brier(dev["p"], dev["y"])},
                  "temperature from APTOS OOF": {"ece": ece(dev_cal, dev["y"]), "nll": nll(dev_cal, dev["y"]), "brier": brier(dev_cal, dev["y"])},
                  "temperature fitted on dev (optimistic)": {"ece": ece(apply_temperature(dev["p"], rep["temperature"]["T_if_fitted_on_dev"]), dev["y"])}}
    rep["reliability"] = {"oof_raw": reliability_points(oof["p"], oof["y"]), "oof_cal": reliability_points(cal, oof["y"]),
                          "dev_raw": reliability_points(dev["p"], dev["y"]), "dev_cal": reliability_points(dev_cal, dev["y"])}

    # 2. alarms: sight-threatening (grade >= 3) and referable (>= 2) at several thresholds, raw and calibrated probabilities
    rep["alarms"] = {}
    for dsname, d, cali in (("APTOS OOF", oof, cal), ("EyePACS dev", dev, dev_cal)):
        for kname, k, frozen in (("sight-threatening", 3, ST_FROZEN), ("referable", 2, REF_FROZEN)):
            for pname, p in (("raw", d["p"]), ("calibrated", cali)):
                s = p[:, k:].sum(1)
                neg = s[d["y"] < k]
                ts = [("frozen", frozen), ("spec 0.90", threshold_at_spec(neg, 0.90)), ("spec 0.95", threshold_at_spec(neg, 0.95)),
                      ("spec 0.975", threshold_at_spec(neg, 0.975)), ("0.10", 0.10), ("0.20", 0.20), ("0.30", 0.30), ("0.50", 0.50)]
                rep["alarms"][f"{dsname} | {kname} | {pname}"] = {n_: alarm_row(d["y"], s, t, k) for n_, t in ts}

    # 3. review rule on OOF, checked on dev
    rep["review"] = {}
    for dsname, d in (("APTOS OOF", oof), ("EyePACS dev", dev)):
        rep["review"][dsname] = {r: coverage_table(s, d["y"], d["p"]) for r, s in review_scores(d["p"]).items()}
    json.dump(rep, open(a.json, "w"), indent=1)
    if a.figures:
        write_figures(rep, a.figures)
    print(json.dumps({k: rep[k] for k in ("n_oof", "n_dev", "temperature", "oof", "dev")}, indent=1))
    for ds in ("APTOS OOF", "EyePACS dev"):
        print(f"\n### {ds}: sight-threatening alarm (raw probabilities)")
        for n_, r in rep["alarms"][f"{ds} | sight-threatening | raw"].items():
            print(f"  {n_:10s} t={r['threshold']:.4f} sens {r['sensitivity']:.3f} spec {r['specificity']:.3f} PPV {r['ppv']:.3f} "
                  f"false alarms/1000 {r['false_alarms_per_1000']:.0f} missed {r['missed']}/{r['positives']}")
        print(f"\n### {ds}: review rule coverage")
        for r, rows in rep["review"][ds].items():
            print(f"  {r}: " + " | ".join(f"cov {x['coverage']:.1f}: err {x['grade_error_auto']:.3f} missed ref {x['referable_missed_auto']}/{x['referable_total']}" for x in rows))


if __name__ == "__main__":
    main()
