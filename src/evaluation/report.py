"""
Aggregate downloaded Kaggle results into docs/experiments.md (runs locally, CPU, seconds).

    python src/evaluation/report.py --runs kaggle/runs --doc docs/experiments.md

Inputs (small files pulled with `kaggle kernels output`):
  <runs>/**/run.json        one per (candidate, fold, seed)        -> per-run table
  <runs>/**/oof_preds.npz   fold-val predictions                    -> stitched OOF per (candidate, seed)
  <runs>/**/v1_repro.json   optional v1 reproduction                -> v1 row

OOF rules:
  * a (candidate, seed, img, tta, quick) group is reported only when all 5 folds are present;
  * decision cut points (C: 4 grade cuts, A: Stage-1 threshold) are fitted with *cross-fitting*
    (fit on 4 folds' OOF, apply to the 5th) so the reported OOF number is not tuned on itself;
  * seeds are then aggregated as mean +/- std.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import metrics as M  # noqa: E402

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
N_FOLDS = 5
A_THRESHOLDS = np.round(np.arange(0.10, 0.71, 0.05), 2)


# ----------------------------------------------------------------------------- loading
def load_runs(runs_dir: str):
    runs = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "**", "run.json"), recursive=True)):
        with open(p) as f:
            r = json.load(f)
        r["_dir"] = os.path.dirname(p)
        runs.append(r)
    return runs


def load_oof_groups(runs):
    """-> {(cand, seed, img, tta, quick): {fold: npz-dict}}"""
    groups = defaultdict(dict)
    for r in runs:
        p = os.path.join(r["_dir"], "oof_preds.npz")
        if os.path.exists(p):
            key = (r["candidate"], r["seed"], r["img_size"], bool(r.get("tta")), bool(r["quick_test"]))
            if r.get("config", {}).get("train_fraction", 1.0) < 1.0:   # learning-curve runs: own group
                key = key[:3] + (f"p{round(r['config']['train_fraction'] * 100)}",) + key[4:]
            if r.get("config", {}).get("extra_data"):                  # mixed APTOS+EyePACS runs: own group
                key = key[:3] + (f"+{r['config']['extra_data']}",) + key[4:]
            with np.load(p, allow_pickle=False) as z:
                groups[key][int(r["fold"])] = {k: z[k] for k in z.files}
    return groups


# ----------------------------------------------------------------------------- OOF with cross-fitted cut points
def _stack(folds: dict, key: str):
    return np.concatenate([folds[f][key] for f in sorted(folds)])


def cross_fit_predictions(candidate: str, folds: dict):
    """Return (y, pred, score_dr, score_ref, fold_ids, info) with cut points fitted on the other folds."""
    order = sorted(folds)
    fold_ids = np.concatenate([np.full(len(folds[f]["y_true"]), f) for f in order])
    y = _stack(folds, "y_true")
    score_dr, score_ref = _stack(folds, "score_dr"), _stack(folds, "score_ref")
    pred = _stack(folds, "pred").copy()
    info = {}
    if candidate == "C":
        s = _stack(folds, "score")
        cuts_per_fold = {}
        for f in order:
            other = fold_ids != f
            cuts = M.fit_thresholds(y[other], s[other])
            pred[fold_ids == f] = M.apply_thresholds(s[fold_ids == f], cuts)
            cuts_per_fold[int(f)] = [round(float(c), 3) for c in cuts]
        info["cuts_per_fold"] = cuts_per_fold
        info["cuts_all_oof"] = [round(float(c), 3) for c in M.fit_thresholds(y, s)]   # for the final model
    elif candidate == "A":
        p1, p2 = _stack(folds, "p1"), _stack(folds, "p2")
        thr_per_fold = {}
        for f in order:
            other = fold_ids != f
            best = max(A_THRESHOLDS, key=lambda t: M.qwk(y[other], M.cascade_predict(p1[other], p2[other], t)))
            pred[fold_ids == f] = M.cascade_predict(p1[fold_ids == f], p2[fold_ids == f], best)
            thr_per_fold[int(f)] = float(best)
        info["threshold_per_fold"] = thr_per_fold
    return y, pred, score_dr, score_ref, fold_ids, info


def oof_summary(groups, n_boot: int = 1000):
    rows = []
    for key, folds in sorted(groups.items()):
        cand, seed, img, tta, quick = key
        if sorted(folds) != list(range(N_FOLDS)):
            rows.append({"key": key, "incomplete": sorted(folds)})
            continue
        y, pred, sd, sr, _, info = cross_fit_predictions(cand, folds)
        default = M.point_metrics(y, _stack(folds, "pred"), sd, sr)
        rep = M.full_report(y, pred, sd, sr, n_boot=n_boot)
        rows.append({"key": key, "report": rep, "default_qwk": default["qwk"], "info": info})
    return rows


def seed_aggregate(oof_rows):
    by_cfg = defaultdict(list)
    for r in oof_rows:
        if "report" in r:
            cand, seed, img, tta, quick = r["key"]
            by_cfg[(cand, img, tta, quick)].append(r["report"])
    return {k: M.mean_std(v) for k, v in by_cfg.items()}


# ----------------------------------------------------------------------------- markdown
def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _ci(rep, k):
    ci = rep.get("ci95", {}).get(k)
    return f"{_f(rep.get(k))} [{_f(ci[0])}–{_f(ci[1])}]" if ci else _f(rep.get(k))


def _label(cand, img, tta, quick):
    extra = f" {tta} data" if isinstance(tta, str) else (" +TTA" if tta else "")
    return f"{cand} {img}px" + extra + (" (QUICK_TEST)" if quick else "")


def render(runs, oof_rows, agg, v1=None) -> str:
    L = [START, "", "_Generated by `src/evaluation/report.py` — do not edit by hand._", ""]
    if v1:
        m = v1["metrics"]
        L += ["### v1 reproduction (old APTOS val, 733 imgs — used for v1 selection, optimistic)", "",
              "| model | thr | QWK [95% CI] | acc | ref. sens/spec | sight-thr. sens/spec |",
              "|---|---|---|---|---|---|",
              f"| v1 cascade | {v1.get('threshold')} | {_ci(m, 'qwk')} | {_f(m['accuracy'])} | "
              f"{_f(m['referable_sens'])}/{_f(m['referable_spec'])} | "
              f"{_f(m['sight_threat_sens'])}/{_f(m['sight_threat_spec'])} |", ""]
    if agg:
        L += ["### Candidates — 5-fold OOF, mean ± std over seeds (model-selection numbers)", "",
              "| config | seeds | QWK | acc | ref. sens | ref. spec | sight-thr. sens | sight-thr. spec |",
              "|---|---|---|---|---|---|---|---|"]
        for (cand, img, tta, quick), a in sorted(agg.items()):
            def ms(k):
                return f"{a[k][0]:.3f} ± {a[k][1]:.3f}" if k in a else "—"
            L.append(f"| {_label(cand, img, tta, quick)} | {a['qwk'][2]} | {ms('qwk')} | {ms('accuracy')} | "
                     f"{ms('referable_sens')} | {ms('referable_spec')} | {ms('sight_threat_sens')} | "
                     f"{ms('sight_threat_spec')} |")
        L.append("")
    if oof_rows:
        L += ["### OOF per seed (cut points cross-fitted; 95% bootstrap CI)", "",
              "| config | seed | QWK [CI] | QWK default cuts | ref. sens [CI] | sight-thr. sens [CI] | cut points |",
              "|---|---|---|---|---|---|---|"]
        for r in oof_rows:
            cand, seed, img, tta, quick = r["key"]
            if "incomplete" in r:
                L.append(f"| {_label(cand, img, tta, quick)} | {seed} | folds present: {r['incomplete']} | | | | |")
                continue
            rep = r["report"]
            cuts = r["info"].get("cuts_all_oof") or r["info"].get("threshold_per_fold") or ""
            L.append(f"| {_label(cand, img, tta, quick)} | {seed} | {_ci(rep, 'qwk')} | {_f(r['default_qwk'])} | "
                     f"{_ci(rep, 'referable_sens')} | {_ci(rep, 'sight_threat_sens')} | {cuts} |")
        L.append("")
    if runs:
        L += ["### Individual runs (fold-val)", "",
              "| run_id | QWK | ref. sens/spec | sight-thr. sens/spec | n train/val | min | code |",
              "|---|---|---|---|---|---|---|"]
        for r in sorted(runs, key=lambda r: r["run_id"]):
            m = r["metrics"]
            L.append(f"| {r['run_id']} | {_f(m['qwk'])} | {_f(m['referable_sens'])}/{_f(m['referable_spec'])} | "
                     f"{_f(m['sight_threat_sens'])}/{_f(m['sight_threat_spec'])} | "
                     f"{r['data']['n_train']}/{r['data']['n_val']} | {r['duration_s'] / 60:.1f} | "
                     f"{r.get('code_version', '?')[:10]} |")
        L.append("")
    L.append(END)
    return "\n".join(L)


def write_doc(doc_path: str, block: str) -> None:
    text = open(doc_path, encoding="utf-8").read() if os.path.exists(doc_path) else ""
    if START in text and END in text:
        text = text[:text.index(START)] + block + text[text.index(END) + len(END):]
    else:
        text = text.rstrip() + "\n\n## Results\n\n" + block + "\n"
    with open(doc_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="kaggle/runs")
    ap.add_argument("--doc", default="docs/experiments.md")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true", help="print instead of writing the doc")
    ap.add_argument("--include-quick", action="store_true", help="also show QUICK_TEST smoke runs")
    a = ap.parse_args(argv)

    runs = [r for r in load_runs(a.runs) if a.include_quick or not r.get("quick_test")]
    v1_files = glob.glob(os.path.join(a.runs, "**", "v1_repro.json"), recursive=True)
    v1 = json.load(open(sorted(v1_files)[-1])) if v1_files else None
    oof_rows = oof_summary(load_oof_groups(runs), n_boot=a.n_boot)
    block = render(runs, oof_rows, seed_aggregate(oof_rows), v1)
    if a.dry_run:
        print(block)
    else:
        write_doc(a.doc, block)
        print(f"updated {a.doc}: {len(runs)} runs, {len(oof_rows)} OOF groups, v1={'yes' if v1 else 'no'}")


if __name__ == "__main__":
    main()
