"""Metrics of the quality gate (Good 0 / Usable 1 / Reject 2). No TensorFlow."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import metrics as M  # noqa: E402

NAMES = ["Good", "Usable", "Reject"]


def confusion(y, pred):
    return M.confusion(y, pred, 3)


def per_class(y, pred):
    C = confusion(y, pred)
    out = {}
    for k, n in enumerate(NAMES):
        tp, fn, fp = C[k, k], C[k].sum() - C[k, k], C[:, k].sum() - C[k, k]
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[n] = {"precision": float(pr), "recall": float(rc), "f1": float(2 * pr * rc / (pr + rc)) if pr + rc else 0.0,
                  "support": int(C[k].sum())}
    return out


def macro_f1(y, pred):
    return float(np.mean([v["f1"] for v in per_class(y, pred).values()]))


def report(y, probs3) -> dict:
    y, p = np.asarray(y), np.asarray(probs3)
    pred = p.argmax(1)
    rej = y == 2
    return {"n": int(len(y)), "accuracy": float((pred == y).mean()), "macro_f1": macro_f1(y, pred),
            "kappa_quadratic": float(M.qwk(y, pred, n_classes=3)), "per_class": per_class(y, pred),
            "confusion": confusion(y, pred).tolist(),
            "reject_auc": float(M.binary_auc(rej.astype(int), p[:, 2])),
            "reject_sensitivity": float((pred[rej] == 2).mean()), "reject_specificity": float((pred[~rej] != 2).mean()),
            "good_called_reject": float((pred[y == 0] == 2).mean()), "reject_called_good": float((pred[rej] == 0).mean())}


def reject_tradeoff(y, probs3, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> list:
    """Reject when P(Reject) >= t: how many bad photos are stopped, how many gradable ones are refused."""
    y, r = np.asarray(y), np.asarray(probs3)[:, 2]
    return [{"threshold": t, "reject_stopped": float((r[y == 2] >= t).mean()), "gradable_refused": float((r[y < 2] >= t).mean()),
             "good_refused": float((r[y == 0] >= t).mean()), "usable_refused": float((r[y == 1] >= t).mean())} for t in thresholds]
