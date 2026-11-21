"""
RetinoGrade AI - evaluation metrics (single source of truth for every reported number).

Pure numpy (+ sklearn only for ROC-AUC) so it runs locally on Python 3.14 without TensorFlow
and identically inside Kaggle kernels.

Grades: 0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative DR.
  referable          = grade >= 2
  sight-threatening  = grade >= 3
"""
from __future__ import annotations

import numpy as np

N_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


# ----------------------------------------------------------------------------- basics
def _as_int(a) -> np.ndarray:
    return np.asarray(a).astype(np.int64).ravel()


def confusion(y_true, y_pred, n_classes: int = N_CLASSES) -> np.ndarray:
    """Rows = true, cols = predicted."""
    t, p = _as_int(y_true), _as_int(y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (t, p), 1)
    return cm


def qwk(y_true, y_pred, n_classes: int = N_CLASSES) -> float:
    """Quadratic weighted kappa (identical to sklearn cohen_kappa_score(weights='quadratic')
    when both label sets are within 0..n_classes-1)."""
    cm = confusion(y_true, y_pred, n_classes).astype(np.float64)
    n = cm.sum()
    if n == 0:
        return float("nan")
    i = np.arange(n_classes)
    w = (i[:, None] - i[None, :]) ** 2
    expected = np.outer(cm.sum(1), cm.sum(0)) / n
    denom = (w * expected).sum()
    if denom == 0:
        return 1.0 if (w * cm).sum() == 0 else 0.0
    return float(1.0 - (w * cm).sum() / denom)


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(_as_int(y_true) == _as_int(y_pred)))


def within1(y_true, y_pred) -> float:
    return float(np.mean(np.abs(_as_int(y_true) - _as_int(y_pred)) <= 1))


def per_class_pr(y_true, y_pred, n_classes: int = N_CLASSES) -> dict:
    cm = confusion(y_true, y_pred, n_classes)
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    precision = cm.diagonal() / np.maximum(cm.sum(0), 1)
    return {"recall": recall.tolist(), "precision": precision.tolist(),
            "support": cm.sum(1).tolist()}


def referable_metrics(y_true, y_pred, thr: int = 2):
    """Sensitivity / specificity of 'grade >= thr' (thr=2 referable, thr=3 sight-threatening).
    Port of referable_metrics() from stage2_severity_cascade.py (Cell 9)."""
    t, p = _as_int(y_true) >= thr, _as_int(y_pred) >= thr
    sens = (t & p).sum() / max(t.sum(), 1)
    spec = (~t & ~p).sum() / max((~t).sum(), 1)
    return float(sens), float(spec)


def binary_auc(y_bin, score) -> float:
    y_bin = np.asarray(y_bin).astype(int).ravel()
    if y_bin.min() == y_bin.max():
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_bin, np.asarray(score, dtype=np.float64).ravel()))


# ----------------------------------------------------------------------------- cascade / ordinal helpers
def cascade_predict(p1, p2, threshold: float = 0.3, prior=None, alpha: float = 0.0) -> np.ndarray:
    """v1 decision rule: grade 0 if p1 < threshold else argmax(p2 * prior**alpha) + 1."""
    p1 = np.asarray(p1, dtype=np.float64).ravel()
    p2 = np.asarray(p2, dtype=np.float64)
    if prior is not None and alpha:
        p2 = p2 * (np.asarray(prior, dtype=np.float64) ** alpha)
    return np.where(p1 >= threshold, p2.argmax(1) + 1, 0).astype(np.int64)


def cascade_probs5(p1, p2) -> np.ndarray:
    """Joint 5-class distribution from a cascade: [1-p1, p1*p2_1..4]."""
    p1 = np.asarray(p1, dtype=np.float64).reshape(-1, 1)
    return np.hstack([1.0 - p1, p1 * np.asarray(p2, dtype=np.float64)])


def expected_grade(probs5) -> np.ndarray:
    probs5 = np.asarray(probs5, dtype=np.float64)
    return probs5 @ np.arange(probs5.shape[1], dtype=np.float64)


DEFAULT_CUTS = (0.5, 1.5, 2.5, 3.5)


def apply_thresholds(score, cuts=DEFAULT_CUTS) -> np.ndarray:
    """Continuous ordinal score -> grade 0..4."""
    return np.searchsorted(np.sort(np.asarray(cuts, dtype=np.float64)),
                           np.asarray(score, dtype=np.float64).ravel(), side="right").astype(np.int64)


def fit_thresholds(y_true, score, init=DEFAULT_CUTS, n_rounds: int = 4, grid: int = 61) -> np.ndarray:
    """Coordinate-ascent search of 4 cut points maximizing QWK.
    Fit on OOF predictions only - never on the test split."""
    y, s = _as_int(y_true), np.asarray(score, dtype=np.float64).ravel()
    cuts = np.array(init, dtype=np.float64)
    best = qwk(y, apply_thresholds(s, cuts))
    for _ in range(n_rounds):
        improved = False
        for k in range(len(cuts)):
            lo = cuts[k - 1] + 1e-3 if k > 0 else cuts[k] - 1.0
            hi = cuts[k + 1] - 1e-3 if k < len(cuts) - 1 else cuts[k] + 1.0
            for c in np.linspace(lo, hi, grid):
                trial = cuts.copy()
                trial[k] = c
                q = qwk(y, apply_thresholds(s, trial))
                if q > best + 1e-9:
                    best, cuts, improved = q, trial, True
        if not improved:
            break
    return cuts


# ----------------------------------------------------------------------------- full report + CIs
def point_metrics(y_true, y_pred, score_dr=None, score_ref=None) -> dict:
    """All headline numbers for one prediction set.
    score_dr  : continuous score for 'any DR'    (p1, or 1 - P(grade 0))        -> any_dr_auc
    score_ref : continuous score for 'referable' (P(grade>=2), expected grade)  -> referable_auc"""
    y, p = _as_int(y_true), _as_int(y_pred)
    ref_s, ref_sp = referable_metrics(y, p, 2)
    st_s, st_sp = referable_metrics(y, p, 3)
    dr_s, dr_sp = referable_metrics(y, p, 1)
    out = {
        "n": int(len(y)),
        "qwk": qwk(y, p),
        "accuracy": accuracy(y, p),
        "within1": within1(y, p),
        "any_dr_sens": dr_s, "any_dr_spec": dr_sp,
        "referable_sens": ref_s, "referable_spec": ref_sp,
        "sight_threat_sens": st_s, "sight_threat_spec": st_sp,
    }
    if score_dr is not None:
        out["any_dr_auc"] = binary_auc(y >= 1, score_dr)
    if score_ref is not None:
        out["referable_auc"] = binary_auc(y >= 2, score_ref)
    return out


BOOT_KEYS = ("qwk", "accuracy", "referable_sens", "referable_spec",
             "sight_threat_sens", "sight_threat_spec", "any_dr_auc", "referable_auc")


def bootstrap_ci(y_true, y_pred, score_dr=None, score_ref=None, n_boot: int = 1000,
                 seed: int = 0, level: float = 0.95, keys=BOOT_KEYS) -> dict:
    """Percentile bootstrap CIs (images resampled with replacement)."""
    y, p = _as_int(y_true), _as_int(y_pred)
    sd = None if score_dr is None else np.asarray(score_dr, dtype=np.float64).ravel()
    sr = None if score_ref is None else np.asarray(score_ref, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in keys}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = point_metrics(y[idx], p[idx],
                          None if sd is None else sd[idx],
                          None if sr is None else sr[idx])
        for k in keys:
            if k in m:
                samples[k].append(m[k])
    lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out = {}
    for k, v in samples.items():
        v = np.asarray(v, dtype=np.float64)
        v = v[~np.isnan(v)]
        if len(v):
            out[k] = [float(np.percentile(v, lo_q)), float(np.percentile(v, hi_q))]
    return out


def full_report(y_true, y_pred, score_dr=None, score_ref=None, n_boot: int = 1000,
                seed: int = 0) -> dict:
    rep = point_metrics(y_true, y_pred, score_dr, score_ref)
    rep["per_class"] = per_class_pr(y_true, y_pred)
    rep["confusion"] = confusion(y_true, y_pred).tolist()
    if n_boot:
        rep["ci95"] = bootstrap_ci(y_true, y_pred, score_dr, score_ref, n_boot=n_boot, seed=seed)
    return rep


def mean_std(reports, keys=None) -> dict:
    """Aggregate scalar metrics over seeds/folds -> {key: [mean, std(ddof=1), n]}."""
    reports = list(reports)
    if not reports:
        return {}
    keys = keys or [k for k, v in reports[0].items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "n"]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in reports if k in r], dtype=np.float64)
        v = v[~np.isnan(v)]
        if len(v):
            out[k] = [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, int(len(v))]
    return out
