"""
Calibrated confidence and the "needs human review" rule (pure functions, no TensorFlow).

    from review import assess
    a = assess(result["probabilities"], quality_label=result["quality"]["label"] if result["quality"] else None)
    a["needs_review"], a["reasons"], a["confidence_calibrated"]

PROPOSED VALUES, NOT APPROVED. They were chosen on APTOS cross-validation predictions only
(docs/components/calibration_review.md); the team still has to choose.

Review rule (a person should look if ANY holds):
  1. top-class probability < MIN_TOP_PROBABILITY        the model is unsure about the grade
  2. |P(grade >= 2) - referral threshold| < REFERRAL_MARGIN   the referral decision is borderline
  3. image quality is "Usable"                           the photo is limited

Calibration: temperature scaling, p_T(k) proportional to p(k)^(1/T). It changes only the displayed confidence;
the grade, the flags and the review rule always use the raw probabilities.
"""
from __future__ import annotations

import os

import numpy as np

from retinograde_inference import CLASS_NAMES, REFERABLE_THRESHOLD

MIN_TOP_PROBABILITY = 0.50
REFERRAL_MARGIN = 0.20
USABLE_NEEDS_REVIEW = True
TEMPERATURE = 1.21            # fitted on APTOS out-of-fold predictions (environment variable RG_TEMPERATURE overrides)
STATUS = "proposed - chosen on APTOS cross-validation only, not approved by the team"


def temperature() -> float:
    return float(os.environ.get("RG_TEMPERATURE", TEMPERATURE))


def calibrate(probs, T: float | None = None) -> np.ndarray:
    p = np.clip(np.asarray(probs, np.float64), 1e-8, 1) ** (1.0 / (temperature() if T is None else T))
    return p / p.sum(axis=-1, keepdims=True)


def _as_array(probabilities) -> np.ndarray:
    if isinstance(probabilities, dict):
        probabilities = [probabilities[c] for c in CLASS_NAMES]
    p = np.asarray(probabilities, np.float64)
    if p.shape != (5,):
        raise ValueError(f"Expected 5 class probabilities, got shape {p.shape}")
    return p


def assess(probabilities, quality_label: str | None = None) -> dict:
    """-> {"needs_review", "reasons", "confidence_raw", "confidence_calibrated", "status"}"""
    p = _as_array(probabilities)
    grade, top = int(np.argmax(p)), float(p.max())
    p_ref = float(p[2:].sum())
    reasons = []
    if top < MIN_TOP_PROBABILITY:
        reasons.append(f"The model is unsure about the grade (top probability {top:.0%}).")
    if abs(p_ref - REFERABLE_THRESHOLD) < REFERRAL_MARGIN:
        reasons.append(f"The referral decision is borderline (P(grade >= 2) = {p_ref:.2f}, threshold {REFERABLE_THRESHOLD:.2f}).")
    if quality_label == "Usable" and USABLE_NEEDS_REVIEW:
        reasons.append("Image quality is limited (Usable).")
    return {"needs_review": bool(reasons), "reasons": reasons, "confidence_raw": top,
            "confidence_calibrated": float(calibrate(p)[grade]), "status": STATUS}
