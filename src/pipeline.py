"""
The whole path for one photo: quality gate -> grading -> one result dictionary.

    from pipeline import RetinoPipeline
    pipe = RetinoPipeline()                 # weights from $RG_MODEL and $RG_QUALITY_MODEL or the default paths
    r = pipe.predict("eye.png")
    r["quality"]["action"]                  # "ok" | "warn" | "retake"
    r["grade"], r["referable"], ...         # every key of RetinoGrade.predict

`quality` is ADDED to the grading output; no other key changes. If the action is "retake" the grade is still in
the dictionary (so nothing downstream breaks), but a user interface must show the message instead of the grade.
Without a quality model (quality_model_path=None and no default file) the pipeline works and `quality` is None.
"""
from __future__ import annotations

import os

from retinograde_inference import RetinoGrade, load_rgb
from quality import quality as Q


class RetinoPipeline:
    def __init__(self, model_path=None, quality_model_path=None, require_quality: bool = False, use_gate: bool = True):
        self.grader = RetinoGrade(model_path)
        if not use_gate:                                # only for tests
            self.gate = None
            return
        try:
            self.gate = Q.QualityGate(quality_model_path)
        except FileNotFoundError:
            if require_quality or quality_model_path or os.environ.get(Q.MODEL_ENV):
                raise                               # an explicitly named model must exist
            self.gate = None

    def predict(self, source) -> dict:
        img = load_rgb(source)
        quality = None
        if self.gate is not None:
            if min(img.shape[:2]) < Q.MIN_SIDE:
                quality = {"label": "Reject", "action": "retake", "confidence": 1.0, "probabilities": None,
                           "message": f"The image is too small to judge (shorter side {min(img.shape[:2])} px, at least {Q.MIN_SIDE} px needed)."}
            else:
                quality = self.gate.predict(img)
        out = self.grader.predict(img)
        out["quality"] = quality
        return out
