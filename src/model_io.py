"""
Loading trained models for CPU inference.

Decision (2026-11-20, evidence in docs/experiments.md and kaggle/runs/rg-latency/latency.json):
load the saved .keras file AS IS, even though the models were trained with mixed precision.

  * The saved model run on a Kaggle CPU reproduces the GPU out-of-fold predictions it made during
    training: max |prob difference| 0.006 (224px) / 0.006 (384px), same grade on 33/33 images.
  * Two attempts to make a "float32 copy" (rebuild + copy weights) gave WRONG outputs on real images
    (only 79-85% of grades agree) while still passing a synthetic-input check. The speed gain would
    have been ~15% (385 vs 448 ms per image), not worth the risk. Not used.

Lesson kept as a rule: any change to how a model is loaded must be checked against stored
predictions on REAL images (see `check_against_reference`), not only on synthetic input.
"""
from __future__ import annotations

import numpy as np


def load_model(path: str):
    """Load a model trained by src/training/train.py for inference (CPU or GPU)."""
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


def check_against_reference(model, images_0_255: np.ndarray, reference_probs: np.ndarray, tol: float = 0.02) -> float:
    """Raise if the model's predictions on real images differ from stored reference predictions."""
    p = np.asarray(model(np.asarray(images_0_255, np.float32), training=False), np.float64)
    diff = float(np.abs(p - np.asarray(reference_probs, np.float64)).max())
    if diff > tol:
        raise RuntimeError(f"model output differs from reference predictions (max prob diff {diff:.3f})")
    return diff
