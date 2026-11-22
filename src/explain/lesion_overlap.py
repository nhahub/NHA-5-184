"""
How well does a Grad-CAM heatmap agree with expert lesion masks? (IDRiD; analysis only, no TensorFlow)

All arrays are in the model's 384x384 frame: the heatmap, the retina mask and the lesion masks
(moved there with transform_mask, so they use exactly the crop / padding / resize of the image).

Metrics for ONE image (inside the retina only; lesions dilated by `tol` pixels to allow for the coarse
12x12 Grad-CAM grid, one cell = 32 px):
  pointing      1 if the hottest pixel is on a (dilated) lesion                    - pointing game
  energy        share of the total heat that falls on the (dilated) lesions        - 0..1
  precision@k   share of the hottest k% of retina pixels that lie on lesions (dilated)
  recall@k      share of the lesion pixels (not dilated) covered by the hottest k%
  enrichment@k  precision@k / lesion fraction of the retina (1 = no better than chance)

Baselines (same metrics, so the numbers are comparable):
  random        expectation of a random map: pointing = energy = precision@k = lesion fraction, recall@k = k
  optic_disc    heat centred on the optic disc (Gaussian)
  centre        heat centred on the retina (Gaussian): the model may simply look at the middle
"""
from __future__ import annotations

import cv2
import numpy as np

TYPES = ("MA", "HE", "EX", "SE")            # microaneurysms, haemorrhages, hard / soft exudates
KS = (5, 10, 20)
BACKGROUND_MAX = 12.0                       # lowest threshold retina_mask may use (max channel, 0..255)


def retina_mask(image_0_255: np.ndarray) -> np.ndarray:
    """(S, S, 3) model input -> bool mask of the retina (used for the metrics and for drawing the heatmap).

    A fixed low threshold is not enough: CLAHE can lift the noise of the black background above it, and the mask then
    spilled into the corners (up to the whole photo rectangle). So: median filter, a threshold between the darkest corner
    and the image centre (dark photos keep their retina), and only the largest region, with its holes filled."""
    v = np.clip(np.asarray(image_0_255, np.float32), 0, 255).astype(np.uint8).max(axis=2)
    S = v.shape[0]
    v7, c = cv2.medianBlur(v, 7), S // 16
    dark = min(float(np.median(p)) for p in (v7[:c, :c], v7[:c, -c:], v7[-c:, :c], v7[-c:, -c:]))     # darkest corner
    centre = float(np.median(v7[S // 4:S - S // 4, S // 4:S - S // 4]))
    level = max(BACKGROUND_MAX, dark + 0.15 * (centre - dark))
    m = (v7 > level).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(v.shape, np.uint8)
    if cs:
        cv2.drawContours(out, [max(cs, key=cv2.contourArea)], -1, 1, -1)
    return out.astype(bool)


def dilate(mask: np.ndarray, tol: int) -> np.ndarray:
    if tol <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def top_k_region(heat: np.ndarray, retina: np.ndarray, k: float) -> np.ndarray:
    """Bool mask of the hottest k% of the retina pixels (exact count, ties broken by position)."""
    idx = np.flatnonzero(retina)
    n = max(1, int(round(len(idx) * k / 100)))
    order = np.argsort(-heat.ravel()[idx], kind="stable")
    out = np.zeros(heat.size, bool)
    out[idx[order[:n]]] = True
    return out.reshape(heat.shape)


def metrics(heat: np.ndarray, lesion: np.ndarray, retina: np.ndarray, tol: int = 16) -> dict:
    heat = np.where(retina, np.maximum(heat, 0), 0).astype(np.float64)
    lesion = lesion & retina
    big = dilate(lesion, tol) & retina
    frac = big.sum() / retina.sum()
    out = {"lesion_fraction": float(frac)}
    if heat.max() > 0 and lesion.any():
        peak = np.unravel_index(np.argmax(heat), heat.shape)
        out["pointing"] = float(big[peak])
        out["energy"] = float(heat[big].sum() / heat.sum())
    else:
        out["pointing"] = out["energy"] = 0.0
    for k in KS:
        region = top_k_region(heat, retina, k)
        out[f"precision@{k}"] = float(big[region].mean())
        out[f"recall@{k}"] = float(region[lesion].mean()) if lesion.any() else np.nan
        out[f"enrichment@{k}"] = out[f"precision@{k}"] / frac if frac > 0 else np.nan
    return out


def random_expectation(lesion: np.ndarray, retina: np.ndarray, tol: int = 16) -> dict:
    """Analytic expectation of `metrics` for a heatmap that is random in the retina."""
    big = dilate(lesion & retina, tol) & retina
    f = float(big.sum() / retina.sum())
    out = {"lesion_fraction": f, "pointing": f, "energy": f}
    for k in KS:
        out[f"precision@{k}"] = f
        out[f"recall@{k}"] = k / 100
        out[f"enrichment@{k}"] = 1.0
    return out


def gaussian_map(shape, centre, sigma) -> np.ndarray:
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return np.exp(-((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2) / (2 * sigma ** 2))


def baseline_map(kind: str, retina: np.ndarray, optic_disc: np.ndarray | None = None) -> np.ndarray:
    """kind "centre" or "optic_disc" (needs the OD mask; falls back to the retina centre)."""
    ys, xs = np.nonzero(retina)
    c, r = (ys.mean(), xs.mean()), (retina.sum() / np.pi) ** 0.5
    sigma = 0.25 * r
    if kind == "optic_disc" and optic_disc is not None and optic_disc.any():
        oy, ox = np.nonzero(optic_disc)
        c, sigma = (oy.mean(), ox.mean()), 2 * (optic_disc.sum() / np.pi) ** 0.5
    return gaussian_map(retina.shape, c, sigma)


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Mean of (a - b) over images with a 95% bootstrap CI (paired by image)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(d))
