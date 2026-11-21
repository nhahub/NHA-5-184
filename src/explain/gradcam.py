"""
Grad-CAM for RetinoGrade model B (EfficientNetB0 backbone nested inside a Keras model).

What it shows: which regions of the (preprocessed) fundus image pushed the model towards a grade.
How (Selvaraju et al., 2017):
  1. run the image through the backbone up to its last convolutional block ("top_activation"),
  2. re-apply the classification head by hand (global average pool -> dense layer),
  3. take the gradient of the chosen target score w.r.t. the last conv feature maps,
  4. weight each feature map by its average gradient, sum, keep positive values (ReLU),
  5. upsample to the input size and scale to [0, 1].
The head is re-applied manually because the backbone is a nested model:
Keras cannot give gradients w.r.t. an inner layer of a nested model through the outer model.

Targets:
  "pred"       the predicted grade (default)
  0..4         a specific grade
  "referable"  log P(grade >= 2) - the screening decision (what made the model refer this eye?)

Sanity check (Adebayo et al., 2018, "model randomization"): a faithful explanation must change when
the trained weights are replaced by random ones. `randomization_check` measures that.
"""
from __future__ import annotations

import numpy as np

LAST_CONV = "top_activation"


def _parts(model):
    """-> (backbone, dense head layer) of a model built by src/training/train.build_model."""
    import tensorflow as tf
    backbone = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)][-1]
    return backbone, dense


def _feature_model(backbone):
    import tensorflow as tf
    return tf.keras.Model(backbone.inputs, backbone.get_layer(LAST_CONV).output)


def gradcam(model, image_0_255: np.ndarray, target="pred", feature_model=None):
    """Grad-CAM heatmap for ONE preprocessed image.

    image_0_255: (S, S, 3) float/uint8 in 0..255 (output of preprocess_rgb * 255, the model's input).
    Returns (heatmap (S, S) float32 in [0, 1], probs5 (5,), target used).
    """
    import tensorflow as tf
    backbone, dense = _parts(model)
    fm = feature_model or _feature_model(backbone)
    x = tf.convert_to_tensor(np.asarray(image_0_255, np.float32)[None])
    W, b = [tf.cast(w, tf.float32) for w in dense.weights[:2]]

    with tf.GradientTape() as tape:
        conv = tf.cast(fm(x, training=False), tf.float32)      # (1, h, w, C)
        tape.watch(conv)
        logits = tf.reduce_mean(conv, axis=(1, 2)) @ W + b      # GAP -> dense (dropout is off at inference)
        probs = tf.nn.softmax(logits)
        if target == "pred":
            target = int(tf.argmax(probs[0]))
        if target == "referable":
            score = tf.math.log(tf.reduce_sum(probs[:, 2:], axis=1) + 1e-7)
        else:
            score = logits[:, int(target)]
    grads = tape.gradient(score, conv)[0]                       # (h, w, C)
    weights = tf.reduce_mean(grads, axis=(0, 1))                # (C,)
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    S = image_0_255.shape[0]
    import cv2
    cam = cv2.resize(cam, (S, S), interpolation=cv2.INTER_LINEAR)
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam.astype(np.float32), probs.numpy()[0], target


def overlay(image_0_255: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """RGB uint8 image with the heatmap (jet colours) blended on top; background stays dark."""
    import cv2
    img = np.clip(np.asarray(image_0_255, np.float32), 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(heatmap, 0, 1)), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)
    retina = img.max(axis=2) > 12                                # keep the black background black
    out[~retina] = img[~retina]
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a.ravel())).astype(np.float64)
    rb = np.argsort(np.argsort(b.ravel())).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def randomization_check(model, images_0_255, target="pred", seed: int = 0) -> dict:
    """Model-randomization sanity check: Spearman correlation between Grad-CAM maps of the trained
    model and of a copy whose weights are re-drawn at random (same shapes, Glorot-like scale).
    Low correlation = the explanation depends on what the model learned (good)."""
    import tensorflow as tf
    rand = tf.keras.models.clone_model(model)
    rng = np.random.default_rng(seed)
    new = []
    for w in model.get_weights():
        if w.ndim >= 2:
            fan = np.prod(w.shape[:-1])
            new.append(rng.normal(0, np.sqrt(2.0 / fan), w.shape).astype(w.dtype))
        else:
            new.append(w)                                        # keep biases / BN statistics
    rand.set_weights(new)
    fm_t, fm_r = _feature_model(_parts(model)[0]), _feature_model(_parts(rand)[0])
    rhos = []
    for img in images_0_255:
        h_t, _, t = gradcam(model, img, target, fm_t)
        h_r, _, _ = gradcam(rand, img, t, fm_r)
        rhos.append(spearman(h_t, h_r))
    return {"spearman_trained_vs_random": rhos, "mean": float(np.mean(rhos)), "max": float(np.max(rhos))}
