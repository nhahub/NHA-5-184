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

    conv = tf.cast(fm(x, training=False), tf.float32)           # (1, h, w, C); computed outside the tape
    with tf.GradientTape() as tape:
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


def display_mask(image_0_255: np.ndarray) -> np.ndarray:
    """(S, S, 3) prepared image -> bool mask of the retina, used for DRAWING only (the IDRiD metrics have their own mask).

    A fixed low threshold is not enough: CLAHE lifts the noise of the black background above it, and the heatmap was
    then painted as speckles outside the eye. So: median filter, a threshold between the darkest corner and the
    image centre (dark photos keep their retina), and only the largest region, with its holes filled."""
    import cv2
    v = np.clip(np.asarray(image_0_255, np.float32), 0, 255).astype(np.uint8).max(axis=2)
    S = v.shape[0]
    v7, c = cv2.medianBlur(v, 7), S // 16
    dark = min(float(np.median(p)) for p in (v7[:c, :c], v7[:c, -c:], v7[-c:, :c], v7[-c:, -c:]))     # darkest corner
    centre = float(np.median(v7[S // 4:S - S // 4, S // 4:S - S // 4]))
    level = max(12.0, dark + 0.15 * (centre - dark))
    m = (v7 > level).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(v.shape, np.uint8)
    if cs:
        cv2.drawContours(out, [max(cs, key=cv2.contourArea)], -1, 1, -1)
    return out.astype(bool)


def overlay(image_0_255: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """RGB uint8 image with the heatmap (jet colours) blended on top; background stays dark."""
    import cv2
    img = np.clip(np.asarray(image_0_255, np.float32), 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(heatmap, 0, 1)), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)
    retina = display_mask(img)                                   # keep the black background black
    out[~retina] = img[~retina]
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a.ravel())).astype(np.float64)
    rb = np.argsort(np.argsort(b.ravel())).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def randomized_copy(model, level: str = "all", seed: int = 0):
    """Copy of the model with weights re-initialised from the top down (Adebayo et al., cascading):
    level "head" = dense layer, "top" = dense + last backbone block + top conv, "all" = every trained layer.
    The ImageNet input normalisation is kept, so the random network still gets correctly scaled input."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    rand = tf.keras.models.clone_model(model)                    # fresh initialisation
    if level == "all":
        for b_src, b_dst in zip(_parts(model)[0].layers, _parts(rand)[0].layers):
            if isinstance(b_src, tf.keras.layers.Normalization) and b_src.get_weights():
                b_dst.set_weights(b_src.get_weights())
        return rand
    top = ("block7", "top_")
    for src, dst in zip(model.layers, rand.layers):
        if isinstance(src, tf.keras.Model):
            for bs, bd in zip(src.layers, dst.layers):
                keep = not (level == "top" and bs.name.startswith(top))
                if keep and bs.get_weights():
                    bd.set_weights(bs.get_weights())
        elif src.get_weights() and not isinstance(src, tf.keras.layers.Dense):
            dst.set_weights(src.get_weights())
    return rand


def randomization_check(model, images_0_255, target="pred", levels=("head", "top", "all"), retina_fn=None) -> dict:
    """Spearman correlation (inside the retina) between Grad-CAM of the trained model and of its randomised
    copies. A faithful explanation must drop clearly. `between_images` is the reference: the correlation of
    maps of DIFFERENT images, i.e. what the shared retina geometry alone produces."""
    fm = _feature_model(_parts(model)[0])
    maps, targets, masks = [], [], []
    for img in images_0_255:
        h, _, t = gradcam(model, img, target, fm)
        maps.append(h)
        targets.append(t)
        masks.append(retina_fn(img) if retina_fn else np.ones(h.shape, bool))

    def corr(a, b, m):
        return spearman(a[m], b[m]) if np.ptp(a[m]) > 0 and np.ptp(b[m]) > 0 else float("nan")

    out = {"n": len(maps), "between_images": [corr(maps[i], maps[i - 1], masks[i] & masks[i - 1]) for i in range(1, len(maps))]}
    for lv in levels:
        rand = randomized_copy(model, lv)
        fr = _feature_model(_parts(rand)[0])
        out[lv] = [corr(m, gradcam(rand, img, t, fr)[0], k) for m, img, t, k in zip(maps, images_0_255, targets, masks)]
    return out
