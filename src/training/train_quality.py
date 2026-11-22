"""
Train the image-quality gate (EyeQ labels, EfficientNetB0 at 224px, 3 classes). Runs on a Kaggle GPU.

Data (kernel rg-quality-data): quality_u8.npy + quality_meta.csv (stem, patient, quality, dr_grade, split)
  qtrain / qval : "rest" EyePACS patients (epoch selection on qval only)
  qtest         : EyePACS DEV patients - scored once per seed at the end, nothing is chosen on them
  The EyePACS held-out TEST patients are not in the file at all.

Augmentation is geometric only (flips, 90-degree turns): changing brightness or contrast would change the
very property the labels describe. The deployed model is seed 42, fixed beforehand.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T                                    # noqa: E402
from data import aptos_v2 as D                       # noqa: E402
from evaluation import quality_metrics as Q          # noqa: E402

DEFAULTS = dict(seeds=[42, 7, 1234], batch=64, epochs_head=2, epochs_ft=10, lr_head=1e-3, lr_ft=1e-4, dropout=0.3,
                class_weight_power=0.5, patience=4, out_dir="/kaggle/working/runs", quick_test=False)


def augment(x, rng):
    """(N,S,S,3) uint8 -> random flips / quarter turns, same for every channel."""
    out = np.empty_like(x)
    for i, im in enumerate(x):
        im = np.rot90(im, int(rng.integers(4)))
        out[i] = im[:, ::-1] if rng.random() < 0.5 else im
    return out


def dataset(X, idx, y, w, batch, seed, training):
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    ds = tf.data.Dataset.from_tensor_slices((idx.astype(np.int64), y.astype(np.int32), w.astype(np.float32)))
    if training:
        ds = ds.shuffle(len(idx), seed=seed, reshuffle_each_iteration=True)
    S = X.shape[1]

    def load(i):
        o = np.argsort(i)
        b = np.empty((len(i), S, S, 3), np.uint8)
        b[o] = X[np.sort(i)]
        return (augment(b, rng) if training else b).astype(np.float32)

    def f(i, yy, ww):
        im = tf.numpy_function(load, [i], tf.float32)
        im.set_shape([None, S, S, 3])
        return im, yy, ww

    return ds.batch(batch).map(f, num_parallel_calls=1).prefetch(2)


def run_seed(cfg, X, meta, seed):
    import tensorflow as tf
    t0 = time.time()
    T.set_seeds(seed)
    q = meta.quality.to_numpy().astype(int)
    ix = {s: np.flatnonzero((meta.split == s).to_numpy()) for s in ("qtrain", "qval", "qtest")}
    if cfg["quick_test"]:
        ix = {s: v[:: max(1, len(v) // 200)] for s, v in ix.items()}
    model, base = T.build_model("softmax3", X.shape[1], cfg["dropout"])
    w = T.balanced_sample_weights(q[ix["qtrain"]], cfg["class_weight_power"], 3)
    ds = dataset(X, ix["qtrain"], q[ix["qtrain"]], w, cfg["batch"], seed, True)
    log, best = [], {"score": -1.0, "weights": None, "wait": 0}

    class Keep(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            p = D.predict_memmap(self.model, X, ix["qval"])
            s = Q.macro_f1(q[ix["qval"]], p.argmax(1))
            log.append({"phase": self.phase, "epoch": epoch, "qval_macro_f1": s, "loss": float(logs["loss"])})
            print(f"  [{self.phase}] epoch {epoch}: qval macro-F1 {s:.4f}")
            if s > best["score"] + 1e-4:
                best.update(score=s, weights=self.model.get_weights(), wait=0)
            else:
                best["wait"] += 1
                if best["wait"] >= cfg["patience"]:
                    self.model.stop_training = True

    cb = Keep()
    for phase, lr, ep in (("head", cfg["lr_head"], cfg["epochs_head"]), ("finetune", None, cfg["epochs_ft"])):
        cb.phase, best["wait"] = phase, 0
        if phase == "finetune":
            T._unfreeze(base, -1)
            lr = tf.keras.optimizers.schedules.CosineDecay(cfg["lr_ft"], int(np.ceil(len(ix["qtrain"]) / cfg["batch"])) * ep)
        T._compile(model, "softmax3", lr)
        model.fit(ds, epochs=1 if cfg["quick_test"] else ep, callbacks=[cb], verbose=2)
    model.set_weights(best["weights"])

    out = os.path.join(cfg["out_dir"], f"quality_s{seed}")
    os.makedirs(out, exist_ok=True)
    rec = {"run_id": f"quality_s{seed}", "seed": seed, "config": cfg, "code_version": T._code_version(),
           "n": {s: int(len(v)) for s, v in ix.items()}, "epochs": log, "best_qval_macro_f1": best["score"]}
    for s in ("qval", "qtest"):
        p = D.predict_memmap(model, X, ix[s])
        np.savez_compressed(os.path.join(out, f"{s}_preds.npz"), stem=meta.stem.to_numpy().astype(str)[ix[s]],
                            patient=meta.patient.to_numpy().astype(str)[ix[s]], y=q[ix[s]], probs3=p,
                            dr_grade=meta.dr_grade.to_numpy().astype(int)[ix[s]])
        rec[s] = Q.report(q[ix[s]], p)
    model.save(os.path.join(out, f"quality_b0_224_s{seed}.keras"), include_optimizer=False)
    rec["duration_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out, "quality_run.json"), "w") as f:
        json.dump(rec, f, indent=1, default=float)
    print(f"=== seed {seed}: qval F1 {best['score']:.4f} | qtest {rec['qtest']['accuracy']:.4f} acc, "
          f"{rec['qtest']['macro_f1']:.4f} F1 ({rec['duration_s']:.0f}s)")
    tf.keras.backend.clear_session()
    return rec


def main(overrides=None):
    import pandas as pd
    import tensorflow as tf
    cfg = {**DEFAULTS, **(overrides or {})}
    if tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    path = D.find_file("quality_u8.npy")
    X = np.load(path, mmap_mode="r")
    meta = pd.read_csv(os.path.join(os.path.dirname(path), "quality_meta.csv"), dtype={"patient": str})
    assert len(X) == len(meta) and set(meta.split) == {"qtrain", "qval", "qtest"}
    for a, b in (("qtrain", "qval"), ("qtrain", "qtest"), ("qval", "qtest")):
        assert not set(meta.patient[meta.split == a]) & set(meta.patient[meta.split == b]), (a, b)
    print("TF", tf.__version__, "|", meta.split.value_counts().to_dict())
    return [run_seed(cfg, X, meta, s) for s in cfg["seeds"]]
