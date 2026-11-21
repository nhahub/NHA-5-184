"""
RetinoGrade AI - v2 training harness (config-driven; runs inside Kaggle kernels, TF 2.20 / Keras 3).

Candidates (see docs/experiments.md):
  A  cascade  : Stage 1 sigmoid (DR vs no DR) + Stage 2 softmax over grades 1-4 (v1 design, retrained on folds)
  B  softmax5 : single EfficientNet, 5-class softmax
  C  ordinal  : single EfficientNet, 1 linear output, Huber loss, grade = thresholds(score)

Per fold it writes to <out_dir>/<run_id>/:
  run.json        config + fold-val metrics + history (small -> downloaded for docs/experiments.md)
  oof_preds.npz   fold-val predictions (small -> downloaded; report.py stitches 5 folds into OOF)
  *.keras         model(s) (stay on Kaggle)
The test split is NOT touched unless cfg["predict_test"] is True (Phase 3, once).

Usage inside a kernel:
    import train; train.main({"candidate": "B", "folds": [0, 1, 2, 3, 4], "seed": 42})
Locally (no TF) only the pure helpers are importable.
"""
from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))           # .../src

from data import aptos_v2 as D                      # noqa: E402
from evaluation import metrics as M                 # noqa: E402

DEFAULTS = dict(
    candidate="B",            # "A" | "B" | "C"
    folds=[0],                # folds run sequentially in this kernel
    seed=42,
    img_size=224,             # must equal the stored array size (384 needs an rg-data-v2 rebuild)
    quick_test=False,         # True: 64 train / 48 val images, 1+1 epochs (smoke test)
    batch_size=32,
    epochs_head=3, lr_head=1e-3,                       # phase 1: backbone frozen
    epochs_ft=15, lr_ft=1e-4, finetune_layers=-1,      # phase 2: -1 = whole backbone (BatchNorm stays frozen)
    patience=5,
    dropout=0.3,
    class_weight_power=0.5,   # sample weight = (balanced weight) ** power, from ORIGINAL images only
    augment=True,             # on-the-fly only (no stored augmented copies)
    mixed_precision=True,
    tta=False,                # 4-view flip TTA at prediction time (candidate G)
    cascade_threshold=0.3,    # candidate A decision threshold (re-tuned on OOF by report.py)
    predict_test=False,       # Phase 3 only
    train_fraction=1.0,       # <1: nested class-stratified subset of the fold's training images (learning curve)
    extra_data=None,          # "eyepacs": add EyePACS rest images to every fold's training set (mixed training)
    extra_n=8000,             # EyePACS images per fold (class-stratified sample of the rtrain pool)
    extra_val_weight=0.5,     # epoch selection score = (1-w) * QWK(APTOS fold-val) + w * QWK(EyePACS rval)
    extra_val_max=800,        # EyePACS rval images scored per epoch (stratified subset, keeps epochs fast)
    eval_eyepacs_dev=False,   # after training, score the EyePACS dev patients (never used for selection)
    ft_stop_epoch=None,       # full-pool model: stop fine-tuning after this many epochs, keeping the
                              # learning-rate schedule of epochs_ft (= where the CV runs peaked)
    save_model=True,
    weights="imagenet",
    data_dir=None,            # None = search /kaggle/input for aptos_v2_u8.npy
    out_dir="/kaggle/working/runs",
)


# ----------------------------------------------------------------------------- pure helpers (testable locally)
def run_id(cfg: dict, fold: int) -> str:
    part = "full" if fold < 0 else f"f{fold}"           # fold -1 = train on the whole training pool
    rid = f"{cfg['candidate']}_{part}_s{cfg['seed']}_{cfg['img_size']}"
    if cfg.get("train_fraction", 1.0) < 1.0:
        rid += f"_p{round(cfg['train_fraction'] * 100)}"
    if cfg.get("extra_data"):
        rid += "_mix"
    if cfg.get("tta"):
        rid += "_tta"
    return rid + ("_qt" if cfg["quick_test"] else "")


def balanced_sample_weights(labels, power: float, n_classes: int) -> np.ndarray:
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=n_classes)
    w = (len(labels) / (n_classes * np.maximum(counts, 1))) ** power
    return w[labels].astype(np.float32)


def quick_subset(idx: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, min(n, len(idx)), replace=False))


def fraction_subset(idx: np.ndarray, labels: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Class-stratified subset of `idx` (labels aligned with idx). Nested: for a fixed seed the subset
    for a smaller fraction is always contained in the subset for a larger one (same permutation, prefix)."""
    if frac >= 1.0:
        return np.asarray(idx)
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        members = np.asarray(idx)[labels == c]
        perm = members[rng.permutation(len(members))]
        keep.append(perm[:max(1, int(round(frac * len(members))))])
    return np.sort(np.concatenate(keep))


def summarize_predictions(candidate: str, y_true, out: dict, cascade_threshold: float = 0.3) -> dict:
    """Turn raw model outputs into pred / score_dr / score_ref (same for fold-val, OOF and test)."""
    if candidate == "A":
        probs5 = M.cascade_probs5(out["p1"], out["p2"])
        pred = M.cascade_predict(out["p1"], out["p2"], cascade_threshold)
        res = dict(probs5=probs5, pred=pred, score_dr=np.asarray(out["p1"]).ravel(),
                   score_ref=probs5[:, 2:].sum(1), p1=np.asarray(out["p1"]).ravel(), p2=out["p2"])
    elif candidate == "B":
        probs5 = np.asarray(out["probs5"], dtype=np.float64)
        res = dict(probs5=probs5, pred=probs5.argmax(1), score_dr=1.0 - probs5[:, 0],
                   score_ref=probs5[:, 2:].sum(1))
    elif candidate == "C":
        score = np.asarray(out["score"], dtype=np.float64).ravel()
        res = dict(score=score, pred=M.apply_thresholds(score), score_dr=score, score_ref=score)
    else:
        raise ValueError(candidate)
    res["y_true"] = np.asarray(y_true).astype(np.int64)
    return res


# ----------------------------------------------------------------------------- TF parts
def set_seeds(seed: int) -> None:
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_model(kind: str, img_size: int, dropout: float, weights="imagenet"):
    """kind: sigmoid | softmax4 | softmax5 | regression. Returns (model, backbone)."""
    import tensorflow as tf
    from tensorflow.keras import layers

    base = tf.keras.applications.EfficientNetB0(weights=weights, include_top=False,
                                                input_shape=(img_size, img_size, 3))
    base.trainable = False
    inputs = layers.Input(shape=(img_size, img_size, 3))      # 0-255 input (EfficientNet rescales inside)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    if kind == "sigmoid":
        outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    elif kind == "softmax4":
        outputs = layers.Dense(4, activation="softmax", dtype="float32")(x)
    elif kind == "softmax5":
        outputs = layers.Dense(5, activation="softmax", dtype="float32")(x)
    elif kind == "regression":
        outputs = layers.Dense(1, activation="linear", dtype="float32")(x)
    else:
        raise ValueError(kind)
    return tf.keras.Model(inputs, outputs), base


def _compile(model, kind: str, lr):
    import tensorflow as tf
    if kind == "sigmoid":
        loss = "binary_crossentropy"
    elif kind == "regression":
        loss = tf.keras.losses.Huber(delta=1.0)
    else:
        loss = "sparse_categorical_crossentropy"
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss)


def _unfreeze(base, n_layers: int) -> None:
    import tensorflow as tf
    base.trainable = True
    layers_ = base.layers if n_layers < 0 else base.layers[-n_layers:]
    keep = set(id(l) for l in layers_)
    for layer in base.layers:
        layer.trainable = id(layer) in keep and not isinstance(layer, tf.keras.layers.BatchNormalization)


def _score_fn(kind: str, X, val_idx, y_val, tta: bool, extra_val=None, extra_w: float = 0.0):
    """Model-selection metric on the fold-val split (higher = better).
    extra_val: optional (X, idx, y) of a second validation set (softmax5 only); the score becomes
    (1 - extra_w) * QWK(fold-val) + extra_w * QWK(extra)."""
    base = _score_fn_single(kind, X, val_idx, y_val, tta)
    if not extra_val:
        return base
    Xe, ie, ye = extra_val

    def mixed(model):
        e = M.qwk(ye, D.predict_memmap(model, Xe, ie, tta=tta).argmax(1))
        return (1 - extra_w) * base(model) + extra_w * e
    return mixed


def _score_fn_single(kind: str, X, val_idx, y_val, tta: bool):
    def score(model):
        out = D.predict_memmap(model, X, val_idx, tta=tta)
        if kind == "sigmoid":
            return M.binary_auc(y_val, out.ravel())
        if kind == "softmax4":
            return M.qwk(y_val, out.argmax(1), n_classes=4)
        if kind == "softmax5":
            return M.qwk(y_val, out.argmax(1))
        return M.qwk(y_val, M.apply_thresholds(out.ravel()))
    return score


def train_one(kind: str, X, tr_idx, y_tr, va_idx, y_va, cfg: dict, n_classes_w: int, extra_val=None):
    """Two-phase training (head, then fine-tune); keeps the best weights by the fold-val metric."""
    import tensorflow as tf

    class BestKeeper(tf.keras.callbacks.Callback):
        def __init__(self, score, patience):
            super().__init__()
            self.score, self.patience = score, patience
            self.best, self.best_weights, self.wait, self.log = -math.inf, None, 0, []

        def on_epoch_end(self, epoch, logs=None):
            if self.score is None:                      # final model on all data: fixed epochs, no selection
                self.log.append({"phase": self.phase, "epoch": epoch,
                                 "loss": float((logs or {}).get("loss", float("nan")))})
                return
            s = float(self.score(self.model))
            self.log.append({"phase": self.phase, "epoch": epoch, "val_score": s,
                             "loss": float((logs or {}).get("loss", float("nan")))})
            print(f"  [{self.phase}] epoch {epoch}: val_score={s:.4f}")
            if s > self.best + 1e-4:
                self.best, self.best_weights, self.wait = s, self.model.get_weights(), 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = True

    model, base = build_model(kind, cfg["img_size"], cfg["dropout"], cfg["weights"])
    weights = balanced_sample_weights(y_tr, cfg["class_weight_power"], n_classes_w)
    labels = np.asarray(y_tr, dtype=np.float32 if kind in ("sigmoid", "regression") else np.int32)
    train_ds = D.make_dataset(X, tr_idx, labels, weights, cfg["batch_size"], training=True,
                              augment=cfg["augment"], seed=cfg["seed"])
    score = None if len(va_idx) == 0 else _score_fn(kind, X, va_idx, y_va, tta=False, extra_val=extra_val,
                                                    extra_w=cfg.get("extra_val_weight", 0.0))
    keeper = BestKeeper(score, cfg["patience"])
    quick = cfg["quick_test"]

    keeper.phase = "head"
    _compile(model, kind, cfg["lr_head"])
    model.fit(train_ds, epochs=1 if quick else cfg["epochs_head"], callbacks=[keeper], verbose=2)

    keeper.phase, keeper.wait = "finetune", 0
    _unfreeze(base, cfg["finetune_layers"])
    steps = math.ceil(len(tr_idx) / cfg["batch_size"]) * max(1, cfg["epochs_ft"])
    _compile(model, kind, tf.keras.optimizers.schedules.CosineDecay(cfg["lr_ft"], steps))
    n_ft = cfg.get("ft_stop_epoch") or cfg["epochs_ft"]
    model.fit(train_ds, epochs=1 if quick else n_ft, callbacks=[keeper], verbose=2)

    if keeper.best_weights is not None:
        model.set_weights(keeper.best_weights)
    return model, {"best_val_score": keeper.best, "epochs": keeper.log}


def _code_version() -> str:
    p = D.find_file("CODE_VERSION.txt", required=False)
    return open(p).read().strip() if p else "unknown"


def run_fold(cfg: dict, X, meta, fold: int) -> dict:
    import tensorflow as tf

    t0 = time.time()
    set_seeds(cfg["seed"] + fold)
    rid = run_id(cfg, fold)
    out_dir = os.path.join(cfg["out_dir"], rid)
    os.makedirs(out_dir, exist_ok=True)

    tr_idx, va_idx = D.fold_indices(meta, fold)
    if cfg["quick_test"]:
        tr_idx, va_idx = quick_subset(tr_idx, 64, cfg["seed"]), quick_subset(va_idx, 48, cfg["seed"])
    grade = meta.grade.to_numpy().astype(np.int64)
    if cfg["train_fraction"] < 1.0:    # validation fold stays complete -> scores comparable across fractions
        tr_idx = fraction_subset(tr_idx, grade[tr_idx], cfg["train_fraction"], 1000 + fold)
    y_tr, y_va = grade[tr_idx], grade[va_idx]
    print(f"\n=== {rid}: train {len(tr_idx)} {np.bincount(y_tr, minlength=5)} | "
          f"val {len(va_idx)} {np.bincount(y_va, minlength=5)}")

    Xtr, tr_fit, y_fit, extra_val, extra_info = X, tr_idx, y_tr, None, None
    if cfg.get("extra_data"):
        assert cfg["extra_data"] == "eyepacs" and cfg["candidate"] == "B", "mixed training: candidate B only"
        Xe, me = D.load_eyepacs("rest")
        ok = me.ok.to_numpy().astype(bool)
        pool = np.where(ok & (me.part == "rtrain").to_numpy())[0]
        rval = np.where(ok & (me.part == "rval").to_numpy())[0]
        ge = me.grade.to_numpy().astype(np.int64)
        sel = fraction_subset(pool, ge[pool], min(1.0, cfg["extra_n"] / len(pool)), 2000 + 10 * cfg["seed"] + fold)
        rv = fraction_subset(rval, ge[rval], min(1.0, cfg["extra_val_max"] / len(rval)), 3000 + fold)
        Xtr = D.StackedArrays(X, Xe)                          # APTOS indices unchanged, EyePACS after
        tr_fit = np.concatenate([tr_idx, len(X) + sel])
        y_fit = np.concatenate([y_tr, ge[sel]])
        extra_val = (Xe, rv, ge[rv])
        extra_info = {"n_extra_train": int(len(sel)), "extra_grades": np.bincount(ge[sel], minlength=5).tolist(),
                      "n_extra_val": int(len(rv)), "extra_val_grades": np.bincount(ge[rv], minlength=5).tolist()}
        print(f"  mixed training: + {len(sel)} EyePACS {extra_info['extra_grades']} | rval {len(rv)}")

    cand, models, hist = cfg["candidate"], {}, {}
    if cand == "A":
        models["stage1"], hist["stage1"] = train_one("sigmoid", X, tr_idx, (y_tr > 0).astype(int),
                                                     va_idx, (y_va > 0).astype(int), cfg, 2)
        tr2, va2 = tr_idx[y_tr > 0], va_idx[y_va > 0]
        models["stage2"], hist["stage2"] = train_one("softmax4", X, tr2, grade[tr2] - 1,
                                                     va2, grade[va2] - 1, cfg, 4)
    elif cand == "B":
        models["model"], hist["model"] = train_one("softmax5", Xtr, tr_fit, y_fit, va_idx, y_va, cfg, 5, extra_val)
    elif cand == "C":
        models["model"], hist["model"] = train_one("regression", X, tr_idx, y_tr, va_idx, y_va, cfg, 5)
    else:
        raise ValueError(cand)

    def predict(idx):
        if cand == "A":
            return {"p1": D.predict_memmap(models["stage1"], X, idx, tta=cfg["tta"]).ravel(),
                    "p2": D.predict_memmap(models["stage2"], X, idx, tta=cfg["tta"])}
        out = D.predict_memmap(models["model"], X, idx, tta=cfg["tta"])
        return {"probs5": out} if cand == "B" else {"score": out.ravel()}

    report = None
    if len(va_idx):
        res = summarize_predictions(cand, y_va, predict(va_idx), cfg["cascade_threshold"])
        np.savez_compressed(os.path.join(out_dir, "oof_preds.npz"), id_code=meta.id_code.to_numpy().astype(str)[va_idx],
                            fold=fold, seed=cfg["seed"], candidate=cand, **res)
        report = M.full_report(res["y_true"], res["pred"], res["score_dr"], res["score_ref"], n_boot=200)

    test_report = None
    if cfg["predict_test"]:
        te_idx = D.test_indices(meta)
        tres = summarize_predictions(cand, grade[te_idx], predict(te_idx), cfg["cascade_threshold"])
        np.savez_compressed(os.path.join(out_dir, "test_preds.npz"), id_code=meta.id_code.to_numpy().astype(str)[te_idx],
                            fold=fold, seed=cfg["seed"], candidate=cand, **tres)
        test_report = M.full_report(tres["y_true"], tres["pred"], tres["score_dr"], tres["score_ref"])

    dev_report = None
    if cfg.get("eval_eyepacs_dev"):
        assert cand == "B"
        Xd, md = D.load_eyepacs("dev")
        okd = np.where(md.ok.to_numpy().astype(bool))[0]
        pd_ = D.predict_memmap(models["model"], Xd, okd)
        yd = md.grade.to_numpy().astype(np.int64)[okd]
        np.savez_compressed(os.path.join(out_dir, "eyepacs_dev_preds.npz"), stem=md.stem.to_numpy().astype(str)[okd],
                            patient=md.patient.to_numpy().astype(str)[okd], y=yd, probs5=pd_)
        dev_report = M.full_report(yd, pd_.argmax(1), 1.0 - pd_[:, 0], pd_[:, 2:].sum(1), n_boot=200)
        print(f"  EyePACS dev: QWK {dev_report['qwk']:.4f}  referable sens/spec "
              f"{dev_report['referable_sens']:.3f}/{dev_report['referable_spec']:.3f}")

    if cfg["save_model"]:
        for name, m in models.items():
            m.save(os.path.join(out_dir, f"{rid}_{name}.keras"))

    gpus = tf.config.list_physical_devices("GPU")
    record = {
        "run_id": rid, "candidate": cand, "fold": fold, "seed": cfg["seed"],
        "img_size": cfg["img_size"], "quick_test": cfg["quick_test"], "tta": cfg["tta"],
        "config": cfg, "code_version": _code_version(),
        "data": {"n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
                 "meta_rows": int(len(meta)), "split_counts": meta.split.value_counts().to_dict()},
        "metrics": report, "test_metrics": test_report, "history": hist,
        "extra_data": extra_info, "eyepacs_dev_metrics": dev_report,
        "env": {"tf": tf.__version__, "python": platform.python_version(), "gpus": [g.name for g in gpus]},
        "duration_s": round(time.time() - t0, 1),
    }
    # the full-pool model has no fold-val metrics -> separate file so report.py's CV tables stay clean
    with open(os.path.join(out_dir, "run.json" if fold >= 0 else "full_run.json"), "w") as f:
        json.dump(record, f, indent=2, default=float)
    if report:
        print(f"=== {rid}: QWK={report['qwk']:.4f}  referable sens/spec="
              f"{report['referable_sens']:.3f}/{report['referable_spec']:.3f}  ({record['duration_s']}s)")
    else:
        print(f"=== {rid}: trained on the full pool for fixed epochs ({record['duration_s']}s)")
    tf.keras.backend.clear_session()
    return record


def main(overrides: dict | None = None):
    import tensorflow as tf

    cfg = {**DEFAULTS, **(overrides or {})}
    if cfg["mixed_precision"] and tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    X, meta = D.load(cfg["data_dir"])
    D.check_splits(meta)
    assert X.shape[1] == cfg["img_size"], f"arrays are {X.shape[1]}px, config wants {cfg['img_size']}"
    print("TF", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"), "| config:", json.dumps(cfg))
    return [run_fold(cfg, X, meta, fold) for fold in cfg["folds"]]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="JSON file with overrides of DEFAULTS")
    a = ap.parse_args()
    main(json.load(open(a.config)) if a.config else None)
