"""
Re-score already-trained fold models (no training). Runs on Kaggle GPU.

Two jobs, both driven by a config dict (injected into kaggle/kernels/rg_predict.py):

  split="oof"   each fold model predicts ITS OWN fold-val images (optionally with 4-view flip TTA).
                Writes runs/<run_id>/{run.json, oof_preds.npz} in the same format as train.py,
                so src/evaluation/report.py picks them up (run_id gets a "_tta" suffix).

  split="test"  Phase 3 only, run ONCE after the configuration is frozen. Every listed model predicts
                the held-out test split; per-model outputs are stored and the ensemble is formed
                locally. Refuses to run unless cfg["i_confirm_frozen"] is True.

Model files are found by name (<run_id>_model.keras) anywhere under /kaggle/input, i.e. the
rg-train-* kernels are attached as kernel sources.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import aptos_v2 as D                      # noqa: E402
from evaluation import metrics as M                 # noqa: E402
import train as T                                   # noqa: E402

DEFAULTS = dict(
    candidate="B",
    seeds=[42, 7, 1234],
    folds=[0, 1, 2, 3, 4],
    img_size=224,
    tta=True,
    split="oof",               # "oof" | "test"
    i_confirm_frozen=False,    # must be True for split="test"
    batch_size=64,
    mixed_precision=True,
    data_dir=None,
    out_dir="/kaggle/working/runs",
)


def model_run_id(cand: str, fold: int, seed: int, img: int) -> str:
    return T.run_id({**T.DEFAULTS, "candidate": cand, "seed": seed, "img_size": img, "quick_test": False}, fold)


def raw_predict(cand: str, model, X, idx, cfg) -> dict:
    out = D.predict_memmap(model, X, idx, batch_size=cfg["batch_size"], tta=cfg["tta"])
    return {"probs5": out} if cand == "B" else {"score": out.ravel()}


def main(overrides: dict | None = None):
    cfg = {**DEFAULTS, **(overrides or {})}
    assert cfg["candidate"] in ("B", "C"), "cascade (A) re-scoring not needed"
    assert cfg["split"] in ("oof", "test")
    if cfg["split"] == "test":
        assert cfg["i_confirm_frozen"], "test split is scored once, after the freeze (docs/experiments.md)"

    import tensorflow as tf
    if cfg["mixed_precision"] and tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    X, meta = D.load(cfg["data_dir"])
    D.check_splits(meta)
    assert X.shape[1] == cfg["img_size"]
    grade = meta.grade.to_numpy().astype(np.int64)
    ids = meta.id_code.to_numpy().astype(str)
    print("TF", tf.__version__, "| config:", json.dumps(cfg))

    records = []
    for seed in cfg["seeds"]:
        for fold in cfg["folds"]:
            t0 = time.time()
            src_rid = model_run_id(cfg["candidate"], fold, seed, cfg["img_size"])
            path = D.find_file(f"{src_rid}_model.keras")
            model = tf.keras.models.load_model(path, compile=False)
            idx = D.fold_indices(meta, fold)[1] if cfg["split"] == "oof" else D.test_indices(meta)
            res = T.summarize_predictions(cfg["candidate"], grade[idx], raw_predict(cfg["candidate"], model, X, idx, cfg))
            rid = src_rid + ("_tta" if cfg["tta"] else "") + ("_test" if cfg["split"] == "test" else "")
            out_dir = os.path.join(cfg["out_dir"], rid)
            os.makedirs(out_dir, exist_ok=True)
            fname = "oof_preds.npz" if cfg["split"] == "oof" else "test_preds.npz"
            np.savez_compressed(os.path.join(out_dir, fname), id_code=ids[idx], fold=fold, seed=seed,
                                candidate=cfg["candidate"], **res)
            report = M.full_report(res["y_true"], res["pred"], res["score_dr"], res["score_ref"], n_boot=200)
            rec = {
                "run_id": rid, "source_run_id": src_rid, "candidate": cfg["candidate"], "fold": fold, "seed": seed,
                "img_size": cfg["img_size"], "quick_test": False, "tta": cfg["tta"], "split": cfg["split"],
                "config": cfg, "code_version": T._code_version(), "model_file": path,
                "data": {"n_train": 0, "n_val": int(len(idx)), "meta_rows": int(len(meta))},
                "metrics": report if cfg["split"] == "oof" else None,
                "test_metrics": report if cfg["split"] == "test" else None,
                "duration_s": round(time.time() - t0, 1),
            }
            # report.py reads run.json for OOF; a test run gets a different name so it never mixes in
            with open(os.path.join(out_dir, "run.json" if cfg["split"] == "oof" else "test_run.json"), "w") as f:
                json.dump(rec, f, indent=2, default=float)
            print(f"=== {rid}: QWK={report['qwk']:.4f} ref sens/spec={report['referable_sens']:.3f}/"
                  f"{report['referable_spec']:.3f} ({rec['duration_s']}s)")
            records.append(rec)
            del model
            tf.keras.backend.clear_session()
    return records
