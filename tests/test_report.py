"""report.py end-to-end on fake Kaggle outputs (5 folds x 2 seeds, candidates A/B/C)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "training"))
from evaluation import report as R  # noqa: E402
import train as T  # noqa: E402


def _fake_outputs(cand, rng, y):
    n = len(y)
    if cand == "A":
        p1 = np.clip((y > 0) * 0.8 + rng.normal(0.1, 0.15, n), 0, 1)
        p2 = rng.dirichlet(np.ones(4), n)
        p2[np.arange(n), np.clip(y - 1, 0, 3)] += 1.5
        return {"p1": p1, "p2": p2 / p2.sum(1, keepdims=True)}
    if cand == "B":
        p = rng.dirichlet(np.ones(5), n)
        p[np.arange(n), y] += 1.5
        return {"probs5": p / p.sum(1, keepdims=True)}
    return {"score": y * 0.85 + rng.normal(0, 0.4, n)}


def _write_runs(root):
    rng = np.random.default_rng(0)
    for cand in "ABC":
        for seed in (1, 2):
            for fold in range(5):
                rid = f"{cand}_f{fold}_s{seed}_224"
                d = root / rid
                d.mkdir(parents=True)
                y = rng.integers(0, 5, 60)
                res = T.summarize_predictions(cand, y, _fake_outputs(cand, rng, y))
                np.savez_compressed(d / "oof_preds.npz", id_code=np.array([f"{rid}_{i}" for i in range(60)]),
                                    fold=fold, seed=seed, candidate=cand, **res)
                from evaluation import metrics as M
                rec = {"run_id": rid, "candidate": cand, "fold": fold, "seed": seed, "img_size": 224,
                       "quick_test": False, "tta": False, "code_version": "abc123",
                       "data": {"n_train": 240, "n_val": 60}, "duration_s": 600.0,
                       "metrics": M.point_metrics(res["y_true"], res["pred"], res["score_dr"], res["score_ref"])}
                (d / "run.json").write_text(json.dumps(rec))
    # one incomplete group
    d = root / "B_f0_s9_224"
    d.mkdir()
    y = rng.integers(0, 5, 60)
    res = T.summarize_predictions("B", y, _fake_outputs("B", rng, y))
    np.savez_compressed(d / "oof_preds.npz", id_code=np.arange(60).astype(str), fold=0, seed=9, candidate="B", **res)
    from evaluation import metrics as M
    (d / "run.json").write_text(json.dumps({"run_id": "B_f0_s9_224", "candidate": "B", "fold": 0, "seed": 9,
                                            "img_size": 224, "quick_test": False, "data": {"n_train": 1, "n_val": 60},
                                            "duration_s": 1.0, "metrics": M.point_metrics(y, res["pred"])}))


def test_report_end_to_end(tmp_path):
    runs = tmp_path / "runs"
    _write_runs(runs)
    doc = tmp_path / "experiments.md"
    doc.write_text("# Experiments\n\nintro\n\n<!-- RESULTS:START -->\nSTALE_BLOCK\n<!-- RESULTS:END -->\n\nfooter\n")
    R.main(["--runs", str(runs), "--doc", str(doc), "--n-boot", "50"])
    text = doc.read_text()
    assert "STALE_BLOCK" not in text and "intro" in text and "footer" in text
    assert text.count(R.START) == 1
    for label in ("A 224px", "B 224px", "C 224px"):
        assert label in text
    assert "folds present: [0]" in text
    assert "| B_f0_s1_224 |" in text
    # idempotent
    R.main(["--runs", str(runs), "--doc", str(doc), "--n-boot", "50"])
    assert doc.read_text().count(R.START) == 1


def test_cross_fit_does_not_use_own_fold(tmp_path):
    rng = np.random.default_rng(3)
    folds = {}
    for f in range(5):
        y = rng.integers(0, 5, 80)
        folds[f] = T.summarize_predictions("C", y, {"score": y * 0.7 + rng.normal(0, 0.3, 80)})
    y, pred, *_ , info = R.cross_fit_predictions("C", folds)
    assert len(info["cuts_per_fold"]) == 5
    assert R.M.qwk(y, pred) > 0.7


def test_oof_npz_from_pandas_ids_loads_without_pickle(tmp_path):
    """train.py must save id_code as a str array (object arrays need pickle, report.py refuses it)."""
    import pandas as pd
    ids = pd.Series(["a1", "b2"]).to_numpy().astype(str)
    np.savez_compressed(tmp_path / "o.npz", id_code=ids, y_true=np.array([0, 1]))
    with np.load(tmp_path / "o.npz", allow_pickle=False) as z:
        assert list(z["id_code"]) == ["a1", "b2"]
