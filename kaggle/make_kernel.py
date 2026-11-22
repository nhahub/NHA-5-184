"""
Kaggle job helper (local, no compute). Everything heavy runs on Kaggle; this only writes folders
and wraps the official `kaggle` CLI.

  python kaggle/make_kernel.py code                       # stage src/ -> kaggle/_build/code (private dataset)
  python kaggle/make_kernel.py build rg-recon             # write kaggle/_build/kernels/rg-recon/
  python kaggle/make_kernel.py build rg-train --suffix b-s42 --config '{"candidate":"B","folds":[0,1,2,3,4]}'
  python kaggle/make_kernel.py push rg-recon              # kaggle kernels push (asks y/N first)
  python kaggle/make_kernel.py status rg-recon
  python kaggle/make_kernel.py pull rg-recon              # small outputs only -> kaggle/runs/rg-recon/

Nothing is pushed without an explicit "y" on the terminal (or --yes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "kaggle", "_build")
RUNS = os.path.join(ROOT, "kaggle", "runs")
KERNEL_SRC = os.path.join(ROOT, "kaggle", "kernels")
PRIVATE_FIGS = os.path.join(ROOT, "docs", "figures_private")      # gitignored: figures derived from competition images

USER = "omarashour206"
CODE_DS = f"{USER}/retinograde-code"
APTOS = "aptos2019-blindness-detection"
APTOS_PRE = f"{USER}/aptos2019-preprocessed-dataset"
V1_MODELS = f"{USER}/aptos-stage1-models"          # best_binary_dr_model.keras (=stage1_v1) + best_severity_model.keras
EYEPACS = "sovitrath/diabetic-retinopathy-2015-data-colored-resized"   # pre-squeezed to 224x224
EYEPACS_HIRES = "tanlikesmath/diabetic-retinopathy-resized"           # ~1024px, aspect ratio kept
FINAL_MODEL = f"{USER}/rg-final-model"                                # frozen v2 model + its EyePACS dev predictions
IDRID = "aaryapatel98/indian-diabetic-retinopathy-image-dataset"   # official layout incl. lesion masks

# Only these (small) files are ever downloaded from kernel outputs.
SMALL_OUTPUTS = r".*(run\.json|latency\.json|eyepacs_dev_preds\.npz|eyepacs_dev_scores\.npz|final_test_preds\.npz|oof_preds\.npz|test_preds\.npz|v1_repro\.json|val_probs_v1\.npz|recon\.json|_summary\.json|\.csv|pip_freeze\.txt|\.md|\.log)$"

SPECS = {
    "rg-recon": dict(script="rg_recon.py", gpu=False, internet=True,
                     competitions=[APTOS], datasets=[CODE_DS, APTOS_PRE, EYEPACS, IDRID]),
    "rg-v1-repro": dict(script="rg_v1_repro.py", gpu=True, internet=False,
                        datasets=[CODE_DS, APTOS_PRE, V1_MODELS]),
    "rg-data-v2": dict(script="rg_data_v2.py", gpu=False, internet=False,
                       competitions=[APTOS], datasets=[CODE_DS, IDRID], kernels=[f"{USER}/rg-recon"]),
    "rg-train": dict(script="rg_train.py", gpu=True, internet=True,      # internet: ImageNet weights
                     datasets=[CODE_DS], kernels=[f"{USER}/rg-data-v2"]),
    # same trainer, reading the 384px arrays (build rg-data-v2 with --suffix 384 --config '{"img":384}')
    "rg-train-384": dict(script="rg_train.py", gpu=True, internet=True,
                         datasets=[CODE_DS], kernels=[f"{USER}/rg-data-v2-384"]),
    # headroom check 1+2: near-duplicates across folds + image stats + wrong-case gallery (train pool only)
    "rg-headroom": dict(script="rg_headroom.py", gpu=False, internet=True, competitions=[APTOS], datasets=[CODE_DS],
                        kernels=[f"{USER}/rg-data-v2"] + [f"{USER}/rg-train-b-s{s}" for s in (42, 7, 1234)]),
    # headroom check 4: EyePACS dev patients, prediction only (CPU); test patients listed, never opened
    "rg-eyepacs-dev": dict(script="rg_eyepacs_dev.py", gpu=False, internet=False, datasets=[CODE_DS, EYEPACS, EYEPACS_HIRES],
                           kernels=[f"{USER}/rg-recon", f"{USER}/rg-train-b-s42", f"{USER}/rg-train-384-b-pilot"]),
    # EyePACS rest (training) + dev (scoring only) arrays for mixed training; test patients never opened
    "rg-eyepacs-data": dict(script="rg_eyepacs_data.py", gpu=False, internet=False, datasets=[CODE_DS, EYEPACS_HIRES],
                            kernels=[f"{USER}/rg-recon", f"{USER}/rg-eyepacs-dev"]),
    # mixed APTOS + EyePACS training at 384 (also scores EyePACS dev patients at the end of each fold)
    "rg-train-mix": dict(script="rg_train.py", gpu=True, internet=True, datasets=[CODE_DS],
                         kernels=[f"{USER}/rg-data-v2-384", f"{USER}/rg-eyepacs-data"]),
    # ONE-SHOT test evaluation (APTOS test, IDRiD, EyePACS test patients); model kernels come from the config
    "rg-final-test": dict(script="rg_final_test.py", gpu=True, internet=False, datasets=[CODE_DS, EYEPACS_HIRES],
                          kernels=[f"{USER}/rg-data-v2-384", f"{USER}/rg-eyepacs-dev", f"{USER}/rg-recon"]),
    # score trained models on EyePACS dev patients (CPU, prediction only); model kernels come from the config
    "rg-score-dev": dict(script="rg_score_dev.py", gpu=False, internet=False, datasets=[CODE_DS],
                         kernels=[f"{USER}/rg-eyepacs-data"]),
    # same one-shot test on CPU (used when the GPU queue is stuck; ~1.5 h)
    "rg-final-test-cpu": dict(script="rg_final_test.py", gpu=False, internet=False, datasets=[CODE_DS, EYEPACS_HIRES],
                              kernels=[f"{USER}/rg-data-v2-384", f"{USER}/rg-eyepacs-dev", f"{USER}/rg-recon"]),
    # end-to-end check of the v2 inference code with the real model (CPU; train pool + EyePACS dev only)
    "rg-infer-check": dict(script="rg_infer_check.py", gpu=False, internet=False, competitions=[APTOS],
                           datasets=[CODE_DS, EYEPACS_HIRES, FINAL_MODEL], kernels=[f"{USER}/rg-data-v2-384"]),
    # CPU latency of the deployed path at 224 vs 384 (no GPU, no internet)
    "rg-latency": dict(script="rg_latency.py", gpu=False, internet=False, competitions=[APTOS], datasets=[CODE_DS],
                       kernels=[f"{USER}/rg-data-v2", f"{USER}/rg-train-b-s42", f"{USER}/rg-train-384-b-pilot"]),
    # re-scoring saved models; no internet (weights come from the rg-train-* kernel outputs)
    "rg-predict": dict(script="rg_predict.py", gpu=True, internet=False, datasets=[CODE_DS],
                       kernels=[f"{USER}/rg-data-v2"] + [f"{USER}/rg-train-b-s{s}" for s in (42, 7, 1234)]),
}


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def _tree_hash(folder: str) -> str:
    h = hashlib.sha256()
    for dp, dns, fns in sorted(os.walk(folder)):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for fn in sorted(fns):
            if fn.endswith(".pyc"):
                continue
            p = os.path.join(dp, fn)
            h.update(os.path.relpath(p, folder).replace("\\", "/").encode())
            with open(p, "rb") as f:
                h.update(f.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def stage_code() -> str:
    out = os.path.join(BUILD, "code")
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, "src"), os.path.join(out, "src"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    sha = _git("rev-parse", "--short", "HEAD") or "nogit"
    dirty = "-dirty" if _git("status", "--porcelain", "--", "src") else ""
    version = f"{sha}{dirty} tree:{_tree_hash(os.path.join(out, 'src'))}"
    with open(os.path.join(out, "CODE_VERSION.txt"), "w") as f:
        f.write(version + "\n")
    with open(os.path.join(out, "dataset-metadata.json"), "w") as f:
        json.dump({"title": "retinograde-code", "id": CODE_DS, "licenses": [{"name": "other"}]}, f, indent=2)
    print(f"staged {out}  ({version})")
    print("first time :  cd kaggle/_build/code && kaggle datasets create -p . -r zip   (already done 2026-11-19)")
    print(f"afterwards :  cd kaggle/_build/code && kaggle datasets version -p . -r zip -m \"{version}\"")
    return version


def kernel_slug(name: str, suffix: str | None) -> str:
    return f"{name}-{suffix}" if suffix else name


def build(name: str, suffix: str | None = None, config: dict | None = None) -> str:
    spec = SPECS[name]
    slug = kernel_slug(name, suffix)
    out = os.path.join(BUILD, "kernels", slug)
    os.makedirs(out, exist_ok=True)
    src = open(os.path.join(KERNEL_SRC, spec["script"]), encoding="utf-8").read()
    if config is not None:
        marker = "CONFIG = {}  # <- injected by make_kernel.py"
        assert marker in src, f"{spec['script']} has no CONFIG marker"
        src = src.replace(marker, f"CONFIG = {config!r}  # injected by make_kernel.py (Python literal)")
    with open(os.path.join(out, spec["script"]), "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    meta = {
        "id": f"{USER}/{slug}", "title": slug, "code_file": spec["script"],
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": spec["gpu"], "enable_tpu": False, "enable_internet": spec["internet"],
        "dataset_sources": spec.get("datasets", []) + ((config or {}).get("model_datasets", [])
                                                       if isinstance(config, dict) else []),
        "competition_sources": spec.get("competitions", []),
        "kernel_sources": spec.get("kernels", []) + [f"{USER}/{k}" for k in
                                                     ((config or {}).get("model_kernels", []) if isinstance(config, dict) else [])],
        "model_sources": [],
    }
    with open(os.path.join(out, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"built {out}  (gpu={spec['gpu']}, sources={meta['dataset_sources'] + meta['kernel_sources']})")
    return out


def _run(cmd, yes: bool) -> int:
    print("$ " + " ".join(cmd))
    if not yes and input("run this? [y/N] ").strip().lower() != "y":
        print("skipped")
        return 1
    return subprocess.call(cmd, cwd=ROOT)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["code", "build", "push", "status", "pull"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--suffix")
    ap.add_argument("--config", help="JSON overrides for rg-train (src/training/train.py DEFAULTS)")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--timeout", type=int, help="push: hard runtime cap in seconds (budget guard)")
    ap.add_argument("--gallery", action="store_true", help="pull: also fetch gallery/*.png into docs/figures_private/")
    ap.add_argument("--accelerator", default="NvidiaTeslaT4",
                    help="push: GPU type; 'default' lets Kaggle pick any free GPU (useful when the T4 queue is stuck)")
    a = ap.parse_args(argv)

    if a.action == "code":
        stage_code()
        return
    if not a.name:
        ap.error("kernel name required")
    slug = kernel_slug(a.name, a.suffix)
    if a.action == "build":
        build(a.name, a.suffix, json.loads(a.config) if a.config else None)
    elif a.action == "push":
        folder = os.path.join(BUILD, "kernels", slug)
        cmd = ["kaggle", "kernels", "push", "-p", folder]
        if SPECS[a.name]["gpu"] and a.accelerator != "default":
            cmd += ["--accelerator", a.accelerator]
        if a.timeout:
            cmd += ["-t", str(a.timeout)]
        sys.exit(_run(cmd, a.yes))
    elif a.action == "status":
        subprocess.call(["kaggle", "kernels", "status", f"{USER}/{slug}"])
    elif a.action == "pull":
        dest = os.path.join(RUNS, slug)
        os.makedirs(dest, exist_ok=True)
        subprocess.call(["kaggle", "kernels", "output", f"{USER}/{slug}", "-p", dest, "-o", "--page-size", "200",
                         "--file-pattern", SMALL_OUTPUTS])
        print("pulled small outputs ->", dest)
        if a.gallery:   # images derived from competition data -> gitignored private folder only
            gdest = os.path.join(PRIVATE_FIGS, slug)
            os.makedirs(gdest, exist_ok=True)
            subprocess.call(["kaggle", "kernels", "output", f"{USER}/{slug}", "-p", gdest, "-o", "--page-size", "200",
                             "--file-pattern", r".*gallery/.*\.png$"])
            print("pulled gallery (private, gitignored) ->", gdest)


if __name__ == "__main__":
    main()
