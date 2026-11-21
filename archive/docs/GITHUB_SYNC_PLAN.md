> **Historical document.** Executed as PR #1 on 2026-11-22. The paths below describe the repository *before* the v2 clean-up that moved the team notebooks to `notebooks/exploration/` and v1 files to `archive/v1/` (see `archive/README.md`).

# GitHub sync plan: local project → `OmarAshour1/Diabetic-Retinopathy-Project`

**In short:** the GitHub repo and this local project are **two separate copies with no shared history**, and only one file (`README.md`) exists in both. The plan adds the new work on a branch the sync branch, merges the two READMEs, and leaves every teammate file on GitHub exactly where it is. Nothing is deleted. Nothing is pushed, and no PR is opened, until you approve. A few decisions are needed from you (section 6).

Inspected 2026-11-22 (read-only: `git ls-remote`, `git fetch`, public GitHub API).

## 1. What is on GitHub now

| Item | Finding |
|---|---|
| Visibility | **Public** (`private: false`). Owner `OmarAshour1`. |
| Branches | Only `main` (default), at `1d5fb16`. |
| Open pull requests | **0** |
| History | 16 commits, all by `omar-khaled67` (web uploads), 2026-10-04 → 2026-11-01. |
| Shared history with local | **None** (`git merge-base` finds no common commit). The local project has 44 commits by `OmarAshour1`. |
| Write access | Very likely: the local commits use the owner account `OmarAshour1`. Confirmed at push time by the browser sign-in. |
| GitHub CLI (`gh`) | **Not installed** on this laptop, so a PR can't be opened from the command line yet (see decision D5). |

Remote file tree (17 files):

| Path | Size | Content |
|---|---|---|
| `README.md` | 0.9 KB | Project overview (quality check → grading → confidence → Grad-CAM) |
| `Diabetic Retinopathy Project — Task Board.md` | 0.1 KB | Link to the Notion task board |
| `Datasets/APTOS 2019 Blindness Detection Dataset.md` | 0.1 KB | Link to the Kaggle competition |
| `Datasets/Diabetic Retinopathy 2015 Data Colored Resized.md` | 0.1 KB | Link to the EyePACS mirror |
| `APSOT Dataset/APTOS Dataset Preparation and Image Preprocessing/dataset-preparation-and-image-preprocessing (1).ipynb` | 14.5 MB | Team notebook: APTOS preparation (1 image output) |
| `APSOT Dataset/Traning Models/one stage train/01…06_*.ipynb` (6 files) | 12–150 KB | Team notebooks: one-stage experiments (4 image outputs in total) |
| `APSOT Dataset/Traning Models/tow stages train/stage 1/Binary classification (DR - NO DR).ipynb` | 16 KB | Team notebook: Stage 1 |
| `APSOT Dataset/Traning Models/tow stages train/stage 1/Models/best_binary_dr_model (2).keras` | 17.1 MB | v1 Stage 1 model file (already public) |
| `Dataset 2015/Dataset_2015 Preparation/01…03_*.ipynb` + empty `README.md` | 1.3–3.7 MB | Team notebooks: EyePACS 2015 audit / validation / quality (12 image outputs in total) |

## 2. Classification of every path

### Remote-only (on GitHub, not local): **keep, unchanged, same paths**
| Path | Proposal | Reason |
|---|---|---|
| All 11 team notebooks under `APSOT Dataset/` and `Dataset 2015/` | **keep in place** | The team's own work and history, which each member presents. Not replaced: the new code in `src/` is a separate pipeline, not a rewrite of these notebooks. |
| `…/Models/best_binary_dr_model (2).keras` (17 MB) | **keep in place** | v1 Stage 1 model, already public; under 50 MB. Deleting it would not remove it from history anyway. |
| `Datasets/*.md` (2 links) | **keep** | Tiny and still correct. `data/README.md` adds the full data table. |
| `Diabetic Retinopathy Project — Task Board.md` | **keep** | Link already public. |
| `Dataset 2015/Dataset_2015 Preparation/README.md` (empty) | **keep** | Teammate's placeholder. |

No remote file is moved to `archive/` or deleted.

### Modified (exists on both sides, content differs): **merge**
| Path | Proposal |
|---|---|
| `README.md` | Replace with the new README, **merged** with the team's overview text (the 5-step pipeline description is kept as the "Project overview" section). Adds a "Repository map" that points to the team notebooks as well as to `src/`. |

### New (only local): **add**
| Group | Files | Content |
|---|---|---|
| Root | `.gitignore`, `requirements.txt`, `requirements-local.txt`, `pytest.ini`, the two local AI working files | Environment pins, test config, project memory for AI-assisted work (see D3). |
| `src/` | 27 files | Inference API (team's `retinograde_inference.py`); team scripts `training/stage1_binary_train.py`, `stage2_severity_cascade.py`, `eyepacs_pretrain_stage2.py`, `preprocessing/eyepacs_prepare.py` (unchanged since they were added); new `training/train.py`, `training/predict.py`, `data/aptos_v2.py`, `model_io.py`, `explain/gradcam.py`, and 16 files in `evaluation/`. |
| `tests/` | 3 files | 30 local tests (no TensorFlow, synthetic data). |
| `kaggle/` | 13 files | `make_kernel.py`, `README.md`, 11 private Kaggle job scripts. |
| `notebooks/` | `colab_final_model.ipynb` | How the final model was trained on Colab (no outputs, no token). |
| `docs/` | 9 files | `v2_model_report.md`, `experiments.md`, `test_results.md`, `frozen_config.json`, `README.md`, `DR_Project_Proposal.pdf`, `legacy/` (3 early plans), and this plan. |
| `data/`, `models/`, `notion/` | README files (+ 2 Notion CSV exports, see D3) | Where data and models live; no data or models are committed. |

### Superseded
None. No remote file is replaced by a file at a different path.

### Possible conflicts
- No other branches and no open PRs. The last remote change was 2026-11-01 (3 weeks ago).
- The only overlapping path is `README.md`, handled by the merge above.

## 3. Safety checks (run 2026-11-22 over everything that would be added)

| Check | Result |
|---|---|
| Secret scan (Kaggle/GitHub/AWS/OpenAI-style tokens, private keys, `kaggle.json` keys) over all local files including notebooks | **none found** |
| Forbidden files (`kaggle.json`, `.env`, `access_token`, `credentials.json`, `.npy/.npz/.keras/.h5`, images, archives, `docs/figures_private/`, `mlruns/`) | **none tracked** |
| Files over 50 MB | **none** (largest local file 31 KB) |
| Datasets or competition images | **none** (data stays on Kaggle; only aggregate numbers and file IDs in docs) |
| `.gitignore` covers `kaggle.json`, `.env`, `.kaggle/`, `*.npy`, `*.npz`, `*.h5`, `models/*.keras`, `data/*`, `kaggle/runs/`, `kaggle/_build/`, `docs/figures_private/`, `mlruns/` | yes |
| `.gitignore` gaps | `*.keras` outside `models/`, and a stray `credentials.json` / `access_token` in the repo. **Proposed:** add `*.keras`, `credentials.json`, `access_token`. Files already tracked on GitHub, such as the 17 MB v1 model, are not affected. |
| v2 model file (81 MB) | **Not committed.** It stays in the private Kaggle dataset `rg-final-model`. Optional: attach it to a GitHub **Release** (D4). |

**Pre-existing issue on GitHub (not caused by this sync):** the team notebooks contain 17 image outputs in total. Some may be sample fundus photos from APTOS / EyePACS, which those datasets' rules don't allow redistributing. The owners of those notebooks should check them and, if needed, clear the outputs in a separate PR. This plan does not touch them.

## 4. How the files get there: two options

| | **A. Snapshot (recommended)** | B. Merge full history |
|---|---|---|
| How | Branch the sync branch from `team/main`; copy the local files in as ~6 new commits. | Branch from `team/main`; `git merge --allow-unrelated-histories` the 44 local commits. |
| Personal data | New commits use your GitHub no-reply address `250317583+OmarAshour1@users.noreply.github.com`. | Publishes **your personal email address** on 43 commits in a **public** repo. Push may be rejected if your GitHub email privacy setting blocks it. |
| Can leave files out | yes (e.g. the Notion CSVs) | no. Anything in the 44 commits stays in public history even if deleted later. |
| Provenance | The commit IDs quoted in `docs/` (e.g. freeze `86f7381`, code `7d93e26`) refer to the development history. That history stays in this local folder and is **tagged `v2-dev-history`**; a note in `docs/experiments.md` says so. | Commit IDs in `docs/` resolve on GitHub. |
| History rewrite / force push | none | none |

Commits planned for option A (author `OmarAshour1 <250317583+OmarAshour1@users.noreply.github.com>`):
1. `chore: Python environment, test config and .gitignore`: `.gitignore`, `requirements*.txt`, `pytest.ini`
2. `feat: v2 data, training, evaluation and Grad-CAM code`: `src/`, `tests/`
3. `feat: private Kaggle job tooling`: `kaggle/`
4. `docs: v2 model report, experiment log, frozen config, one-shot test results`: `docs/`, `data/README.md`, `models/README.md`, `notion/README.md`
5. `feat: Colab notebook used to train the final model`: `notebooks/`
6. `docs: merged README and project memory`: `README.md`, the two local AI working files (per D3)

Then: run `pytest` in a fresh clone of the branch, and check that the README quick start works.

## 5. After the push
- Open a PR from the sync branch to `main` with: summary, what changed, how to run and test, what teammates must do. **Not merged** by me.
- Teammates after the merge: `git pull` (or re-download). Anyone with local copies of the notebooks keeps working on the same paths. Nothing they have was moved.

## 6. Decisions (answered by the user on 2026-11-22)

| # | Decision |
|---|---|
| D1 | **Option A, snapshot.** The 44 development commits stay local, tagged `v2-dev-history`; new commits use the GitHub no-reply address. |
| D2 | `.gitignore` extended: `*.keras`, `credentials.json`, `access_token`, and the local-only AI working files. |
| D3 | **Included:** `docs/DR_Project_Proposal.pdf`, `notion/*.csv`. **Not published:** the two local AI working files; the 7 references to them in docs and code were reworded so nothing dangles. |
| D4 | v2 model stays in the private Kaggle dataset `rg-final-model` (no GitHub Release). |
| D5 | PR opened by the user from a prefilled github.com link (GitHub CLI not installed). |

Planned commit 6 therefore contains only the merged `README.md`.

### Original questions
| # | Question | My recommendation |
|---|---|---|
| D1 | Option **A (snapshot)** or **B (full history)**? | **A**: the repo is public, and B would publish your personal email on 43 commits. |
| D2 | Add `*.keras`, `credentials.json`, `access_token` to `.gitignore`? | Yes. |
| D3 | Include two local AI working files (project memory and kickoff prompt, with team names and plans), `notion/*.csv` (task-board export: names, statuses, private Notion links), `docs/DR_Project_Proposal.pdf`? | Include the local AI project-memory file and the proposal. **Leave out `notion/*.csv`**: Notion is the source of truth, and the CSV goes stale. the kickoff prompt: your choice (harmless, low value). |
| D4 | Publish the v2 model (81 MB) as a GitHub Release asset? | Optional; it's fine to keep it only on Kaggle for now. |
| D5 | PR creation: install the GitHub CLI (`winget install GitHub.cli`, then `gh auth login` in your browser), **or** I push the branch and give you a one-click link to open the PR on github.com? | The link: no install needed, same result. |
