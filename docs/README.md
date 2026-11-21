# Docs
Put these here (they are not included in this bundle — copy them from the team's files):
- `DEPI_R5_Project_Documentation_Guidelines.pdf`  (the "Project Documentation Guidelines" 7-step document)
- `DEPI_ML_Projects_Booklet.pdf`                   (the track booklet: "Microsoft Machine Learning Projects")
Included:
- `DR_Project_Proposal.pdf` — the approved proposal (5 milestones).
- `legacy/` — early planning guides written when the team was starting. **Partly outdated**: preprocessing code there was superseded by `src/retinograde_inference.py::preprocess_rgb`; the plans predate the cascade. Use for history only.
Model v2 (Phases 1–3):
- `v2_model_report.md` — **start here**: what was done, what changed, final performance, limitations.
- `experiments.md` — every run and decision, with data versions and the generated results tables.
- `frozen_config.json` — the final model's frozen settings (written before the one-shot test).
- `test_results.md` — the one-shot test (APTOS 733, IDRiD 516, EyePACS 3,428) with confidence intervals.
- `figures_private/` — **gitignored**: galleries derived from competition images (label review only).

Still to create: `model_card.md`, report/diagram sources.
