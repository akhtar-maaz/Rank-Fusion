# Frozen experiment records

- `lambdamart_cv.json`, `lambdamart_cv_v5.json`: complete fifth-round TRAIN
  comparisons, frozen guards and promotion rule, member specifications, all
  fold metrics and native-export equivalence checks.
- `lambdamart_v5_parts/`: source five-fold reports for the new candidates.
- `dev_metrics.json`: final frozen DEV evaluation of the current artifacts.
- `artifacts.json`: local Python 3.12 loading, hashes, determinism and HPC checks.
- `lambdamart_reproducibility.json`: independent byte-identical full-TRAIN refit.
- `execution.json`: PBS outcomes, resource use, preflight and verification.
- `robustness_seed_v5.json`: post-selection TRAIN-fold diagnostics with seed
  271828, reporting both configurations without changing the chosen artifact.
- `leaderboard_v1.json` through `leaderboard_v4.json`: user-provided aggregate
  website feedback, with no hidden examples or labels.
- Versioned `_v1` through `_v4` records preserve earlier checkpoints.
  Fusion and neural choices remain v2; the C1 recipe and its diagnostics
  remain v3. Earlier source reports remain in `lambdamart_v4_parts/`.
- `recipe_selection_v3.json`, `recipe_domains_v3/`: TRAIN-only nested diagnostics.
- `recipe_budget_v3.json`: deterministic wide compute check; the initial check
  remains in `recipe_budget_initial_v3.json`.
- `recipe_smoke_v3.json`: synthetic arbitrary-width portability checks.

Local diagnostics do not estimate undisclosed C1/C2 scores. No held-out
examples, labels or outside benchmark data are included here.
