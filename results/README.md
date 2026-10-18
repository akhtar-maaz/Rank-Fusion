# Frozen experiment records

- `lambdamart_cv.json`, `lambdamart_cv_v4.json`: the 36 complete fourth-round
  TRAIN-only settings, frozen promotion rule, selected members, all fold
  metrics and native-export equivalence checks.
- `dev_metrics.json`: final frozen DEV evaluation of the current artifacts.
- `artifacts.json`: local Python 3.12 loading, hashes, determinism and HPC checks.
- `lambdamart_reproducibility.json`: independent byte-identical full-TRAIN refit.
- `execution.json`: PBS job outcomes, resource use and verification.
- `leaderboard_v1.json`, `leaderboard_v2.json`, `leaderboard_v3.json`:
  aggregate feedback supplied by the user.
- `_v1`, `_v2` and `_v3` records preserve earlier checkpoints. Fusion and
  neural selections remain v2; the C1 recipe and its diagnostics remain v3.
- `recipe_selection_v3.json`, `recipe_domains_v3/`: TRAIN-derived nested diagnostics.
- `recipe_budget_v3.json`: deterministic wide budget check; the initial check
  remains in `recipe_budget_initial_v3.json`.
- `recipe_smoke_v3.json`: synthetic arbitrary-width portability checks.

Local diagnostics do not estimate undisclosed C1/C2 scores. No held-out
examples, labels or outside benchmark data are included here.
