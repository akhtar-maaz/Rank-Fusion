# Frozen experiment records

- `lambdamart_cv.json`: v3 complete five-fold settings, fixed promotion rule,
  selected native-ensemble members, all fold scores and explicit runtime exclusions.
- `dev_metrics.json`: one final frozen evaluation of the current artifacts.
- `artifacts.json`: local Python 3.12 loading, hashes, determinism and HPC-copy checks.
- `lambdamart_reproducibility.json`: independent byte-identical full-TRAIN refit.
- `recipe_selection_v3.json`, `recipe_domains_v3/`: TRAIN-only adaptive-policy diagnostics.
- `recipe_budget_v3.json`: final deterministic-budget wide compute check;
  `recipe_budget_initial_v3.json` retains the initial 500-tree-ceiling check.
- `recipe_smoke_v3.json`: final synthetic arbitrary-width portability checks.
- `execution.json`: completed and runtime-cancelled PBS jobs, limits and verification.
- `leaderboard_v1.json`, `leaderboard_v2.json`: aggregate results supplied by the user.
- `_v1` and `_v2` records preserve earlier checkpoints; fusion and neural selections
  remain their v2 versions. No new neural or fusion tuning occurs in v3.

Local diagnostics and synthetic sets do not estimate undisclosed C1/C2 scores.
No held-out examples, labels or outside benchmark data are included here.
