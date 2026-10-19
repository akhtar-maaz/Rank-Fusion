# Frozen experiment records

- `lambdamart_cv.json`, `lambdamart_cv_v6.json`: sixth-round complete
  TRAIN tree comparisons, frozen guards, full folds and native equivalence.
- `lambdamart_v6_parts/`: four source five-fold tree reports.
- `neural_selection_v6.json`: all four fixed-epoch five-fold pairwise reports,
  exact control reproduction and frozen mean/win promotion guards.
- `pairwise_refit_v6.json`, `pairwise_reproducibility_v6.json`: full-TRAIN
  neural fitting or exact baseline preservation and independent verification.
- `dev_metrics.json`: final frozen DEV evaluation of current artifacts.
- `artifacts.json`, `artifacts_hpc_v6.json`: hashes and native/staff loading.
- `lambdamart_reproducibility.json`: independent tree refit or exact preservation.
- `execution.json`: complete PBS outcomes, resource use, preflight and checks.
- `robustness_seed_v6.json`: post-selection perturbation sensitivity check,
  without changing the artifact; skipped when the v5 tree is retained.
- `leaderboard_v1.json` through `leaderboard_v5.json`: user-reported aggregate
  feedback, never hidden examples or relevance labels.
- Versioned records through `_v5` preserve earlier checkpoints and sweeps.
- `recipe_selection_v3.json`, `recipe_domains_v3/`, `recipe_budget_v3.json`
  and `recipe_smoke_v3.json`: unchanged C1 TRAIN-derived and synthetic checks.

Local diagnostics do not estimate undisclosed C1/C2 scores. No official
test examples, labels or outside benchmark data are included here.
