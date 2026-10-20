# Frozen experiment records

- `lambdamart_cv.json`, `lambdamart_cv_v7.json`: complete TRAIN tree
  comparisons, exact v6 control, all promotion gates and native equivalence.
- `lambdamart_v7_parts/`: four complete source TRAIN-fold tree reports.
- `neural_selection_v7.json`: three fixed seed candidates, all five folds,
  exported-artifact metrics, exact v6 control and frozen promotion checks.
- `pairwise_refit_v7.json`, `pairwise_reproducibility_v7.json`: promoted
  member histories/plain-MLP compilation or exact v6 preservation, plus
  independent reproduction. Refit uses `candidate` and member records.
- `dev_metrics.json`: one final DEV evaluation after both choices freeze.
- `artifacts.json`, `artifacts_hpc_v7.json`: native/staff loading and hashes.
- `lambdamart_reproducibility.json`: independent tree fit or exact fallback.
- `execution.json`: ten completed actual PBS records, requests and observed
  resources, synthetic preflights, final verification and protected files.
- `robustness_seed_v7.json`: descriptive post-selection seed391729 check;
  explicitly skipped if the exact v6 tree is retained.
- `leaderboard_v1.json` through `leaderboard_v6.json`: user-provided
  aggregate staff feedback, without hidden examples or relevance labels.
- Versioned records through `_v6` preserve earlier fits and checkpoints.
- `recipe_selection_v3.json`, `recipe_domains_v3/`, `recipe_budget_v3.json`
  and `recipe_smoke_v3.json`: unchanged C1 TRAIN-derived/synthetic checks.

Repeated adaptive TRAIN-fold estimates are not fresh held-out estimates.
Public diagnostics do not reconstruct staff C2 or identify C1 benchmarks.
Formal report writing remains deferred.
