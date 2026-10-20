# Frozen experiment records

- `selection_freeze_v8.json` and `v8_*_audit.json`: TRAIN-only choices,
  independent checks and file hashes recorded before inspecting fresh DEV.
- `lambdamart_cv.json`, `lambdamart_cv_v8.json`: complete TRAIN tree
  comparisons, exact v7 control, all promotion gates and native equivalence.
- `lambdamart_v8_parts/`: four complete source TRAIN-fold tree reports.
- `neural_selection_v8.json`: five fixed candidates and 45 member fits,
  all five folds, exact compiled v7 control, source/runtime/cache identities,
  exported-artifact metrics and frozen promotion checks.
- `pairwise_refit_v8.json`, `pairwise_reproducibility_v8.json`: promoted
  member histories/plain-MLP compilation or exact v7 preservation, plus
  independent reproduction. Refit uses `candidate` and member records.
- `dev_metrics.json`: one final DEV evaluation after both choices freeze.
- `artifacts.json`, `artifacts_hpc_v8.json`: native/staff loading and hashes.
- `lambdamart_reproducibility.json`: independent tree fit or exact fallback.
- `execution.json`: ten completed actual PBS records, requests and observed
  resources, synthetic preflights, final verification and protected files.
- `robustness_seed_v8.json`: descriptive post-selection seed615019 check;
  explicitly skipped if the exact v7 tree is retained.
- Versioned `leaderboard_*.json` records through `leaderboard_v7_1.json`:
  user-provided aggregate staff feedback without hidden examples or labels.
- `reproducibility_fix_v7_1.json`: preserved graded seeded-training correction
  and regressions, confirmed by submission 243's 10/10 staff correctness.
- Versioned records through `_v7_1` preserve earlier fits and checkpoints.
- `recipe_selection_v3.json`, `recipe_domains_v3/`, `recipe_budget_v3.json`
  and `recipe_smoke_v3.json`: unchanged C1 TRAIN-derived/synthetic checks.

Repeated adaptive TRAIN-fold estimates are not fresh held-out estimates.
Public diagnostics do not reconstruct staff C2 or identify C1 benchmarks.
Formal report writing remains deferred.
