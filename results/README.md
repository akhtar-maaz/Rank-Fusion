# Frozen experiment records

- `selection_freeze_v9.json`, `v9_*_audit.json`: complete TRAIN-only selection,
  independent checks and hashes committed before inspecting fresh DEV.
- `lambdamart_cv.json`, `lambdamart_cv_v9.json`: complete TRAIN comparisons,
  exact V8 control, frozen promotion gates and five-fold native equivalence.
- `lambdamart_v9_parts/`: completed source tree reports from four planned
  candidates, each with all five folds at fixed 400/800/1200 checkpoints.
  Actual completed counts and whole-candidate exclusions are explicit in
  the selection and execution records.
- `dev_metrics.json`: final DEV run after the frozen TRAIN-only tree choice.
- `artifacts.json`, `artifacts_hpc_v9.json`: native/staff loading and hashes.
- `lambdamart_reproducibility.json`: independent promoted-tree refit or exact
  evaluated V8 copying; `refitted` distinguishes those cases.
- `execution.json`: nine actual PBS records with requested/observed resources,
  protected V8 files/history, synthetic mixed-objective preflight and checks.
- `robustness_seed_v9.json`: descriptive post-selection seed902617 comparison,
  explicitly skipped when the exact V8 tree is retained.
- `neural_selection_v8.json`, `pairwise_refit_v8.json`,
  `pairwise_reproducibility_v8.json`, `neural_history_v8.json`: original V8
  provenance of unchanged neural artifacts. They are not V9 fits or selection.
  Current `neural_history.json` preserves the archived V8 bytes unchanged.
- `leaderboard_v8.json`: latest user-provided aggregate staff feedback;
  versioned earlier leaderboard/experiment records preserve previous rounds.
- `reproducibility_fix_v7_1.json`: preserved graded seeded-training correction,
  confirmed by submission 243's 10/10 staff correctness.
- `recipe_selection_v3.json`, `recipe_domains_v3/`, `recipe_budget_v3.json`
  and `recipe_smoke_v3.json`: unchanged C1 TRAIN-derived and synthetic checks.

Repeated adaptive TRAIN-fold estimates are not fresh held-out evidence.
Public diagnostics neither reconstruct C2 nor identify hidden C1 benchmarks.
Formal report writing remains deferred.
