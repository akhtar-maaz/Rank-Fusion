# Frozen experiment records

These are raw measurements for the later report, not leaderboard percentiles.

* `lambdamart_cv.json`: all 34 configurations evaluated on the same five
  query-grouped folds within Fold1 TRAIN, seed 0. The first 32 settings combine
  four tree counts with eight regularization configurations; the final two
  extend the initial winner to 800 and 1200 trees. No reported fold controls
  early stopping. Each row retains all five scores, their sample standard
  deviation, training mean, and validation/training ratio.
* `neural_history.json`: stopping-epoch selection on internal TRAIN fold 0,
  followed by fixed-epoch refits on all TRAIN. The helper's field name
  `dev_ndcg@10` refers to that internal TRAIN selection fold. It does not
  refer to the released DEV split. Final refits use their last epoch.
* `fusion_selection.json`: a predeclared set of seven fusion settings compared
  on TRAIN. Selection uses 0.8 times raw nDCG@10 plus 0.2 times raw MAP@10 as a
  local proxy; actual class percentiles are unavailable locally.
* `dev_metrics.json`: final evaluation after model configurations are frozen,
  through the ranking interface and pristine staff-compatible neural loader.
  Includes the BM25 floor, candidate-contract checks, and per-call timings.
* `lambdamart_reproducibility.json`: SHA-256 comparison against an independent
  fit using the committed selected CV configuration.
* `artifacts.json`: sizes, SHA-256 hashes checked against the HPC copies,
  neural architecture, and Python 3.12 artifact-load verification.
* `execution.json`: successful PBS job outcomes and synthetic transfer checks.

Official Fold1 TRAIN and DEV are the only real dataset partitions used.
The held-out partition and other folds were never opened. Raw data and local
environments are excluded from Git and the submission ZIP.
