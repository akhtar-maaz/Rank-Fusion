# Third-round protocol, frozen before running

The v2 website result is retained in `results/leaderboard_v2.json`. This round
keeps fusion and both neural artifacts fixed and uses only Fold1 TRAIN.
The v2 Git commit and ZIP remain recoverable checkpoints.

## Candidates and measurement

Use the same seeded five query-grouped TRAIN folds, fixed tree counts, and
no early stopping on reported folds. `scripts/cv_robust_lambdamart.py`
contains four candidates:

1. The exact v2 control: raw values plus query percentiles, 127 leaves,
   minimum leaf size 100, L2 10, 200/400/800 trees.
2. Replace raw values by within-query min-max values, retaining percentiles;
   all other parameters/checkpoints match the control. This representation
   is invariant to positive affine feature changes in exact arithmetic.
3. Raw plus percentiles, 63 leaves, minimum leaf size 200, L2 20,
   50% per-node feature sampling, path smoothing 20, truncation 13,
   400/800/1200 trees.
4. Raw plus percentiles, 127 leaves, minimum leaf size 100, L2 10,
   random split thresholds, path smoothing 10, truncation 13,
   400/800/1600 trees.

All use learning rate 0.05, tree feature fraction 0.8, row fraction 0.8,
four threads, deterministic training and seed 0. Extra parameters follow
the [LightGBM 4.6 documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html).
No outside ranking implementation is copied.

Evaluate each held-out TRAIN fold clean and under three explicit, label-free
diagnostics. Each query gets deterministic randomness from its opaque ID:

- Positive affine feature changes: scales log-uniform in [0.5, 2]; shifts
  uniform within 25% of each feature's within-query range.
- Feature loss: independently replace 10% of a query's columns by zero.
- Additive Gaussian noise, SD 5% of the within-query feature SD.

These are local diagnostics, not reconstructions or estimates of staff's
undisclosed C2 construction. Missing entries remain missing. No validation
label enters the transformation or training.

## Promotion rule

Among complete five-fold candidates, retain those within 0.0025 clean nDCG
of the highest clean mean (including v2). Select the largest equal mixture
of clean mean and the mean over the three diagnostics. Ties prefer fewer
trees. Require at least 0.001 improvement in this mixture over v2 before
replacing its artifact; otherwise keep v2. This tolerance is half the
public Track B rounding band and bounds the clean-performance tradeoff.

Record all fold scores and discarded runs honestly. Freeze the choice and
refit on full TRAIN before a final DEV evaluation. Website feedback is only
aggregate feedback; no held-out examples or labels are accessed. Report
writing remains deferred.
