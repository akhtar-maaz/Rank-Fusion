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

## Adaptive transfer policy, also frozen before measurement

Separately test a recipe that chooses between two leaf counts on the supplied
validation file, with at most 500 trees and patience 40 per fit. Counts depend
on training query count: 15/31 below 200 queries, 31/63 below 2000, otherwise
63/127. Keep v2's other recipe regularization and query-rank features, changing
LambdaRank truncation from 15 to 13. Exact validation ties prefer fewer leaves.
Both fits use four threads. Fixed iteration limits keep search deterministic.

Compare this policy with the exact v2 recipe on three constructed TRAIN-only
domains (300 queries / 40 features / labels 0–4; 1200 / 90 / 0–4;
4000 / 136 / labels clipped to 0–2). Whole queries and shuffled feature subsets
are sampled with seeds 2026/2027/2028. Features receive fixed positive scale
changes. Hold out outer fold 0 of grouped five-fold seed 0; choose trees and
capacity using inner fold 0 of grouped five-fold seed 1 on the remaining data.
The outer queries never control fitting or early stopping. These three
single-holdout diagnostics are not the required five-fold LambdaMART sweep
above and do not estimate unknown benchmarks. Adopt the new recipe only if
its average outer nDCG exceeds v2's and all portability tests pass. Otherwise
retain the v2 policy. No DEV or hidden dataset is consulted.

## Bounded addition after the first TRAIN fold

The first completed fold suggests complementary errors: normalized features
improve affine robustness but lose clean accuracy. Before any remaining-fold
results are examined, add two equal-weight score averages of candidates 0/1:
400+200 trees and 800+200 trees. Reuse their completed out-of-fold predictions;
compute actual training-fold ensemble scores from saved fold models. Apply
the same clean guard and mixture improvement threshold to all completed rows.
This is an adaptive TRAIN-only extension, not an unbiased new test set.

If selected, combine the constituent numeric trees into one native LightGBM
text model, with half-scaled leaf outputs and feature indices remapped into
raw/rank/min-max blocks. Loading and scoring use only this one artifact.
Tests compare the exported native model with explicit arithmetic averages,
including after save/load and from `scripts/train_model.py`'s configuration
fit helper. No neural model or distillation is involved.
