# Fifth-round protocol, frozen before measurement

The user-provided v4 leaderboard snapshot reports 6th of 21 teams, raw LTR
0.4969, pairwise held-out 0.4333, C2 0.3698 and live score 0.4290/0.65.
C2 increased from v3's 0.3625. The current leader has C2 0.3991. No hidden
examples or labels are available. Preserve v4's Git commit, ZIP and native
model as the fallback. Fusion, both neural artifacts and the C1 recipe stay
fixed; formal report writing remains deferred.

## Four new fits

Use the same five query-grouped TRAIN folds, seed 0, with fixed 200/400/800
tree checkpoints and no early stopping on reported folds. All fits use
permitted LightGBM LambdaMART, learning rate 0.05, row fraction 0.8 with
frequency 1, four threads, deterministic row-wise fitting and seed 0.
Restore truncation 30: v4's complete comparison found truncation 13 alone
weaker than the earlier control. Exact configurations are frozen in
`configs/v5_candidates.json`:

1. Mixed augmentation, raw values plus query percentiles; 127 leaves,
   minimum leaf size 100, L2 10 and tree feature fraction 0.8.
2. The same fit with all three feature blocks: raw, percentile and query
   min-max values. Existing native feature metadata supports this layout.
3. V4 dropout/noise augmentation, raw plus percentiles; 63 leaves, minimum
   leaf size 200, L2 20 and tree feature fraction 0.5.
4. The third fit with min-max values plus percentiles instead of raw values.

Mixed augmentation keeps every original training query and appends one copy
of about half. SHA256 of `augment:20261020:<qid>` selects copying by even
first byte and affine/dropout/noise by second byte modulo 3. Transforms are
the existing public diagnostics, with training seed 20261020, distinct from
diagnostic seed 1729. The v4 augmentation and seed 20261019 are unchanged.
Labels are copied without inspection; copies have distinct group IDs.
Augmentation runs after splitting TRAIN. All feature computations are local
to one query. No fitting statistic is shared across training and validation.

## Fixed averages and promotion

Reuse the complete v3/v4 fold models and predictions. Include six cached
averages: v3's equal 800-rank/200-normalized scorer with v4's augmented rank
scorer at 400/800 trees, with augmented weights 0.25/0.5/0.75. This includes
the exact v4 control (800 augmented trees, weight 0.5).

Also evaluate all 12 new single-model checkpoints and fixed averages of
each with v4 at new-model weights 0.25/0.5/0.75. This is 54 declared settings:
6 cached, 12 single models and 36 new/v4 averages. Flatten positive weights
when exporting one ordinary native LightGBM model. No continuous weight
fitting, per-query choice or use of website labels is allowed.

Among complete settings within 0.0025 clean nDCG of the highest complete
mean, require that no individual diagnostic falls more than 0.0025 below
v4's corresponding value. Maximize the existing equal mixture of clean
mean and the mean of affine/dropout/noise diagnostics. Exact ties prefer
fewer total trees. Require a mixture gain of at least 0.0005 over v4 for
promotion; otherwise retain v4. This added guard seeks a balanced gain
after v4 traded affine performance for noise and feature-loss resilience.

Record all fold training/validation metrics and explicit runtime exclusions.
Incomplete candidates are excluded entirely, never compared by partial means.
Freeze selection before full-TRAIN fitting and one final DEV evaluation.
Verify native export, independent refitting, submission behavior and time
limits. These reused-fold adaptive selection estimates are not an unbiased
new held-out test, and the public diagnostics do not identify staff's C2
construction or establish a hidden-score gain.

## Post-selection stress measurement

After freezing a promoted choice, compare it with v4 on the same TRAIN
validation folds under perturbation seed 271828. This seed differs from
the selection and augmentation seeds. Run all three diagnostics, report
every fold and both means, and do not use this check to change the artifact
or promotion decision. It measures sensitivity to perturbation randomness;
reusing validation queries means it is not a new held-out label test.

## Completed outcome

All four new candidates complete every fold. The result contains 54 complete settings.
The selected `v4_plus_1_800_weight0.5` has clean mean 0.487398492,
diagnostic mean 0.474074548 and mixture
0.480736520, versus v4's 0.478149029.
Choice is frozen before full-TRAIN fitting and the final DEV evaluation.
Independent full-TRAIN fits are byte-identical. All 91 tests and actual
submission-interface/runtime checks pass. Fusion, neural artifacts and the
C1 recipe preserve v4 hashes; v4 remains a recoverable Git/ZIP checkpoint.

The post-selection check with perturbation seed 271828 improves all three
diagnostics. Their mean changes from 0.470960962 to 0.474550300; full fold
results are in `results/robustness_seed_v5.json`. This measurement does not
change the frozen artifact or promotion decision.
