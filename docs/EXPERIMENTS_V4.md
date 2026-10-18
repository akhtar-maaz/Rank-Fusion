# Fourth-round protocol, frozen before measurement

The user reports v3 at rank 7/15, with LTR 0.4942, held-out pairwise 0.4333,
C2 0.3625 and live score 0.4093/0.65. The leading entry has LTR 0.5676,
pairwise 0.4489 and C2 0.3884. Only these aggregate website numbers were
provided; no hidden rows or labels are available. Preserve v3 as a fallback.
Fusion, both neural models, and the v3 C1 recipe remain fixed this round.
The report remains deferred.

## Candidates

Use the same five query-grouped TRAIN folds, seed 0. All four new settings
are LambdaMART with 127 leaves, minimum leaf size 100, L2 10, learning rate
0.05, tree feature fraction 0.8, row fraction 0.8, four threads, deterministic
row-wise fitting and seed 0. Change truncation from 30 to 13, focusing the
pair comparisons closer to the nDCG@10 cutoff. Consulted the official
[LightGBM 4.6 parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html).
The explicitly permitted LambdaMART objective is retained.

1. `rank_focus`: raw plus query percentiles, no augmentation.
2. `normalized_focus`: query min-max plus percentiles, no augmentation.
3. `normalized_augmented`: the second representation with augmentation.
4. `rank_augmented`: the first representation with augmentation.

Each is fitted to 800 trees per fold and scored at fixed 200/400/800-tree
checkpoints. Reported folds never control early stopping. An incomplete run
is excluded entirely and explicitly, without using partial-fold means.

Augmentation keeps every training query unchanged and appends one version
of about half the queries. SHA256 of `augment:20261019:<qid>` chooses copying
by the parity of its first byte and feature-dropout versus noise by its
second byte. The transform is the existing 10% query-column dropout or
Gaussian noise with SD 5% of query-feature SD, using seed 20261019. Labels
are copied without inspection. Original and copied queries get distinct
group IDs. Augmentation happens after splitting; validation queries never
enter a fit. Diagnostic randomness uses the separate existing seed 1729.

## Fixed score averages

Reuse the complete v3 out-of-fold predictions and native fold models.
Compare all 12 combinations of v3 rank checkpoints 400/800, normalized
checkpoints 200/400, and rank weights 0.25/0.5/0.75. This includes v3's
selected 800+200 equal average.

For each of the 12 new single-model checkpoints, also score a fixed 50/50
average with the v3 selected scorer. Flatten this into three positive
constituent weights (0.25 rank-800, 0.25 normalized-200, 0.5 new scorer).
No continuous weight fitting or per-query choice is allowed. Together this
gives 36 declared settings: 12 cached blends, 12 new single models, and 12
new/v3 averages. Training-fold and validation scores are recorded for all.

## Promotion and limits

Evaluate clean, affine, feature-dropout and noise conditions exactly as v3.
These public diagnostics are not the staff C2 construction. Among complete
settings within 0.0025 of the highest clean mean, maximize
`0.5 clean + 0.5 mean(affine, feature_dropout, noise)`. Exact ties prefer fewer
total trees. Require improvement of at least 0.0005 in this mixture over
v3 before promotion. Otherwise retain v3. The narrower threshold measures
incremental progress over the already improved v3, rather than v2.

Freeze selection and refit full TRAIN before one final DEV evaluation.
DEV never chooses a candidate, stopping point or fallback. Compare the
native exported winner with the explicit score average on all TRAIN folds,
check the submission interface and runtime, and independently refit to
verify reproducibility. Preserve all fold measurements and exclusions.
These are adaptively selected TRAIN estimates, not an unbiased new test.

## Completed outcome

All four candidates complete five folds. All 36 fixed settings are recorded
in `results/lambdamart_cv_v4.json` (also the current `lambdamart_cv.json`).
The selected `v3_plus_3_800` has clean mean 0.486042660,
diagnostic mean 0.470255398 and mixture
0.478149029, versus v3's 0.476999608.
The choice is frozen before full-TRAIN fitting and the final DEV run.
Independent full-TRAIN fits are byte-identical; 83 tests and actual
submission-interface/runtime checks pass. Fusion, neural artifacts and the
C1 recipe preserve v3 hashes. V3 remains a recoverable Git/ZIP checkpoint.
