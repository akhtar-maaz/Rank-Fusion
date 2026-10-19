# Sixth-round protocol, frozen before measurement

The user-provided v5 snapshot reports fourth of 21 teams, C2 0.3724,
LTR 0.4976, pairwise held-out 0.4333 and live score 0.4695/0.65.
Preserve v5 commit `be4a32d`, its ZIP, native model and neural artifacts.
Only released Fold1 TRAIN is used for fitting and selection. No DEV tuning,
early stopping, official test labels, other folds or undisclosed datasets.
DEV is evaluated once after freezing the final configurations. Report
writing remains deferred. Fusion, pointwise and C1 recipe stay fixed.

## Tree candidates

`configs/v6_candidates.json` freezes four five-fold query-grouped TRAIN
fits, using the same split seed 0 and fixed 200/400/800 checkpoints.
There is no early stopping on reported folds. The first three use v5's
half-query mixed augmentation, seed 20261020:

1. V5's mixed `query_all` configuration with tree seed 17, to measure
   whether independent stochastic fitting improves a fixed average.
2. The same configuration with `query_normalized`, to remove dependence
   on absolute feature scales.
3. Mixed `query_all` with 63 leaves, minimum leaf size 200 and L2 20;
   keep feature fraction 0.8 rather than v5's regularized candidate's 0.5.
4. V5's mixed `query_all` configuration with one perturbed copy of every
   fit query, seed 20261021. Each query gets affine, dropout or noise by
   the same opaque-query SHA256 rule. Originals remain present, labels
   are only copied, and augmented copies have separate group IDs.

Unless changed above, every candidate uses 127 leaves, minimum leaf 100,
L2 10, learning rate 0.05, feature/row fractions 0.8, row frequency 1,
truncation 30, deterministic row-wise fitting and four threads. Augmentation
runs after splitting; all feature transformations stay within each query.

Compare exactly 49 settings: the exact cached v5 control, 12 new single
checkpoints and 36 averages of those checkpoints with v5 at new-model
weights 0.25/0.5/0.75. Flatten positive weights into one native model.
Use the unchanged v5 promotion rule: retain settings within 0.0025 clean
nDCG of the best complete mean; no diagnostic may fall more than 0.0025
below v5. Maximize half clean plus half the mean of affine/dropout/noise
(diagnostic seed 1729); ties prefer fewer trees. Require mixture gain
at least 0.0005 over v5 or retain the exact v5 artifact.

All candidates must finish all five folds or have an explicit runtime
exclusion. Never compare partial-fold means. Reproduce the exact v5
control and verify exported native predictions against explicit averages.
After freezing a promoted choice, measure both configurations under
perturbation seed 271828 without changing selection. These diagnostics
do not reconstruct staff C2 or guarantee a hidden-score gain. The reused
folds and repeated selection limit confidence in the estimates.

## Pairwise candidates

`configs/v6_neural_candidates.json` freezes four plain 136→128→64→1 ReLU
networks, loaded unchanged by the staff MLP loader. All use seed 0,
batch size 1024, Adam and handwritten nDCG-weighted RankNet, preceded by
supervised MSE warm-up at learning rate 0.001 with a fresh optimizer for
fine-tuning. The graded `ranknet_loss` remains plain unweighted RankNet.
No distillation, model targets, external labels or feature changes.

1. Exact v5 control: six MSE epochs, two RankNet epochs at 0.0001,
   at most 300 distinct strictly ordered pairs per query per epoch.
2. The same with at most 1,000 pairs per query.
3. Six MSE epochs, four RankNet epochs at 0.00005, 300 pairs per query.
4. Ten MSE epochs, two RankNet epochs at 0.0001, 300 pairs per query.

Evaluate each fixed epoch configuration on all five query-grouped TRAIN
folds (seed 0); fit standardization only on each training fold. Use final
fixed-epoch scores, never the best epoch on a reported validation fold.
Report every fold's training and validation nDCG. Check that control fold 0
reproduces the earlier 0.44476016845475064 result within 1e-10.
Select highest five-fold mean; ties prefer fewer pairwise epochs, then the
candidate index. Promote only if the mean improves by at least 0.001 over
control and improves at least three of five folds. Otherwise preserve the
exact v5 pairwise artifact. Then fit the frozen choice on full TRAIN and
independently refit it, with staff-loader and reproducibility checks.

## Completed outcome

All reported settings contain five complete folds. Tree promotion:
True; selected `v5_plus_3_800_weight0.25` with clean mean
0.487342862, diagnostic mean 0.475630864
and mixture 0.481486863. Pairwise promotion:
True; selected `longer_warmup` with five-fold mean
0.435046121, versus control
0.431178559. Choices were frozen from TRAIN
before full fitting and final DEV evaluation. All 96 tests pass locally
and on the college server; promoted artifacts reproduce byte-identically.
Pointwise, fusion and the C1 recipe preserve v5 hashes. Formal report
writing remains deferred. V6 still needs staff C2/held-out/live evaluation.
