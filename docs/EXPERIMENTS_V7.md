# Seventh-round protocol, frozen before measurement

V6 submission 173 reports eighth of 24 teams, pairwise held-out 0.4367,
C2 0.3747, LTR 0.4979, live score 0.4253/0.65, 7/7 conformance and
10/10 correctness. Preserve evaluated commit `0646047`, its ZIP and
artifacts. Only released Fold1 TRAIN is used for fitting and selection;
no DEV tuning, early stopping, official test labels, other folds,
undisclosed benchmark discovery or distillation. Formal report remains
deferred. Fusion, pointwise and the C1 recipe stay fixed.

## Tree candidates and augmentation

The exact four specifications are in `configs/v7_candidates.json`.
All use `query_all`, fixed 200/400/800 checkpoints on five query-grouped
TRAIN folds (split seed 0), with no early stopping on reported folds.
Shared parameters are LambdaMART, 127 leaves, minimum leaf 100, L2 10,
learning rate 0.05, feature/row fractions 0.8, row frequency 1, seed 0,
deterministic row-wise fitting and four threads. Truncation is 30 except:

1. Full mixed augmentation with truncation 13, emphasizing the head.
2. Full mixed augmentation with split-gain weights 0.5 on 136 raw
   columns and 1 on the 272 rank/min-max columns (`feature_contri`).
3. Full mixed augmentation with randomized thresholds (`extra_trees`,
   `extra_seed=0`), keeping the permitted LambdaMART objective.
4. Full-query broader augmentation, seed 20261022, truncation 30.

Each keeps all original fit queries and appends one label-preserving
copy with a distinct group ID. The original full mixed scheme and seed
20261021 are unchanged for candidates 1–3. Candidate 4 deterministically
chooses affine/dropout/noise by SHA256 of its opaque query ID and seed,
with affine scales 0.25–4 and shifts within half the query-feature range,
20% whole-query-column dropout, or Gaussian noise SD 10% of query-feature
SD. Labels are copied without inspection. Augmentation occurs after
splitting; validation queries never enter fitting. These are broad public
diagnostics, not a reconstruction of staff C2. Default diagnostic transforms
remain byte-identical to v6; training and diagnostic seeds differ.

## Fixed tree comparisons and promotion

Compare exactly 49 settings: the exact cached v6 control, 12 new single
checkpoints, and 36 v6/new averages at new-model weights 0.25/0.5/0.75.
V6's flattened source weights are 0.09375/0.09375/0.1875/0.375/0.25
over v3 rank800, v3 normalized200, v4 augmented800, v5 mixed800 and
v6 full-mixed800. No new/new weight search or per-query model choice.

Every reported setting must contain five complete folds. Runtime exclusions
must be explicit and exclude a whole candidate, never use a partial mean.
Reproduce v6's member specifications and aggregates within 1e-12. Keep
settings within 0.0025 clean nDCG of the best complete mean and no more
than 0.0025 below v6 on any standard diagnostic (seed 1729). Pick highest
half-clean plus half-mean-diagnostic score; ties prefer fewer total trees.
Promote that winner only if mixture gain is at least 0.0005, diagnostic
mean gain at least 0.001, and paired fold-mixture gain is strictly positive
in at least three folds. If any gate fails, retain the exact v6 artifact
without searching a runner-up. Record every gate and paired gain.

After freezing a promoted choice, run descriptive comparison with v6
under perturbation seed 391729. Earlier post-selection seed 271828 has
already been inspected; the new seed is fixed before v7 measurement.
Do not change the selected artifact based on this check. Reused query
folds and repeated adaptive selection limit confidence; neither seed is
a new held-out label test or a guarantee of private-score improvement.

## Pairwise seed averages inside the staff model bounds

`configs/v7_neural_candidates.json` freezes three candidates: the exact
v6 seed-0 control, an equal average of seeds 0 and 17, and an equal average
of seeds 0 and 41. Each member is 136→128→64→1 ReLU with ten supervised
MSE warm-up epochs at 0.001, then two handwritten nDCG-weighted RankNet
epochs at 0.0001 with a fresh Adam optimizer. Batch size is 1024 and
sampling caps 300 distinct strictly ordered same-query pairs per epoch.
Graded `ranknet_loss` stays plain unweighted RankNet. No model's scores
become training targets; arithmetic averaging happens after fitting.

Use the same five TRAIN folds, fixed epochs and fold-local standardization.
Cache the three seed fits within each fold (15 fits total), with TRAIN-file
and split-index hashes plus exact fit parameters and artifact hashes.
All same-fold standardizer statistics must be identical. Reproduce v6's
control training and validation nDCG in all five folds within 1e-10.

Two members compile into a single ordinary 136→256→128→1 ReLU network:
concatenate first-layer units, use block-diagonal second-layer weights,
and average outputs in the final linear layer. The unchanged pristine
staff loader handles it within two hidden layers of width at most 256.
The artifact format is unchanged. Use exported-artifact nDCG for selection;
record explicit arithmetic-average nDCGs and float32 score errors too.
Require finite scores and compilation agreement at atol 2e-6, rtol 2e-5.

Select the highest five-fold validation mean among the two seed averages;
ties prefer the earlier candidate index. Promote only if its mean improves
by at least 0.001 and it improves at least three folds over control. Otherwise
retain the exact v6 pairwise bytes. Full-TRAIN fitting uses the frozen
configuration only. Independent reproduction uses fresh fits and a separate
cache, retaining the output basename `pairwise_model.pt` in both directories.

## Final checks

Only after both choices are frozen, fit promoted configurations on full
TRAIN and evaluate DEV once. Independently reproduce both artifacts,
check pristine staff loading under Python 3.12, native tree export on
every fold, protected artifact hashes, candidate/unique-ID contracts,
determinism and prepare/rerank limits. Run the full existing test suite.
Heavy real-data work runs only in scheduled college jobs, four threads;
tree jobs request 12 GB and neural jobs 8 GB. Record actual resources,
runtime exclusions and outcomes. Preserve v6 even if v7 is promoted.

Parameter meanings follow the [official LightGBM 4.6 documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html).
