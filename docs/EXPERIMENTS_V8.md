# Eighth-round protocol, frozen before real-data measurement

V7.1 submission243 restores10/10 correctness, with7/7 conformance,
rank7 among29 scored teams, held-out pairwise0.4383, C2 0.3790,
LTR0.4984 and live score0.4493/0.65. One additional entry is queued.
Preserve evaluated commit `0f51d772bb4c357fd46354e45df3901f84f9caac`, its
ZIP and every current artifact. Formal report writing remains deferred.

All fitting, tuning and stopping use released Fold1 TRAIN only. Use the
same five query-grouped TRAIN folds, split seed0, fixed tree checkpoints
and fixed neural epochs. No DEV tuning or stopping, official test labels,
other folds, external relevance labels, model-score distillation or discovery
of undisclosed C1 benchmarks. DEV is evaluated once after both choices freeze.
Fusion, pointwise, C1 recipe and the corrected graded neural functions stay fixed.

## Tree candidates

The exact four fits are in `configs/v8_candidates.json`. Shared settings:
`query_all`, LambdaMART (`lambdarank`),127 leaves, minimum leaf100, L2=10,
learning rate0.05, feature fraction0.8, row fraction0.8/frequency1,
truncation30, `extra_trees=true`, `extra_seed=0`, seed0, deterministic
row-wise fitting, four threads, and full mixed augmentation seed20261021.
Keep originals and append one label-preserving query copy after splitting.
Each candidate has fixed checkpoints200/400/600, without early stopping.

1. Fresh augmentation/threshold diversity: `extra_seed=17`, augmentation
   `full_query_mixed_v8` seed20261023. Use exactly the existing full mixed
   algorithm and magnitudes, with a fresh label-independent query hash.
2. Query sampling: add `bagging_by_query=true`, retaining row fraction0.8
   and frequency1.
3. Node feature sampling: add `feature_fraction_bynode=0.6`, retaining
   feature fraction0.8. No training-speed improvement is assumed.
4. Lower capacity:63 leaves, minimum leaf200, L2=20.

Directions are chosen using archived TRAIN evidence. V7's extra400 blend
beat extra200 and extra800 blends; its pure model's training/validation gap
was smaller than the final mixture's. Previous normalized-only, raw-feature
gain prior, truncation13 and stronger augmentation were weaker than the
selected v7 mixture. These prior results motivate this finite family; no
new real-data measurement may change it.

## Tree comparison and promotion

Compare exactly49 settings: exact v7 control,12 single checkpoints and36
v7/new averages at new weights0.25/0.5/0.75. No new/new averages or adaptive
weight search. The six control sources are v3/0/800, v3/1/200, v4/3/800,
v5/1/800, v6/3/800, v7/2/400, with weights
0.0703125/0.0703125/0.140625/0.28125/0.1875/0.25.

Require five complete folds per reported setting and exact reproduction
of the archived v7 control specifications and metrics within1e-12. Any
runtime exclusion must name a whole candidate explicitly, never a partial
fold mean. Keep settings within0.0025 clean nDCG of the best complete mean,
and no more than0.0025 below control on any public diagnostic. Maximize
half clean plus half mean(affine/dropout/noise); ties prefer fewer trees.
Promote that winner only with mixture gain at least0.0005, diagnostic mean
gain at least0.001, and strictly positive paired fold-mixture gains in at
least3/5 folds. If any gate fails, preserve exact v7 bytes without searching
a runner-up. Verify the native exporter against explicit scores on all folds.

Public diagnostics and selection seed1729 stay unchanged. After freezing a
promoted choice, run one descriptive perturbation seed615019, fixed here
before measurement. Previously inspected seeds271828 and391729 are not new
checks. Do not change the artifact based on this descriptive result. These
generic transformations do not reconstruct private C2; repeated adaptive
fold reuse limits confidence and is not a fresh held-out label test.

## Pairwise candidates and fresh control

`configs/v8_neural_candidates.json` fixes five candidates. The control is
the current equal average of seeds0/41, each136→128→64→1 ReLU,10 MSE
warm-up epochs at0.001, then2 weighted RankNet epochs at0.0001 with fresh
Adam, batch1024 and pair cap300. New candidates, with all other settings fixed:

1. The same two-member average with14 MSE warm-up epochs per member.
2. The same average with20 MSE warm-up epochs per member.
3. The same average with10 MSE warm-up epochs and pair cap1000.
4. One directly trained136→256→128→1 ReLU network, seed0,10 MSE warm-up
   epochs and pair cap300.

All use existing relevance-label supervision and absolute nDCG@10 swap
weights, with the plain graded RankNet loss unchanged. Past TRAIN-only
results motivate longer warm-up, more pairs and the seed average. The
direct dense model tests capacity against the block structure of the current
compiled average. No three-member compilation or custom inference loader.

Fresh-fit all nine distinct member specifications per fold:45 total fits,
including10 control fits. Cache by the complete fit specification, TRAIN
file/split hashes and source identity; a seed alone is not a cache key.
Each same-fold standardizer must be identical and fitted only on fold TRAIN.
The corrected-source control must reproduce the archived v7 candidate and
every fold's exported training/validation nDCG within1e-10, plus recorded
member histories and artifact/state identity where comparable. Stop on a
control mismatch; never silently change the reference.

Select the highest exported-artifact five-fold mean among the four new
candidates, ties by earlier index. Promote only with mean gain at least
0.001 and strict improvements in at least3/5 folds; otherwise preserve the
exact v7.1 pairwise artifact. Compilation must load with pristine staff code,
stay within three hidden layers/width256, have finite scores and agree with
the explicit arithmetic average at atol2e-6/rtol2e-5. Selection uses exported
scores, including float32 effects. Each output is136→256→128→1.

## Final execution and safeguards

Heavy real-data work runs only in college PBS jobs, four threads. Tree CV
requests12GB/60min; neural CV8GB/70min; tree selection12GB/50min;
primary and independent fits12GB/90min; descriptive seed check8GB/30min;
verification8GB/12min. Save actual per-job requests, resource use and exit
status. Choices are independent: retain the corresponding exact baseline
artifact when its gate fails.

After both choices freeze, fit promoted configurations on full TRAIN and
evaluate DEV once. Independently fresh-refit both artifacts in separate cache
directories, using the same basename `pairwise_model.pt` for byte identity.
Verify pristine Python3.12 staff loading, protected hashes, deterministic
training, candidate/unique-ID contracts and prepare/rerank limits; run the
complete test suite including the v7.1 reproducibility regressions. Preserve
all previously evaluated ZIPs. Package and push only verified outputs.

Parameter semantics follow the [official LightGBM4.6 documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html).
