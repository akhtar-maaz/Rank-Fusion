# Ninth-round protocol, frozen before real-data measurement

User-reported V8 submission 271 is ninth of 31 scored teams, with 7/7
conformance, 10/10 correctness, held-out pairwise 0.4400, C2 0.3790,
LTR 0.4984 and live score 0.4576/0.65. Preserve evaluated commit
`bb9ea0afb8dcd97dc69a9caceab3f7868ce0c0c4`, its ZIP and artifacts.
Formal report writing remains deferred.

All fitting, tuning and stopping use released Fold1 TRAIN only. Use the same
five query-grouped folds, split seed 0. No DEV tuning, fitting or early stopping,
official test labels, other folds, external labels or hidden-benchmark discovery.
One final DEV evaluation follows the frozen choice. Pairwise, pointwise, fusion,
C1 recipe and the corrected graded neural source retain exact V8 bytes.

## Motivation and finite candidates

The code audit found consistent training/inference transformations and native
export errors below 2.4e-14 with identical held-out TRAIN nDCG. Archived tree
CV clean means plateau near 0.488, while the selected model's fit mean is
0.732. V8's best eligible mixture failed its fixed diagnostic improvement gate;
that decision is final. V9 tests different ranking gradients and substantially
shallower, smoothed trees rather than changing V8's gates or rescuing its runner-up.
Competitors' displayed scores do not establish their training procedures or an
attainable TRAIN-only score.

The exact four fits are in `configs/v9_candidates.json`. All use `query_all`,
learning rate 0.05, feature fraction 0.8, row bagging fraction 0.8/frequency 1,
seed 0, deterministic row-wise fitting and four threads. Fixed checkpoints
are 400/800/1200, with no early stopping.

1. `xendcg_clean`: XE_NDCG_MART (`rank_xendcg`), objective seed 0,
   63 leaves, maximum depth 8, minimum leaf 100, L2 10, no augmentation.
2. `xendcg_augmented`: the same settings with existing full mixed augmentation
   seed 20261021, retaining originals plus one separate label-preserving query copy.
3. `shallow_smoothed_clean`: LambdaRank, truncation 30, 31 leaves, maximum
   depth 5, minimum leaf 300, L2 30, path smoothing 100, no augmentation.
4. `shallow_smoothed_augmented`: the same shallow settings with existing full
   mixed augmentation seed 20261021.

The assignment permits an equivalent gradient-boosted listwise ranking objective.
LightGBM documents XE_NDCG_MART as a ranking objective and path smoothing as
regularization in its [official 4.6 parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html).
No new dependency, feature mode, augmentation algorithm or neural fit is added.
The exporter may combine ordinary numeric single-output `lambdarank` and
`rank_xendcg` boosters; no custom inference loader is introduced. Synthetic
tests must prove mixed-objective native score equivalence before real jobs.

## Comparison and promotion

Compare exactly 49 settings: exact V8 tree control, 12 new single checkpoints,
and 36 V8/new score averages at new-model weights 0.25/0.5/0.75. No new/new
mixtures, score calibration or adaptive weights. V8 retained the six-source
V7 tree: v3/0/800, v3/1/200, v4/3/800, v5/1/800, v6/3/800, v7/2/400;
weights 0.0703125/0.0703125/0.140625/0.28125/0.1875/0.25.

Retain the V8 rules without relaxation: five complete folds, exact control
specifications and metrics within 1e-12, clean mean within 0.0025 of the best
complete setting and each public diagnostic no more than 0.0025 below control.
Select the highest half-clean/half-diagnostic-mean score, ties fewer trees.
Promote that winner only with mixture gain at least 0.0005, diagnostic mean
gain at least 0.001 and strict positive paired mixture gains in at least 3/5
folds. Otherwise retain exact V8 tree bytes, without searching a runner-up.
Whole-candidate runtime exclusions must be explicit; no partial-fold means.
Native versus explicit averaging must agree on all five folds within 5e-12
absolute score error and 1e-12 nDCG. Public diagnostics and seed 1729 stay fixed.

After a promoted selection, one descriptive perturbation seed 902617 compares
V8/V9; it cannot select or alter the artifact. Previously used diagnostic seeds
are not fresh evidence. Reused folds and repeated adaptive development limit
confidence. None of these transformations reconstructs the private C2 versions.

## Execution and verification

Heavy real-data work runs only in college PBS, four CPUs. Four CV array jobs
request 12 GB/90 minutes; selection 12 GB/50 minutes; primary and independent
full-TRAIN fits 12 GB/90 minutes; descriptive seed check 8 GB/30 minutes;
verification 8 GB/12 minutes. There are nine actual jobs, including four array
children. Every job ID is saved atomically; no duplicate submissions.

Commit and push this protocol before real measurements. Verify and push the
TRAIN-only selection record before downloading or inspecting final DEV.
Fit promoted trees twice independently; fallback copies preserve exact V8
bytes and must not be reported as refits. Check protected neural/core/fusion/
recipe hashes, pristine staff Python 3.12 loading, full tests and prepare/rerank
limits. Record actual resource use, failures and all complete results truthfully.
Preserve V8/V7.1/V7/V6 ZIPs. Push and package only verified outputs; the user
uploads the final V9 ZIP manually. No private-score or class-rank promise.
