# V11: scale-resistant trees and TRAIN-selected weighted fusion

## Scope and data boundaries, fixed before new measurements

The user requested another round after V10 submission 482: rank 10/35,
B 0.6824, P 0.4400, C2 0.3795, correctness 10/10. V10's disclosed DEV
tree fitting substantially increased its in-sample B score, while C2 changed
only 0.0005. This motivates testing different query-local representations;
it does not reveal how the staff construct C2 or establish a private optimum.
The formal report remains deferred.

All selection, tuning and stopping below use released Fold1 TRAIN only.
Reuse the historical five query-grouped folds, seed 0. No official test,
other fold, undisclosed benchmark, outside labels or leaderboard-driven
candidate/threshold changes. DEV is used only for the authorized final tree
fit after choices freeze, and for one final evaluation, labeled in-sample
for the tree. Do not tune on that evaluation. Neural artifacts, graded neural
source, history and C1 recipe remain exact evaluated V10 copies.

## Finite tree comparison

Four specifications in `configs/v11_candidates.json` cross two feature modes
with clean versus existing `full_query_stress` augmentation: percentile-only
(one midrank percentile per raw feature) and query-normalized (minmax plus
percentiles). Percentiles remove dependence on positive monotonic rescaling;
minmax removes positive affine scale/shift, subject to numerical precision.
No feature meanings or labels enter those transforms. Augmentation is applied
only after separating training and validation queries.

Use LambdaRank, 127 leaves, minimum leaf 100, L2 10, learning rate .05,
feature/bagging fractions .8, bagging frequency 1, truncation 30, seed 0,
deterministic row-wise training on four CPUs. Train each fold to 800 trees,
evaluate only prefixes 400 and 800; no early stopping. Evaluate the existing
clean/affine/dropout/noise TRAIN diagnostics, seed 1729. These are public
diagnostics, not a reconstruction or estimate of the hidden C2 versions.

Reuse the exact archived six-source TRAIN-fold control from V9/V10: v3/0/800,
v3/1/200, v4/3/800, v5/1/800, v6/3/800, v7/2/400, weighted .0703125,
.0703125, .140625, .28125, .1875, .25. Verify it reproduces the archived
TRAIN CV to 1e-12; it is a cached TRAIN-only control, not a fresh V10 refit.
For each of eight new checkpoints compare the single model and its fixed
averages with that control at new-model weights .25/.5/.75: 33 settings total.

Keep the established gates: clean within .0025 of best clean, each diagnostic
within .0025 of the control; maximize .5 clean + .5 mean(three diagnostics),
ties prefer fewer trees. The winning eligible setting must improve this
mixture by at least .0005, diagnostics by at least .001, and the mixture on
at least three of five paired folds. Otherwise retain the exact V10 tree;
do not rescue a runner-up. Missing/failed candidates block selection rather
than silently reducing the search. Verify native export against the explicit
average on all five validation folds, maximum error 5e-12 and identical nDCG.

## Finite fusion comparison

Hand-written weighted RRF and weighted minmax CombSUM use the four provided
named channels only. Compare all nonnegative tenth-step weights summing to
one (286), with RRF constants 0, 10, 60 (858 settings), plus 286 CombSUM,
ordinary CombMNZ and the exact current quarter-weight CombSUM control:
1,146 distinct configurations. Keep stable candidate-order tie breaking.
Zero-weight channels contribute nothing; negative/nonfinite weights and
invalid RRF constants are rejected. Required unweighted graded methods stay
unchanged. Vectorized selection must match the hand-written inference scores
and harness metrics on toy examples, including ties/constant channels.

Score each rule on all TRAIN queries and the same five query folds; the fixed
proxy is .8 nDCG@10 + .2 MAP@10. Choose the highest overall proxy (stable
configuration order on ties). Promote only if its proxy exceeds the current
control by .0002 and on at least three paired folds; otherwise retain control.
This proxy cannot compute class percentiles or predict Track A bands. No DEV
fusion selection. All candidate metrics and the gate decision are retained.

## Execution, freezing and release

Commit/push this protocol/configuration before new real-data measurements.
Test implementation and compatibility before college execution. Four CV jobs
request four CPUs/16GB/three hours; fusion and dependent tree selection use
four CPUs with bounded walltimes. Save actual IDs atomically and preserve
scheduler metadata, including failures. Never submit duplicate work.

Download/audit/commit/push complete TRAIN-only tree and fusion selections
before final DEV inspection. If promoted, fit the selected tree twice fresh
on released TRAIN+DEV with namespaced query IDs (958671 original rows, 8000
queries, 136 raw features), exact frozen parameters/counts/weights, no stopping
or post-fit selection. Require byte identity and input/config/source/runtime
identity. If tree is not promoted, preserve V10's exact fitted tree and its
provenance; do not call it a fresh fit. Do not refit neural models.

Final evaluation never changes choices. Require local/server full tests,
native/independent checks, staff neural loading on HPC and pristine local
Python 3.12, actual artifact hashes and prepare/rerank limits. Update truthful
artifact-specific fitting/tuning/stopping declarations, preserve AI provenance,
all earlier records and V10/V9/V8/V7.1/V7/V6 ZIPs. Commit/push verified changes
and build `dist/rank-fusion-submission-v11.zip` from committed HEAD. The user
uploads manually; no private score or class rank is promised.

## Frozen selection outcome

All four candidates and all six first-stage job records completed successfully.
The exact archived control reproduced the historical TRAIN-fold metrics. None
of the new tree settings beat the control under the frozen comparison, so
V11 retains V10's exact 3,800-tree TRAIN+DEV artifact; there is no V11 tree
refit or neural fitting. Five-fold native equivalence checks pass.

The fusion winner is minmax CombSUM with weights .3 BM25 whole-document,
.3 BM25 title, .4 LMIR.DIR and zero LMIR.JM. Its TRAIN proxy gain is
0.0006460764 and four of five paired folds improve, passing both fixed gates.
The complete 1,146-rule result and all tree/parts/control/gate/native evidence
are saved in results/fusion_selection_v11.json, results/lambdamart_cv_v11.json,
results/cv_v11/, results/selection_jobs_v11.json and
results/v11_selection_audit.json. These selections are committed and pushed
before any new final DEV evaluation or application of the selected fusion.
No selection threshold, candidate, stopping count or weight changed after
inspecting measurements. Formal report writing remains deferred.
