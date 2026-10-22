# V12: balanced training domains and fixed longer neural training

## Authorization, data and preservation

After evaluated V11 submission558 the user requested further improvement.
Rank10/35, B .6824, held-out P .4400, C2 .3795, 10/10correctness, live.4797.
V11 fusion gains stayed inside the same nearest-.005 scoring bands; C2 is
unchanged. Preserve evaluated commit1bbeefc9f258dc2da8b945db6df43bd31c34e49b,
all artifacts and evaluated ZIPs. The formal assignment report stays deferred.

Freeze this finite plan/configurations in Git and push before new measurements.
All fitting, tuning and stopping for selection use released Fold1 TRAIN only:
five query-grouped folds, seed0, no early stopping. No official test, other
fold, external label, hidden-benchmark identification or distillation target.
DEV is used only for the already authorized final TREE TRAIN+DEV fitting
after selection and one final evaluation. Pairwise fitting remains TRAIN only.
No post-fit DEV-based choice. Tree DEV accuracy is explicitly in-sample.
Keep V11 fusion, pointwise model, C1 recipe and corrected graded neural source
exactly unchanged. Each tree/neural choice is independent and keeps the exact
evaluated artifact if its gate fails. Do not rescue an unqualified runner-up.

## Four balanced-domain tree candidates

Existing augmentation appends one scenario per query. This round instead
keeps every training query and appends three versions: positive affine
rescaling/shift, whole-feature dropout, and Gaussian noise. Every training
query therefore appears in all four public training domains, with its labels
copied unchanged. Apply only after splitting queries; group every version
separately. Use existing label-free perturb_features with new seed20261024,
distinct from diagnostic seed1729. The new name is balanced_query_domains;
strength1/2 retain the existing documented perturbation magnitudes. Validation
uses unchanged clean/affine/dropout/noise diagnostics. These are not staff C2
versions, and their metrics cannot predict private performance.

`configs/v12_candidates.json` crosses strengths1/2 with feature fractions.8/.5.
All use query_all (raw/midrank/minmax), LambdaRank,127leaves, minimum leaf100,
L2=10, learning rate.05, row fraction.8/frequency1, truncation30, seed0,
deterministic row-wise fitting and4threads. Train to400trees and evaluate only
fixed200/400prefixes. No warm starts, adaptive checkpoints or new-new mixtures.

Compare33settings: exact six-source historical TRAIN-only control plus each
of8new checkpoints alone and averaged with control at new weights.25/.5/.75.
Control sources v3/0/800,v3/1/200,v4/3/800,v5/1/800,v6/3/800,v7/2/400 and
weights.0703125/.0703125/.140625/.28125/.1875/.25 must reproduce archived
TRAIN metrics/specifications within1e-12. Retaining this cached TRAIN control
does not refit the evaluated V10/V11 TRAIN+DEV artifact.

Preserve the established tree gates: clean within.0025of best complete mean;
no public diagnostic more than.0025below control; maximize .5clean+.5mean
diagnostics, ties fewer trees. Winner needs mixture gain≥.0005, diagnostic
gain≥.001 and strict positive paired mixture gains on≥3/5folds. Otherwise
retain exact V11 tree. All4candidates/5folds required, no runtime exclusions.
Verify native compiled winner against the explicit average on all five
validation folds, maximum absolute error5e-12 and identical nDCG.

## Five fixed pairwise configurations

`configs/v12_neural_candidates.json` contains the exact evaluated V8/V11
warmup20_seeds_0_41 control and four candidates: warm-up30/40 then2RankNet
epochs, or warm-up20/30 then4RankNet epochs. Every candidate averages seeds0
and41 equally, each136→128→64→1ReLU, compiled into one ordinary136→256→128→1
staff-loadable MLP. All other parameters remain unchanged: MSE warm-up LR.001,
fresh Adam for handwritten nDCG-weighted RankNet at.0001, pair cap300,
batch1024, fixedepochs, no DEV. Graded stable RankNet/MSE/SGD logic remains
exact evaluated source, including corrected deterministic initialization.

Freshly fit10distinct full specifications per fold (50fits total). Cache
identities include all settings, TRAIN hash/indices, standardizer, source
and runtime. Do not use seed alone as a cache key. Same-fold standardizers
must be identical and fit only fold TRAIN. The fresh control must reproduce
archived results/neural_selection_v8.json selected candidate, each exported
fold's train/validation metric and each member history within1e-10, plus
same-basename artifact/state/statistics identities where comparable. A
mismatch blocks promotion; never silently alter the reference or tolerance.

Choose the new candidate with highest exported-MLP five-fold validation mean,
ties earlier candidate index. Promote only if mean gain≥.001 and≥3/5folds
strictly improve over fresh exact control; otherwise preserve V11 pairwise
bytes. Compilation must load with pristine staff code, hidden widths≤256,
finite scores and match explicit averaging at atol2e-6/rtol2e-5. Assess the
exported artifact including float32 effects. No architecture/fusion search.

## Execution and final release

First-stage PBS work: four treeCV jobs4CPU/24GB/2h; pairwiseCV/selection
4CPU/8GB/2h; dependent tree selection4CPU/16GB/1h. Preserve actual IDs with
atomic pre-qsub intent and completed scheduler metadata; never submit a
duplicate. All source tests/CI must pass before real fitting. Deadline is
22Oct23:59IST; resource requests bound jobs, not a prediction of completion.
Historical runtimes motivate this bounded round; no timing-driven selection.

Download/audit/commit/push complete TRAIN-only tree and neural choices before
new DEV inspection or promoted final fits. For promoted TREE, run two fresh
exact selected TRAIN+DEV fits (958671original rows/8000namespaced queries/
136rawfeatures) on4CPU/24GB/3h; no tuning/early stopping. For promoted neural
choice, run two fresh exact full-TRAIN fits in separate caches, same artifact
basename,4CPU/8GB/1h. Retained artifacts are copies, never fresh-refit claims.
Require independent byte identity and full input/config/source/runtime
identity for each newly fitted artifact. A final verification job requires
all evidence, protected hashes, staff loading, full tests and inference limits.

Evaluate DEV once after both choices freeze; no post-evaluation adjustment.
Run final local/server tests and local/HPC content hashes, pristine LOCAL
Python3.12 staff loading (college runtime is Python3.9.21). Write truthful
artifact-specific TRAIN+DEV tree versus TRAIN-only neural fitting/tuning/
stopping declarations, preserve AI provenance and all historical evidence.
Commit/push verified outputs, build and verify dist/rank-fusion-submission-v12.zip
from committed HEAD; preserve all evaluated older ZIPs. Notify the user for
manual upload with actual results/limitations, without promising rank/private
scores, then pause the existing heartbeat. Formal report remains deferred.
