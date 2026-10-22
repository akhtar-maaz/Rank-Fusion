# V13: robustness-first complementary tree ensembles

## Scope and data boundaries

After V12 submission597 the user explicitly requested C2 improvement, its
35% weight being the priority. Their supplied 22October20:29:49 snapshot
reports rank10/35, C2 .3795, B .6824, held-out P .4419, 10/10 correctness
and live .4722/.65. Highest displayed C2 is .4011. Preserve evaluated V12
commit f28b61880a56cac51566890671405db013bb3c94 and all eight evaluated ZIPs.
Keep its exact pairwise/pointwise artifacts, fusion, graded neural code and
C1 recipe. No new neural training. The formal report remains deferred.

Freeze and push this plan and the finite basis before measuring new blends.
All selection uses the released Fold1 TRAIN only, the same five query-grouped
folds/seed0. Reuse only previously completed TRAIN-fold models/OOF scores;
do not access official test, other folds, hidden benchmark identities, outside
labels or distilled scores. Existing TRAIN results inform the choice of
complementary families; new joint combinations have not been measured.
The public affine/dropout/noise diagnostics do not reconstruct staff C2 or
establish a private score. Do not tune to leaderboard or DEV outcomes.

## Frozen complementary family

Earlier rounds compared one added family with the control at a time. This
round jointly combines query-level/node-level randomization, balanced-domain
training and scale-invariant feature handling. All underlying member settings,
seeds, tree counts and augmentations stay at their archived values.

`configs/v13_blend_basis.json` fixes four bases:

1. Exact six-member current TRAIN-CV control: v3/0/800, v3/1/200,
   v4/3/800, v5/1/800, v6/3/800, v7/2/400; internal weights
   .0703125/.0703125/.140625/.28125/.1875/.25.
2. Either v8/1/600 (query bagging extra trees) or v8/2/600 (node subsampling
   extra trees); full-query mixed augmentation, seed20261021.
3. Either v12/1/400 or v12/3/400 (balanced strength2, feature fraction .8/.5),
   originals plus all three training domains, seed20261024.
4. v11/3/800 (query-normalized features, stress augmentation seed20261022).

For each of the four family pairs, enumerate every nonnegative quarter-step
four-basis weight vector summing to one. Remove zero-weight members and
deduplicate identical source/weight specifications. This is exactly85 unique
settings, including the exact control. No new-new fitting, adaptive weight
search, score normalization, runtime exclusions or additional families.
At most5600 total trees and408 internal features; inference remains one
native LightGBM scorer through the existing ranking interface.

Require all contributing source reports, five folds per report and exact
OOF labels/query identities/checkpoints; hash the reports/OOF/model caches.
The exact control must reproduce archived TRAIN metrics/specifications within
1e-12. No current TRAIN+DEV model is used to create OOF predictions.

## Selection, frozen before the new measurements

Prior rounds' clean/diagnostic mixed objective did not qualify newer families.
This new round explicitly prioritizes diagnostic accuracy, reflecting the
user's C2 request, without revising any past round's decisions or gates.

Eligibility: clean mean at least control clean minus .0025, and each of the
three diagnostic means at least its control mean minus .0025. Choose the
highest mean over affine/dropout/noise; ties prefer fewer trees, then earlier
setting order. The single chosen proposal must meet ALL of:

- diagnostic mean gain at least .001;
- gain at least .0005 in the historical .5clean+.5diagnostic score;
- strictly positive paired diagnostic-mean gains on at least three of five folds.

Otherwise retain the exact evaluated V12 tree; do not rescue another setting.
These numerical gain thresholds are unchanged from the earlier tree gates;
the preregistered objective, clean eligibility reference and paired score now
focus on robustness. Never change them based on these measurements.
Check native export against explicit weighted OOF predictions in every fold
and all four scenarios, maximum absolute score error5e-12 and identical nDCG
within1e-12. Complete TRAIN-only results/audit must be committed and pushed
BEFORE final fitting or any new DEV inspection.

## Final fitting, verification and deadline

If promoted, freeze the exact selected configuration and run two fresh fits
on authorized released TRAIN+DEV, with split-namespaced IDs:958671 original
rows/8000queries/136rawfeatures. Fixed counts/seeds, four threads, no early
stopping or post-fit choice. Request each4CPU/24GB/2h; the selection job uses
4CPU/16GB/1h. Resource bounds are not completion predictions. Persist pre-qsub
intent and every actual ID atomically; never duplicate jobs. Reproduce complete
model bytes and input/config/source/runtime identity independently. Retained
artifacts are copies, not fresh fits. Preserve exact V12 neural artifacts/history.

Then verify protected hashes/native dimensions/staff loading and evaluate DEV
once. Tree DEV accuracy is IN-SAMPLE after TRAIN+DEV fitting; do not use it
to choose the shipped model. Run final local/server/source-CI checks, pristine
LOCAL Python3.12 staff loading (HPC3.9.21), local/HPC content hashes and 60s
prepare/2s per-call limits. Record actual artifact-specific fitting/selection/
stopping, AI provenance, all jobs/resources and limitations truthfully.

Deadline is22October2026 23:59IST. Preserve the ready V12 ZIP throughout,
leave time for manual upload, and never ship a partial/unverified new tree.
Commit/push verified outputs and create/verify dist/rank-fusion-submission-v13.zip
from committed HEAD, including CRC/blobs/original paths/exclusions/artifact
hashes and eight older evaluated ZIPs. Notify the user of actual results,
without promising any private C2 score/rank. Formal report stays deferred.
