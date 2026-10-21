# V10: disclosed TRAIN+DEV final tree fit, frozen before fitting

## Authorization and scope

The user explicitly approved a declared TRAIN+DEV tree fit after receiving
their V9 score. The latest feedback is submission 295, rank 11/34,
LTR 0.4984, held-out pairwise 0.4400, C2 0.3790, 10/10 correctness and
live score 0.4160/0.65 (`results/leaderboard_v9.json`). That feedback has
now been received, satisfying the requested order of work.

The assignment PDF's data-declaration paragraph (page 13) explicitly says:
"Training on dev is permitted" and requires disclosure of fitting,
hyperparameter tuning and early-stopping splits. This round changes the
final tree fitting split only. Earlier V1–V9 declarations remain historically
TRAIN-only and their results/archives remain preserved. The formal report
is still deferred; current repository submission declarations must be updated.

## Frozen method

Use the exact evaluated six-member, 3,800-tree configuration selected using
the archived five query-grouped TRAIN folds. `configs/v10_tree.json` copies
the complete selected specification from `results/lambdamart_cv_v9.json`
without changing any parameter, feature mode, seed, count or weight.
There is no new hyperparameter search, DEV-based selection or early stopping.

The only new fitting condition is the union of released Fold1 TRAIN and DEV:
723,412 TRAIN rows/6,000 queries plus 235,259 DEV rows/2,000 queries,
958,671 original rows/8,000 namespaced queries, 136 raw features.
Existing augmentation is applied per member after combining these inputs;
the trainer namespaces IDs as `train:<qid>` and `dev:<qid>` so reused IDs
cannot merge. Its deterministic hashes therefore use these declared names.
No official test split, other fold, hidden benchmark, external label or
distillation target may be accessed.

Retain exact V8/V9 pairwise and pointwise artifacts, fusion, C1 recipe,
graded neural source and neural-history records. No neural fitting or
selection is authorized in this round. The native inference interface stays
unchanged. Preserve evaluated V9 commit
`dc96beac2abbdd0da473be88cabfab18c171eba3` and all earlier evaluated ZIPs.

## Execution and checks

Commit/push this plan and the fixed configuration before any real fit.
Run two fresh independent fits on college PBS, each with four CPUs,
16 GB RAM and 90-minute walltime. Use the existing explicit `--use-dev`
route in `scripts/train_model.py`, with a fit record of input hashes,
row/query counts, fixed configuration hash, source hashes, runtime and
artifact hash. Outputs first go to separate ignored primary/independent
directories, preserving the current submission until verification succeeds.

A third dependent verification job requests four CPUs, 8 GB and 15 minutes.
It requires byte-identical fitted tree artifacts and matching data/config/
source identities, checks protected V9 hashes, then copies the verified tree
into the submission and evaluates the existing interface once. Record actual
scheduler requests, usage and exit status; save every submitted job ID
atomically, including pre-submission intent to prevent duplicate retries.

DEV tree accuracy is now **in-sample**, because DEV labels enter tree fitting.
Report it as such; do not describe an increase over V9's unseen-DEV score as
a held-out generalization gain. Fusion/neural DEV metrics retain their
TRAIN-only fitting interpretation. Staff-held-out pairwise, C2 and future
class rank are not inferred from the in-sample DEV result. The user will
upload the verified ZIP and report the score. No post-fit model selection,
parameter changes or stopping decisions use the new DEV measurement.

After both fits and verification pass, download and check the final tree,
fit/reproduction records, provenance and protected hashes. Run the full local
suite and pristine staff Python 3.12 loading; check prepare/rerank limits.
Update README and submission notes with an artifact-by-artifact data-use
table: tree TRAIN+DEV, neural models TRAIN only, fusion TRAIN-selected,
recipe staff-supplied benchmark inputs. Preserve the AI/provenance disclosure.
Commit/push only verified outputs and package `dist/rank-fusion-submission-v10.zip`.

## Operational recovery after a scheduler timeout

The original primary fit (1110120) exited zero. Its independent repeat
(1110121) was killed for exceeding its requested 90-minute PBS walltime,
exit -29 after 90m37s, before saving any model or fit record. Dependent
verification (1110122) never started. The submission still contains V9.
Original IDs, failed metadata and logs remain preserved.

Retry only the failed independent fit using the unchanged fitting source,
PBS file and frozen configuration, overriding the scheduler walltime to
three hours with the same four CPUs and 16 GB. Reuse the completed primary
fit and run the unchanged verification workflow after a successful retry.
Save both new IDs separately without replacing original IDs or submitting
duplicates of successful/running work. This is an operational resource
recovery, with no new candidate, tuning, early stopping or DEV-based choice.
No final DEV measurement was inspected to make this decision. Record all
five actual job attempts, including the failed and never-started originals;
do not describe all original jobs as successful. Details are frozen before
the retry in `results/v10_runtime_recovery.json`.

## Completed outcome

Two fresh TRAIN+DEV fits completed successfully and produced byte-identical
3,800-tree/408-internal-feature artifacts. Source/configuration/input identities
match. Five actual PBS attempt records preserve the primary success, the original independent timeout, the original never-started dependency job, the successful unchanged independent retry and successful verification. Only the failed fit was retried, with the scheduler allocation extended from 90 minutes to three hours; the fitting method and four-CPU/16-GB request were unchanged. All 152 local/server tests and
pristine local staff Python 3.12 neural loading pass. The final tree SHA256 is
`f2fa40faac67aaab067bb43eb169fd7d7db5a13c5eeb0a6d248c88f941e44864`. See `results/execution.json` and the two fit records.

The single final local tree DEV nDCG@10 is 0.682399113,
**in-sample** because DEV labels entered fitting. It did not change selection,
parameters, weights or stopping. Other models and their history remain exact
V9/V8 copies. No new private score or class rank is claimed; the user will
upload the verified archive. Formal report writing remains deferred.
