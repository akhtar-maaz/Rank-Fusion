# V14: user-authorized C2-first choice from completed V13 evidence

On 22 October, after V13's failed promotion gate was explained, the user
explicitly reiterated that scoring improvement, especially C2, is the
priority and reminded us of the 23:59 IST submission deadline. This round
accepts the small clean-accuracy tradeoff of V13's already measured highest
robustness candidate, complement_30. This is a retrospective choice using
known TRAIN CV results under the clarified preference, not a new search,
independent validation result, or a claim that V13's original gates passed.
V13's report, failed joint gate, and audit remain immutable.

The fixed candidate's TRAIN diagnostic mean is .4783456884857688 versus
.4770249896490902 for the evaluated control, a gain of .0013206988366786243.
All three diagnostic means improve and all five paired diagnostic folds win.
Clean mean changes from .4875599715916302 to .4870522660747267
(-.0005077055169035). The historical half-clean/half-diagnostic gain is
.00040649665988751194, below V13's .0005 gate. We explicitly accept this
tradeoff for this separate C2-first candidate; no retroactive gate change.
Public affine/dropout/noise scenarios do not reconstruct private C2.
Repeated selection on the same TRAIN folds is subject to overfitting and
cannot demonstrate a private-score improvement or predict class rank.

## Fixed configuration and efficient fitting

Freeze configs/v14_tree.json and configs/v14_new_members.json before any
new fitting or DEV inspection. The final scorer is exactly:

- .5 times the evaluated 3,800-tree V10/V12 base, retained with its original
  TRAIN+DEV fitting/reproduction evidence;
- .25 times v8/1/600, query-bagging extra trees, full_query_mixed seed20261021;
- .25 times v12/1/400, balanced_query_domains seed20261024 strength2.

Only the last two members are newly fitted, twice independently. Their
parameters, counts, seeds and augmentations are exactly the archived TRAIN
candidate settings. The unchanged train_model.py fits their equal-weight
1,000-tree group on authorized released TRAIN+DEV: 958,671 original rows,
8,000 split-namespaced queries and 136 raw features. No early stopping,
search, neural fitting, DEV tuning or post-fit selection. Each fit requests
4 CPUs/24 GB/65 minutes. Retaining the verified base avoids redundant
3,800-tree refits; do not describe this as an all-member fresh refit.

Compile the retained base and fresh group with weights .5/.5 into one
4,800-tree, 408-internal-feature native LightGBM model. Require both new
full-group artifacts and final compiled artifacts to have identical bytes,
plus matching data/configuration/source/runtime identities. Preserve all
neural artifacts/history, fusion, C1 recipe, graded code and staff harness.

## Verification and delivery

Use TRAIN caches only to check native nested composition against explicit
OOF averaging in all five folds and all four public scenarios. Check all
contributing cache hashes/labels/query IDs. Score tolerance5e-12, nDCG1e-12.
No new selection takes place in this verification. Run the existing185 tests,
staff loading on local Python3.12 and HPC3.9.21, protected hashes, local/HPC
identities, and 60s preparation/2s per-call limits. Evaluate final DEV once
only after the choice is committed/pushed, with tree accuracy labeled
IN-SAMPLE after disclosed TRAIN+DEV fitting. Never choose based on that DEV.
No official test, other fold, hidden benchmark identity or outside labels.

Persist every pre-qsub intent and actual ID atomically; never duplicate jobs.
Preserve eight previously evaluated ZIPs and the V13 audit ZIP. Commit/push
verified artifacts and truthful partial-refit/data/AI provenance. Build and
verify dist/rank-fusion-submission-v14.zip from committed HEAD. Formal report
remains deferred. Deadline is TODAY22October23:59IST; target verified delivery
well before it. If verification is incomplete by23:40IST, retain uploaded#597
and do not offer an unverified artifact. Fixed fitting counts must not be
changed to force deadline completion. Private C2 may improve or regress.
