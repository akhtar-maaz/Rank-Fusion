# V14 recovery: fixed averaging of both completed stochastic fits

V14 primary1112539 and independent1112540 both completed successfully with
identical data/configuration/source/runtime identities. Original final
verification1112542 failed BEFORE installation or DEV evaluation because
the two newly trained 1,000-tree groups had different bytes. Of their first
600 query-bagging trees,599 differ; the last400 balanced-domain trees match
exactly. Primary group SHAfe216a8bc758117315e7243a9683dbc7b093ef096bc52f19cfdd33ee948180ad;
independent SHA f497f500762706f2baac74c2c07183cff9a02aca0fb2483dec25b16585731b29.
The existing deterministic/force-row-wise settings did not produce identical
query-bagging fits in this execution. We have not established the root cause. The official [LightGBM4.6 parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html) describe deterministic CPU mode with fixed row/column handling; our observation is recorded without attributing a confirmed library defect.
Tree training is NOT byte-reproducible here; do not claim otherwise. The
unchanged graded neural training/reproducibility checks remain separate.

Under the user's explicit C2-first scoring priority and23:59IST deadline,
freeze an ensemble of BOTH completed groups with equal weights, without
choosing a run by training or DEV accuracy. No more fitting, altered seed,
capacity, iteration count, library version, or new label source. The final
native score is .5retainedbase +.25primarygroup +.25independentgroup.
This exports5,800trees/408internalfeatures (the400matching balanced trees
are present twice). Effective contribution is .5base+.125baggingprimary+
.125baggingindependent+.25balanced. This is a documented change to V14's
training-byte-identity acceptance requirement, prompted by observed fitting
variation, not a claim that its original verification passed.

Both new groups remain complete and fixed. Compile twice from these same
three fixed artifacts and require identical export bytes, explicit arithmetic
score agreement, deterministic inference, staff loading and all185tests.
Two exports from the same fixed inputs do NOT constitute independent fresh
training reproduction. Preserve original failed verification and all job IDs;
submit one separate recovery verification, no new fitting. Evaluate final
DEV once only after this recovery choice is committed/pushed. Tree DEV is
IN-SAMPLE after historical/base and new-member TRAIN+DEV fitting; never
choose the shipped ensemble using that result.

The original20TRAIN native-composition checks cover the V13-selected family
and fixed weighting, not an independent assessment of the newly averaged
stochastic replicas. Report the original TRAIN CV measurements as family
selection evidence, not freshly measured performance of the final two-run
ensemble. No private C2 improvement or class rank can be established locally.
Preserve V13's failed performance gate and V14's failed identity gate honestly.
All neural/fusion/C1 artifacts/history/graded source remain unchanged.
Formal report deferred. Verify and deliver before the hard manual deadline.
