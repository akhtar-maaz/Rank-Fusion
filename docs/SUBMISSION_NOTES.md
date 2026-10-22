# Submission notes — V14

## Changed C2-first scorer

V14 ships a new **5,800-tree native LightGBM scorer**, with 408 internal
features for 136 raw inputs. Its score is .5 × the retained 3,800-tree
V10/V12 base + .25 × the completed primary 1,000-tree group + .25 × the
completed independent 1,000-tree group. Each new group contains 600
query-bagging extra trees and 400 balanced-domain trees. The balanced
component matches exactly and is exported twice; effective contributions
are .5 base, .125 each bagging run, and .25 balanced.

This is **partial refitting**: the base retains its original V10 fit and
historical reproduction evidence; only the two added members were trained
again in each of the two new runs. No further fitting occurred in recovery.
Final SHA256: `1ead2b749463e3fad64157878c0f3c90644cdcfae2dd373d1616a3e3096a4e5c`.

## Explicit training-variation disclosure

The two new group fits share data/configuration/source/runtime identities,
but their model bytes differ. Of their first 600 query-bagging trees, 599
differ; all last 400 balanced trees match. **Tree training is not
byte-reproducible in these runs.** A root cause has not been established.
Original verification `1112542.pbshpc` failed before installation or
DEV evaluation and is preserved in the actual five-attempt scheduler record.

A separate recovery was frozen and pushed at `e80391302bd1d5d9d4c24b1371e68c9c5579dc3b` before
DEV. Both completed groups are included with equal weights, without choosing
a run using training or DEV accuracy. Two exports from the SAME three fixed
input artifacts match complete bytes, and inference is deterministic. This
is reproducible export, **not independent fresh-training reproduction**.
See docs/EXPERIMENTS_V14_RECOVERY.md, results/v14_variation_audit.json,
results/tree_composition_v14.json and results/execution.json. The graded
neural seed/reproducibility implementation remains unchanged.

## Known-result TRAIN choice and limits

After V13 retained unchanged models, the user clarified C2-first scoring
priority. The original V14 family choice, complement_30, was retrospectively
selected from already completed five-fold TRAIN results and pushed at
`b14de0ea2f8ae91213cab77da05ea2581c8fa61f` before fitting. Its original V13 joint gate failed;
the failed gate/report/audit have not been changed. The small clean-accuracy
tradeoff was accepted under the clarified C2-first preference.

| Existing TRAIN CV family metric | Evaluated control | Selected family |
|---|---|---|
| Clean nDCG@10 | 0.487559972 | 0.487052266 |
| Mean affine/dropout/noise nDCG@10 | 0.477024990 | 0.478345688 |
| Half-clean/half-diagnostic score | 0.482292481 | 0.482698977 |

The family improves all three public diagnostic means and all five paired
folds. These are reused TRAIN folds with repeated-selection risk, not a new
independent test. The 20 native OOF checks passed for the original selected
family, maximum error 2.93e-14. They do not freshly
measure the two-run stochastic ensemble's generalization. Final arithmetic
and deterministic-inference checks use the actual fixed group artifacts.
Public diagnostics do not reconstruct staff C2 or predict private score/rank.
The private score can improve or regress.

## Artifact-specific data and stopping declaration

The retained base and new tree members were fitted on released **Fold1
TRAIN and DEV**: 723,412 TRAIN rows/6,000 queries plus 235,259 DEV rows/2,000
queries, totaling 958,671 original rows/8,000 split-namespaced queries and
136 raw features. Training augmentation adds copies in memory. DEV fitting
is disclosed and permitted; tree DEV accuracy is **in-sample**. No DEV tuning,
early stopping, post-fit accuracy choice, official test, other fold, hidden
benchmark identity, outside label or distillation target was used.
The formal report remains deferred.

| Component | Fit data/provenance | Stopping/selection |
|---|---|---|
| 3,800-tree base | TRAIN + DEV; retained original V10 bytes | Historical TRAIN CV; no new fit |
| Two 1,000-tree groups | TRAIN + DEV; two fresh complete runs, differing bagging outputs | Fixed archived 600/400 counts/settings/seeds; include BOTH equally |
| pointwise_model.pt | TRAIN only; exact original bytes retained | Original TRAIN fold0; six MSE epochs; no V14 fit |
| pairwise_model.pt | TRAIN only; exact V12 bytes retained | V12 TRAIN CV; 30 MSE warm-up +2 RankNet epochs; no V14 fit |
| fusion_config.json | Exact V11 TRAIN-selected weights retained | No V14 change |
| recipe.py | Supplied benchmark TRAIN at grading | Unchanged validation-selected capacity/stopping |

## Final checks and DEV interpretation

One final DEV interface evaluation followed the committed recovery choice.
The tree value is IN-SAMPLE after TRAIN+DEV fitting and did not choose the
shipped model. No new private score has been observed at packaging time. The in-sample tree DEV value decreased from the baseline 0.682399113 to 0.643507117; this result did not change the frozen ensemble choice.

| Metric | Actual value |
|---|---|
| Tree nDCG@10 (IN-SAMPLE) | 0.643507117 |
| Pairwise nDCG@10, retained TRAIN-only fit | 0.452027035 |
| Pointwise nDCG@10, retained TRAIN-only fit | 0.431707743 |
| Fusion nDCG@10 | 0.309012995 |
| Fusion MAP@10 | 0.129671266 |

All 185 tests pass locally and on HPC; current-choice CI 39818200321 passed 185.
Actual attempts are four successful jobs and the original failed identity
verification, all preserved; recovery verification `1112616.pbshpc`
passed. Staff loading passes on local Python 3.12.14
and HPC Python 3.9.21. Local/HPC hashes match; preparation
0.394s, max reranking 0.837s
and max fusion 0.001s satisfy 60s/2s limits.
Neural artifacts/history, fusion, C1 recipe, graded source and pristine staff
harness remain exact. Eight evaluated ZIPs and the V13 audit ZIP are preserved.
The last supplied staff result was V12 #597: C2 .3795, B .6824, held-out P .4419,
live .4722/.65, 10/10 correctness. V14 has no staff result at packaging.

## Transfer policy and limits

Two supplied-validation-selected leaf counts adapt to training query count:
15/31, 31/63 or 63/127 at cutoffs 200 and 2,000. Features are raw plus query
percentiles, regardless of their meaning. Minimum leaf size scales with row
count. Each fit has a deterministic 200–500-tree cap based on rows × raw
width, patience 40, L2 10, learning rate 0.05, truncation 13 and four threads.
The near-maximum TRAIN-derived 300-column check finishes in
208.2s at 4.19 GiB,
with caveats recorded in `results/recipe_budget_v3.json`. Synthetic checks
prove portability; the unknown datasets remain staff-only.

## AI-use disclosure

OpenAI Codex assisted with specification reading, algorithms, implementation,
tests, environment setup, scheduled HPC training, diagnostics, interpretation,
documentation and packaging. The student must understand and explain these
methods during the course walkthrough. Report writing remains deferred.

## Code-provenance statement

The original starter supplies the unchanged harness, network, standardizer,
grouped folds and metrics. Hand-written fusion, MSE and RankNet follow the
assignment and primer. Query transforms, metric swap weights, perturbations,
training augmentation, native numeric-tree export, bounded plain-MLP averaging and adaptive capacity/budget
rules are implemented directly. No external fusion library, neural ranking trainer, solution code,
other student's code, or extra labeled dataset is used. LightGBM 4.6's
permitted ranking objectives implement the tree learners, including V9's
XE_NDCG_MART candidates. Official references:
[MSLR](https://www.microsoft.com/en-us/research/project/mslr/),
[LightGBM parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html),
[IIT Delhi HPC](https://supercomputing.iitd.ac.in/).
