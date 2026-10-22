# Submission notes — V13

## Data declaration

The shipped tree was fitted on released **Fold1 TRAIN and DEV**: 723,412
TRAIN rows/6,000 queries and 235,259 DEV rows/2,000 queries, totaling
958,671 original rows/8,000 split-namespaced queries and 136 raw features.
The assignment permits disclosed DEV fitting. Tree DEV accuracy is
**in-sample**, not evidence of unseen-query improvement. Tree status:
exact evaluated V10/V12 artifact retained; no V13 final tree fit. Retained artifacts are preserved copies, not fresh refits.

All new V13 selection uses released TRAIN only, five query-grouped folds,
seed 0. There is no DEV tuning, early stopping or post-fit choice. No official
test, other fold, hidden benchmark identity, outside relevance label or
distillation target was used. The formal assignment report remains deferred.

| Artifact | Fitting data | Selection and stopping |
|---|---|---|
| `model.txt` | TRAIN + DEV; retained V10 bytes | TRAIN-only frozen 85-setting comparison; fixed 3800 trees; no early stopping |
| `pointwise_model.pt` | TRAIN only; retained exact bytes | Original TRAIN fold 0; six MSE epochs; no V13 fit |
| `pairwise_model.pt` | TRAIN only; retained V12 bytes | V12 five-fold TRAIN selection; 30 MSE warm-up + 2 RankNet epochs; no V13 fit |
| `fusion_config.json` | Weights selected on TRAIN; exact V11 bytes retained | No V13 fusion change |
| `recipe.py` | Staff-supplied benchmark TRAIN | Unchanged; supplied validation selects capacity/stopping |

## V13 C2-focused TRAIN comparison

The [frozen V13 plan](EXPERIMENTS_V13.md) and implementation preceded
new measurements. Complete TRAIN-only choices were audited, committed and
pushed at `85e654a3a46a0edfff5a011fc4bd4d1e38c148d7` before final fits or new DEV.
See `results/lambdamart_cv_v13.json`, `results/v13_selection_audit.json`,
`results/v13_source_parts/` and `results/identity_preflight_v13.json`.

The 85 fixed settings jointly mix the historical six-member control,
query-bagging or node-subsampling extra trees, balanced training domains,
and query-normalized stress training. Quarter-step four-basis weights are
enumerated and identical specifications deduplicated. Eleven existing source
checkpoints and 77 report/OOF/model cache files were verified. No new CV
models were fitted; these are reused TRAIN folds, not a new independent test.

Selection maximizes mean affine/dropout/noise accuracy, subject to clean and
per-diagnostic floors of control minus .0025. Its single highest proposal
must gain .001 diagnostic mean, .0005 historical .5clean+.5diagnostic score,
and three positive paired diagnostic folds. Actual diagnostic gain:
0.001320699; historical joint gain: 0.000406497;
positive folds: 5/5. Promotion: **False**. No runner-up rescue.
The exact control reproduced its archived TRAIN metrics and member settings.
Native scores matched explicit averaging in all five folds and all four
scenarios, at maximum absolute error 2.75e-14.

These public diagnostics do not reconstruct the staff's five hidden C2
versions. Selection on reused folds is subject to multiple-comparison risk;
it cannot establish a private C2 score or class rank.

| Final local DEV metric | V13 value | Interpretation |
|---|---|---|
| Fusion nDCG@10 | 0.309012995 | Retained TRAIN-selected V11 weights |
| Fusion MAP@10 | 0.129671266 | Retained TRAIN-selected V11 weights |
| Tree nDCG@10 | 0.682399113 | IN-SAMPLE after TRAIN+DEV fitting |
| Pointwise nDCG@10 | 0.431707743 | Retained TRAIN-only fit |
| Pairwise nDCG@10 | 0.452027035 | Retained V12 TRAIN-only fit |

Latest supplied feedback is evaluated V12 submission597: rank 10/35,
held-out P .4419, C2 .3795, B .6824, live .4722/.65 and 10/10 correctness
(`results/leaderboard_v12.json`). V13 has no staff result at packaging time.

## Shipped tree and retained components

The shipped tree has 3800 trees for 136 raw inputs. Its exact
configuration is `configs/v10_tree.json`, and fit record `results/tree_fit_v10.json`.
SHA256: `f2fa40faac67aaab067bb43eb169fd7d7db5a13c5eeb0a6d248c88f941e44864`. Current
`results/lambdamart_cv.json` separates TRAIN selection from final fitting.
Independent-fit evidence is newly generated only when the tree is promoted;
retained V10 evidence remains historical.

| Member | Feature mode | Trees | Weight | Augmentation |
|---|---|---|---|---|
| 1 | query_rank | 800 | 0.0703125 | none |
| 2 | query_normalized | 200 | 0.0703125 | none |
| 3 | query_rank | 800 | 0.140625 | {'name': 'half_query_dropout_noise', 'seed': 20261019} |
| 4 | query_all | 800 | 0.28125 | {'name': 'half_query_mixed', 'seed': 20261020} |
| 5 | query_all | 800 | 0.1875 | {'name': 'full_query_mixed', 'seed': 20261021} |
| 6 | query_all | 400 | 0.25 | {'name': 'full_query_mixed', 'seed': 20261021} |

Every member keeps its archived parameters, seeds, augmentation and fixed
tree count. Models are compiled into one native LightGBM scorer; query
transforms inspect only that query's candidates. No special hidden-data
detection or retraining is used during evaluation.

Pairwise remains the V12 plain 136→256→128→1 ReLU model, compiled from equal
seeds 0/41, each 136→128→64→1 with 30 MSE warm-up epochs at .001 followed by
two weighted RankNet epochs at .0001, cap 300/batch 1024. Pointwise remains
136→128→64→1 with six MSE epochs at .001. Their exact artifacts and history
are preserved. Fusion retains V11 minmax CombSUM weights .3 BM25 whole document,
.3 BM25 title, .4 LMIR.DIR, and 0 LMIR.JM. Graded hand-written fusion/loss/pair/training
code, corrected deterministic initialization and C1 recipe are unchanged.

## Verification

All 185 tests pass locally and on the college server. All 2 actual
V13 PBS jobs succeeded; actual requested/observed resources and fitting status
are in `results/execution.json`. Source CI [39805019136](https://github.com/akhtar-maaz/Rank-Fusion/actions/runs/39805019136) passed.
Local/HPC content hashes agree, ranking is deterministic, and pristine staff
loading passes on local Python 3.12.14 and HPC Python 3.9.21. Preparation took 9.570s;
maximum fusion 0.001s and reranking
0.457s satisfy 60s/2s limits. All eight
earlier evaluated ZIPs are preserved.

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
