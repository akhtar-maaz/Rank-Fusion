# Submission notes — V11

## Data declaration

The shipped tree is the exact evaluated V10 artifact, originally fitted on
released **Fold1 TRAIN and DEV**: 723,412 TRAIN rows/6,000 queries plus
235,259 DEV rows/2,000 queries, 958,671 original rows/8,000 split-namespaced
queries and 136 raw features. The assignment permits disclosed DEV fitting.
Tree DEV accuracy is therefore **in-sample**, not an unseen-query estimate.
V11 fits TRAIN-fold tree candidates but does not refit the final tree or any neural model. Historical V10 independent-fit
records remain original evidence, rather than new V11 refits.

All V11 tuning and selection use released TRAIN only, five query-grouped folds,
seed 0. The four new tree candidates and 33 fixed comparisons did not beat the
archived control under the predeclared gates. The exact V10 tree was retained.
Fusion alone changes, selected from 1,146 fixed TRAIN-only configurations.
The final DEV measurement did not choose weights, parameters or stopping.
No official test, other MSLR fold, outside relevance labels, distillation target
or undisclosed benchmark was accessed. The unchanged C1 recipe uses only its
staff-supplied inputs. The formal report remains deferred.

| Artifact | Fitting data | Selection / stopping |
|---|---|---|
| `model.txt` | TRAIN + DEV in V10; retained exact bytes | Historical TRAIN CV fixed 3,800 trees; V11 TRAIN comparison retained it; no early stopping |
| `pointwise_model.pt` | TRAIN only; retained exact bytes | Original TRAIN fold 0; six MSE epochs; no V11 fit |
| `pairwise_model.pt` | TRAIN only; retained V8 bytes | V8 TRAIN-only selection; no V11 fit |
| `fusion_config.json` | No trained scorer; weights selected on TRAIN | V11 fixed grid and paired-fold gates; no DEV tuning |
| `recipe.py` | Staff-supplied benchmark TRAIN | Unchanged; supplied validation chooses capacity/stopping |

## V11 selection and final local evaluation

The [frozen V11 protocol](EXPERIMENTS_V11.md) preceded measurements.
TRAIN-only choices and complete gate/native evidence were audited, committed
and pushed at `08f85262dd1d0f17b5be4461dbb74b4f1484eb74` before the final DEV run.
See `results/v11_selection_audit.json`, `results/lambdamart_cv_v11.json`,
`results/cv_v11/`, and `results/fusion_selection_v11.json`.

Fusion uses hand-written minmax CombSUM with weights .3 BM25 whole-document,
.3 BM25 title, .4 LMIR.DIR and 0 LMIR.JM. The TRAIN proxy (.8 nDCG + .2 MAP)
improves by 0.0006460764, with four of five paired folds positive, meeting the
fixed .0002/three-fold gates. That proxy does not compute class percentiles.
Required unweighted RRF/CombSUM/CombMNZ remain hand-written and tested;
weighted RRF was compared but not selected.

| Local DEV metric | V11 value | Interpretation |
|---|---|---|
| Fusion nDCG@10 | 0.309012995 | One final evaluation of TRAIN-selected weights |
| Fusion MAP@10 | 0.129671266 | One final evaluation of TRAIN-selected weights |
| Tree nDCG@10 | 0.682399113 | In-sample; exact retained V10 tree |
| Pointwise nDCG@10 | 0.431707743 | Retained TRAIN-only fit |
| Pairwise nDCG@10 | 0.449922200 | Retained TRAIN-only fit |

Latest supplied feedback is V10 submission 482: rank 10/35, B .6824,
held-out P .4400, C2 .3795, live .4827/.65, 7/7 conformance and 10/10
correctness (`results/leaderboard_v10.json`). V11 has no new staff evaluation
at packaging time. This round changes fusion only; no C2 or rank improvement
is established by the public diagnostics or local DEV metrics.

## Retained tree method and provenance

The six-member V10 specification remains `configs/v10_tree.json`, selected
using historical TRAIN-only folds before its original TRAIN+DEV fitting.

| Member | Features | Trees | Weight | Augmentation |
|---|---|---|---|---|
| 1 | query_rank | 800 | .0703125 | none |
| 2 | query_normalized | 200 | .0703125 | none |
| 3 | query_rank | 800 | .140625 | half_query_dropout_noise, seed 20261019 |
| 4 | query_all | 800 | .28125 | half_query_mixed, seed 20261020 |
| 5 | query_all | 800 | .1875 | full_query_mixed, seed 20261021 |
| 6 | query_all | 400 | .25 | full_query_mixed, seed 20261021; extra trees |

Members use LambdaRank, 127 leaves, minimum leaf 100, L2 10, learning rate .05,
feature/bagging fractions .8, bagging frequency 1, seed 0, deterministic
row-wise fitting, four threads and fixed counts. Truncation is 30 except
member 3 uses 13; extra-trees seed is 0. Weighted native export contains
3,800 trees/408 internal features for 136 raw inputs. `query_rank` combines
raw values and midrank percentiles; `query_normalized` combines minmax and
percentiles; `query_all` combines all three. Constants map to .5, NaNs stay
missing, and transforms inspect only the current query's candidates.

The rejected V11 candidates used percentile-only or query-normalized features,
with/without fixed stress augmentation. These are public scale/dropout/noise
diagnostics, not reconstructions of the staff's undisclosed C2 conditions.
Augmentation follows query splitting and preserves labels. Its deterministic
hashes use the declared split-namespaced IDs in original final V10 fitting.

Original V10 records `results/tree_fit_v10.json` and
`results/tree_fit_reproduced_v10.json` establish independent byte identity.
They are preserved, not relabeled as V11 fits. Model SHA256 is
`f2fa40faac67aaab067bb43eb169fd7d7db5a13c5eeb0a6d248c88f941e44864`. Current
`results/lambdamart_cv.json` retains V11 TRAIN comparisons with a separate
`final_refit` annotation explaining the retained V10 fitting provenance.

## Retained neural models and graded source

**V11 performs no neural training, selection or refit.** Pairwise retains
V8's `warmup20_seeds_0_41`, two 136→128→64→1 ReLU members compiled as one
ordinary 136→256→128→1 MLP by concatenated first layers, block-diagonal
second layers and weighted output.

| V8 seed | Hidden sizes | Weight | MSE warm-up epochs | RankNet epochs | LR | Pair cap | Batch |
|---|---|---|---|---|---|---|---|
| 0 | 128→64 | 0.5 | 20 | 2 | 0.0001 | 300 | 1024 |
| 41 | 128→64 | 0.5 | 20 | 2 | 0.0001 | 300 | 1024 |

The original V8 members used supervised MSE warm-up at 0.001 followed by
fresh Adam and handwritten nDCG-weighted RankNet. Targets remain relevance
labels only; graded `ranknet_loss` remains plain stable softplus. Original
selection, fitting and independent reproduction stay in
`results/neural_selection_v8.json`, `results/pairwise_refit_v8.json` and
`results/pairwise_reproducibility_v8.json`. They are V8 evidence, not V11
refits. Current neural-history bytes match the evaluated V9/V8 record.

Pointwise remains 136→128→64→1 ReLU with six MSE epochs at 0.001.
Corrected deterministic initialization and all graded neural training/loss
functions preserve their evaluated bytes. Neither neural artifact is refitted
or retuned in V11.

## Verification

All 163 tests pass locally and on the college server. Seven actual
V11 PBS jobs completed successfully: four CV children, fusion selection,
tree selection and final verification. No V11 final-artifact fitting or reproduction job was
submitted. Requested and observed resources are in `results/execution.json`.
The initial non-login launch lacked `qsub` in PATH before any scheduler process
started; a login shell resolved it, with no duplicate or failed PBS attempt.

Source CI [39751619056](https://github.com/akhtar-maaz/Rank-Fusion/actions/runs/39751619056)
passed. Local/HPC content hashes agree; ranking is deterministic. Staff neural
loading passes on HPC Python 3.9.21 and on pristine local
Python 3.12.14 with unchanged starter harness code.
DEV interface preparation took 8.344s,
maximum fusion 0.001s and maximum
reranking 0.506s, within 60s/2s limits.
Tree, neural artifacts/core, recipe, history and evaluated earlier ZIPs are
preserved; the fusion change matches the frozen TRAIN-only choice.

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
