# Submission notes — V10

## Training-data declaration

The final V10 tree is fitted on the union of released **Fold1 TRAIN and DEV**:
723,412 TRAIN rows/6,000 queries plus 235,259 DEV rows/2,000 queries,
958,671 original rows/8,000 split-namespaced queries and 136 raw features.
The assignment permits disclosed DEV fitting. Its nDCG on DEV is therefore
**in-sample**, and is not an estimate of unseen-query generalization.

The complete configuration, six member weights and 3,800-tree stopping count
were frozen from archived five-fold TRAIN-only selection before V10 fitting.
V10 performs no new hyperparameter search, DEV-based tuning, early stopping
or selection after the final DEV measurement. Earlier TRAIN-only CV results
remain historical measurements of the original fit procedure.

Pointwise, pairwise, fusion, corrected graded neural source, neural history
and C1 recipe preserve exact evaluated V8/V9 bytes. No neural fitting occurs
in V10. No official test split, other MSLR fold, outside relevance label,
distillation target or hidden benchmark enters fitting or selection. The
unchanged C1 recipe uses only its staff-supplied TRAIN/validation inputs.
User feedback consists of aggregate leaderboard numbers. Formal report
writing remains deferred.

| Artifact | Fitting data | Selection / stopping |
|---|---|---|
| `model.txt` | Released Fold1 TRAIN + DEV | Historical TRAIN-only CV; fixed 3,800 trees; no early stopping |
| `pointwise_model.pt` | Original Fold1 TRAIN only | Retained V8/V9 bytes; original TRAIN fold 0; six MSE epochs |
| `pairwise_model.pt` | Original Fold1 TRAIN only | Retained V8/V9 bytes; V8 TRAIN-only selection; no V10 fit |
| `fusion_config.json` | No fitted model parameters | Fixed TRAIN-selected fusion weights; no V10 tuning |
| `recipe.py` | Staff-supplied benchmark TRAIN | Unchanged recipe; supplied validation chooses capacity/stopping |

## Local DEV interpretation

| Local DEV metric | V10 value | Interpretation |
|---|---|---|
| Fusion nDCG@10 | 0.307785 | Unchanged TRAIN-selected configuration |
| Fusion MAP@10 | 0.128917 | Unchanged TRAIN-selected configuration |
| Tree nDCG@10 | 0.682399 | In-sample: DEV labels used for tree fitting |
| Pointwise nDCG@10 | 0.431708 | Unchanged TRAIN-only fit |
| Pairwise nDCG@10 | 0.449922 | Unchanged TRAIN-only fit |

## Final tree method and evidence

The [V10 protocol](EXPERIMENTS_V10.md) freezes the exact selected
specification in `configs/v10_tree.json`. Its historical label
`cached_v8_control` identifies the configuration; V10 actually
performs two fresh TRAIN+DEV fits, rather than copying the old tree.

| Member | Features | Trees | Weight | Augmentation | Truncation | Extra trees |
|---|---|---|---|---|---|---|
| 1 | query_rank | 800 | 0.0703125 | none | 30 | no |
| 2 | query_normalized | 200 | 0.0703125 | none | 30 | no |
| 3 | query_rank | 800 | 0.140625 | half_query_dropout_noise (seed 20261019) | 13 | no |
| 4 | query_all | 800 | 0.28125 | half_query_mixed (seed 20261020) | 30 | no |
| 5 | query_all | 800 | 0.1875 | full_query_mixed (seed 20261021) | 30 | no |
| 6 | query_all | 400 | 0.25 | full_query_mixed (seed 20261021) | 30 | yes |

All six members use LambdaRank, 127 leaves, unrestricted depth, minimum
leaf size 100, L2 10, learning rate 0.05, feature fraction 0.8, bagging
fraction 0.8/frequency 1, seed 0, deterministic row-wise fitting and four
threads. The extra-trees member uses extra seed 0. Fixed counts replace
early stopping. The final ordinary numeric LightGBM artifact contains
3,800 trees and 408 internal features for 136 raw inputs. Feature indices
and weighted outputs are compiled into one native model.

`query_rank` combines raw values with tied-midrank percentiles;
`query_normalized` combines query min-max values with percentiles;
`query_all` contains all three. Constants map to 0.5 and missing entries
remain missing. These transforms use only the current query's candidates.
TRAIN and DEV IDs are namespaced as `train:<qid>` and `dev:<qid>` before
fitting and augmentation. Existing deterministic augmentation hashes use
these declared names. Query copies preserve labels; their configurations
are unchanged from the archived TRAIN-only choice.

The primary and independent fits have identical input/configuration/source/
runtime identities and byte-identical model files. Their records are
`results/tree_fit_v10.json` and `results/tree_fit_reproduced_v10.json`;
`results/lambdamart_reproducibility.json` records the comparison. Each fit
uses 958,671 original rows and 8,000 original queries before member-specific
augmentation. The primary fit took 2022.1s. The model SHA256 is
`f2fa40faac67aaab067bb43eb169fd7d7db5a13c5eeb0a6d248c88f941e44864`.

`results/lambdamart_cv_v9.json` preserves historical TRAIN-only selection;
`results/lambdamart_cv.json` retains those measurements with a separate
`final_refit` annotation. V10 runs no new CV and makes no post-fit DEV-based
choice. A higher in-sample DEV score than V9's unseen-DEV score would not
establish a held-out improvement. Private C2 and class rank require staff
evaluation; no V10 leaderboard result is available at packaging time.

## Unchanged neural models, fusion and graded source

**V10 performs no neural training, selection or refit.** Pairwise retains
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
`results/pairwise_reproducibility_v8.json`. They are V8 evidence, not V10
refits. Current neural-history bytes match the evaluated V9/V8 record.

Pointwise remains 136→128→64→1 ReLU with six MSE epochs at 0.001. Fusion
retains min-max CombSUM weights: BM25 whole-document 0.25, BM25 title 0.25,
LMIR.DIR 0.5, LMIR.JM 0. Corrected deterministic initialization and all
graded neural training/loss functions preserve their evaluated bytes.

## Verification

All 152 tests pass locally and on the college server. Five actual PBS attempt records preserve the primary success, the original independent timeout, the original never-started dependency job, the successful unchanged independent retry and successful verification. Only the failed fit was retried, with the scheduler allocation extended from 90 minutes to three hours; the fitting method and four-CPU/16-GB request were unchanged.
Requested and observed resources are in `results/execution.json`.
Local/HPC hashes agree and native ranking is deterministic. Staff neural
loading passes on the HPC runtime and on pristine local Python 3.12.
DEV interface preparation took 4.936s;
maximum reranking took 0.461s,
within the 60s/2s limits. This is interface evidence; tree accuracy on those
DEV queries is in-sample. Protected neural/core/fusion/recipe files and all
evaluated earlier ZIPs are preserved.

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
