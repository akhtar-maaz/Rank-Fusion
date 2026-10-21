# A3 — Rank Fusion and Learning to Rank

Current submission: V10. The formal report remains deferred.

## Data declaration

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

## Final local DEV measurement

| Local DEV metric | V10 value | Interpretation |
|---|---|---|
| Fusion nDCG@10 | 0.307785 | Unchanged TRAIN-selected configuration |
| Fusion MAP@10 | 0.128917 | Unchanged TRAIN-selected configuration |
| Tree nDCG@10 | 0.682399 | In-sample: DEV labels used for tree fitting |
| Pointwise nDCG@10 | 0.431708 | Unchanged TRAIN-only fit |
| Pairwise nDCG@10 | 0.449922 | Unchanged TRAIN-only fit |

Latest supplied feedback is V9 submission 295:
rank 11/34, tree 0.4984,
held-out pairwise 0.4400, C2 0.3790,
live score 0.4160/0.65, 7/7 conformance and 10/10
correctness (`results/leaderboard_v9.json`). It predates this final fit.

## Final tree method and evidence

The [V10 protocol](docs/EXPERIMENTS_V10.md) freezes the exact selected
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

## Adaptive C1 recipe

`submission/recipe.py` reads only its two supplied files, handles arbitrary
feature width and labels 0–4, and uses raw values plus query percentiles.
It validates two capacities: 15/31 leaves below 200 training queries,
31/63 below 2,000, otherwise 63/127. Validation selects capacity and the
best iteration; exact ties keep fewer leaves.

Both fits use learning rate 0.05, L2 10, minimum leaf size
`max(20, min(200, training_rows // 2000))`, feature fraction 0.9 below 100
raw features (otherwise 0.8), row fraction 0.9, truncation 13, seed supplied
by staff, four threads and patience 40. Each fit's deterministic tree ceiling
is `max(200, min(500, 30_000_000_000 // (training_rows * raw_width)))`.
There are no wall-clock-based selection decisions. Native missing-value
handling preserves NaNs.

Three constructed TRAIN-only nested holdouts improve mean outer nDCG from
0.448871 to
0.450507. These single-holdout method
checks do not estimate hidden-benchmark performance. Records are in
`results/recipe_domains_v3/` and `results/recipe_selection_v3.json`.
The wide compute check fits 499,918 rows × 300 derived
columns, validates 142,390 rows, finishes in
208.2 seconds and uses
4.19 GiB peak RSS. It repeats/rescales
TRAIN columns in memory; parsing uses the original 136-feature file.
It is a budget check, not an unknown-benchmark simulation. The provided
40/90/250-feature synthetic portability tests also pass.

## Setup and reproduction

Exact tree reproduction uses the recorded HPC Python 3.9.21, NumPy 2.0.2
and LightGBM 4.6.0. Staff neural loading is separately verified with local
Python 3.12.14. V10 requires no neural training command; original V8 neural
artifacts are shipped.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/smoke_test.sh
python scripts/train_model.py --train data/train --dev data/dev --use-dev --config configs/v10_tree.json --out runs/reproduced_v10/model.txt --fit-record runs/reproduced_v10/fit.json
# DEV tree accuracy is in-sample after this fit:
python scripts/evaluate_models.py --features data/dev/vali.txt
# Package the reviewed, committed revision:
git archive --format=zip --output=dist/rank-fusion-submission-v10.zip HEAD
```

Obtain the official MSLR archive using `scripts/prepare_mslr.py`; preparation
opens only released Fold1 TRAIN/validation entries. See
[docs/HPC.md](docs/HPC.md) for the college environment. On macOS install
OpenMP and share Torch's runtime for combined Torch/LightGBM loading, as
the smoke script arranges. On Linux the recorded CPU Torch wheel is used.
Real data, environments, intermediate runs and Git metadata are excluded
from the archive. Original starter paths are retained.

## AI-use disclosure and code provenance

OpenAI Codex assisted with reading the specification, implementation, tests,
HPC execution, experiment interpretation, documentation and packaging.
The unchanged starter supplies the harness, MLP, standardizer, metrics and
grouped-fold helper. Fusion, MSE and RankNet follow the assignment and primer.
Percentiles, swap weighting, diagnostic perturbations, training augmentation,
native tree averaging, bounded plain-MLP averaging and adaptive recipe logic are implemented directly; no external fusion or
neural ranking trainer is copied. LightGBM's permitted ranking objectives `lambdarank` and `rank_xendcg`
train the tree candidates; final member objectives are documented above. Consulted references include the official dataset page,
[LightGBM 4.6 parameter documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html),
and the [IIT Delhi HPC guide](https://supercomputing.iitd.ac.in/).
See [docs/SUBMISSION_NOTES.md](docs/SUBMISSION_NOTES.md) for the declarations.
