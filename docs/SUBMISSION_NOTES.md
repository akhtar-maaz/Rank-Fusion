# Submission notes — V9

## Training-data declaration

Every shipped MSLR model is fitted on **Fold1 TRAIN only**: 723,412 original
rows, 6,000 original queries and 136 raw features. Tree choices and stopping
counts use TRAIN folds only. V9 evaluates released DEV (235,259 rows,
2,000 queries) once after its tree choice freezes. No DEV fitting, tuning
or early stopping, official test split, other MSLR fold, outside relevance
label, distillation target or undisclosed benchmark enters fitting or selection.
The unchanged neural models retain their original TRAIN-only selection and
training declarations. No attempt is made to identify hidden C1 benchmarks;
the separate C1 recipe uses only its staff-supplied TRAIN/validation inputs.
User feedback consists solely of aggregate leaderboard numbers.

| Artifact | Fit data | Selection / stopping |
|---|---|---|
| `model.txt` | Full released Fold1 TRAIN | Five grouped TRAIN folds; frozen 3800-tree configuration |
| `pointwise_model.pt` | Original full Fold1 TRAIN | Unchanged V8 bytes; original internal TRAIN fold 0/six MSE epochs |
| `pairwise_model.pt` | Original full Fold1 TRAIN | Unchanged V8 warm-up20 seed0/41 average; no V9 fit |
| `recipe.py` | Staff-supplied benchmark TRAIN | Unchanged recipe; supplied validation chooses capacity/stopping |

## Final tree model

The frozen V9 rule retained the exact evaluated V8 tree bytes; current choice is `cached_v8_control`.

| Member | Features | Trees | Weight | Augmentation (seed) |
|---|---|---|---|---|
| 1 | query_rank | 800 | 0.0703125 | none |
| 2 | query_normalized | 200 | 0.0703125 | none |
| 3 | query_rank | 800 | 0.140625 | half_query_dropout_noise (20261019) |
| 4 | query_all | 800 | 0.28125 | half_query_mixed (20261020) |
| 5 | query_all | 800 | 0.1875 | full_query_mixed (20261021) |
| 6 | query_all | 400 | 0.25 | full_query_mixed (20261021) |

| Member | Objective | Leaves | Max depth | Minimum leaf | L2 | Path smoothing | LambdaRank truncation | Seed | Other settings |
|---|---|---|---|---|---|---|---|---|---|
| 1 | lambdarank | 127 | unlimited | 100 | 10 | 0 | 30 | 0 | none |
| 2 | lambdarank | 127 | unlimited | 100 | 10 | 0 | 30 | 0 | none |
| 3 | lambdarank | 127 | unlimited | 100 | 10 | 0 | 13 | 0 | none |
| 4 | lambdarank | 127 | unlimited | 100 | 10 | 0 | 30 | 0 | none |
| 5 | lambdarank | 127 | unlimited | 100 | 10 | 0 | 30 | 0 | none |
| 6 | lambdarank | 127 | unlimited | 100 | 10 | 0 | 30 | 0 | extra_trees; extra_seed=0 |

All members use learning rate 0.05, feature fraction 0.8, bagging fraction
0.8/frequency 1, seed 0, deterministic row-wise fitting and four threads.
Fixed counts replace early stopping. The shipped native artifact has
3800 trees, 408 internal features and 136 raw inputs.
For averages, feature indices are remapped and tree outputs weighted in one
ordinary numeric LightGBM model; no constituent files or custom inference
loader are needed. Each member's original training objective is recorded
above, including mixed-objective averages when selected. Native held-out
TRAIN scores match explicit averaging on all five folds, maximum error
2.4e-14, with identical nDCG.

`query_rank` uses raw values and tied-midrank percentiles; `query_normalized`
uses query min-max values and percentiles; `query_all` contains all three.
Constants map to 0.5, missing entries remain missing, and transforms use only
the current query's candidates. Sorting temporaries stay bounded to 16 columns.

## V9 TRAIN-only experiment

The [frozen protocol](EXPERIMENTS_V9.md) specifies two XE_NDCG_MART
(`rank_xendcg`) fits: 63 leaves, maximum depth 8, minimum leaf 100, L2 10,
objective seed 0; and two substantially shallower LambdaRank fits: 31 leaves,
maximum depth 5, minimum leaf 300, L2 30, path smoothing 100 and truncation 30.
For each objective/capacity, one fit uses clean TRAIN and one uses the
unchanged full mixed augmentation, seed 20261021. Checkpoints are fixed at
400/800/1200. No neural fits or new feature/augmentation algorithms are added.

Augmentation retains originals and adds one separate query copy, chosen by
opaque-query SHA256 as positive affine change, column dropout or Gaussian
noise. Scales are 0.5–2 with shifts ±0.25 query-feature range; dropout 10%;
noise SD 5% of query-feature SD. Labels are copied without inspection and
copies enter only after query splitting. Earlier members retain their original
augmentation settings. Full-TRAIN row/query counts are in `results/execution.json`.

Compare 49 planned settings: exact V8 control, 12 new single checkpoints and
36 fixed V8/new averages at new weights 0.25/0.5/0.75. There are
49 complete reported settings; runtime exclusions:
[]. Every reported setting contains five
query-grouped TRAIN folds, split seed 0. Reported fit metrics score original
clean training queries. The exact V8 control reproduces archived specifications
and metrics within 1e-12; no partial-fold means enter selection.

Preserve V8's filters: within 0.0025 clean nDCG of the best complete mean and
no more than 0.0025 below control on any public diagnostic. Maximize half
clean plus half mean affine/dropout/noise; ties prefer fewer trees. That
winner must improve mixture by 0.0005, diagnostic mean by 0.001 and paired
fold mixture in at least three folds. Otherwise preserve exact V8 tree bytes
without searching a runner-up. The public diagnostic definitions and seed
1729 stay unchanged; these generic transforms do not reconstruct private C2.

The highest eligible candidate was `v8_plus_0_1200_weight0.25`: mixture gain +0.000645585, diagnostic mean gain +0.000162203, positive paired mixture gains in 5/5 folds. Mixture/diagnostic/paired gates: True/False/True.

| TRAIN-CV condition | V8 control | V9 shipped choice |
|---|---|---|
| clean | 0.487560 | 0.487560 |
| affine | 0.479000 | 0.479000 |
| feature_dropout | 0.477154 | 0.477154 |
| noise | 0.474921 | 0.474921 |

Diagnostic mean: 0.477025 → 0.477025;
mixture: 0.482292 → 0.482292.
Selected clean sample SD is 0.009312, fit mean
0.732114, validation/fit ratio 0.665962.
Repeated adaptive TRAIN-fold reuse and the generalization gap limit confidence.
These are not fresh held-out estimates or private-score guarantees.

The seed-902617 descriptive check was skipped because the exact V8 tree was retained.

## Unchanged V8 neural models, fusion and core functions

**V9 performs no neural training, neural selection or neural refit.** The
pairwise, pointwise, fusion, corrected graded neural source and C1 recipe
retain exact evaluated V8 bytes. `results/neural_history.json` preserves
`results/neural_history_v8.json` byte-for-byte, with separate model widths/seeds.

Pairwise retains V8's `warmup20_seeds_0_41`: two independently trained
136→128→64→1 ReLU members, compiled as one ordinary 136→256→128→1 MLP by
concatenated first layers, block-diagonal second layers and weighted output.

| V8 member seed | Hidden sizes | Weight | MSE warm-up epochs | RankNet epochs | RankNet LR | Pair cap | Batch |
|---|---|---|---|---|---|---|---|
| 0 | 128→64 | 0.5 | 20 | 2 | 0.0001 | 300 | 1024 |
| 41 | 128→64 | 0.5 | 20 | 2 | 0.0001 | 300 | 1024 |

Each V8 member used supervised MSE warm-up at 0.001, followed by fresh Adam
and handwritten nDCG-weighted RankNet. Targets remain relevance labels only;
the graded `ranknet_loss` stays plain stable softplus. V8's selected five-fold
TRAIN mean was 0.440371; this is historical
evidence, not a new V9 measurement. Its original full fit and independent
reproduction remain `results/pairwise_refit_v8.json` and
`results/pairwise_reproducibility_v8.json`; they must not be presented as V9 fits.

Pointwise remains 136→128→64→1 ReLU, six MSE epochs at 0.001, originally
selected on internal TRAIN fold 0. Fusion retains min-max CombSUM weights:
BM25 whole-document 0.25, BM25 title 0.25, LMIR.DIR 0.5 and LMIR.JM 0.


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

## Verification and reproducibility

All 148 tests pass locally and on the server; actual frozen DEV respects
prepare/rerank limits. The tree has independent byte identity, from fresh
refitting if promoted or exact V8 copying if retained. No neural model was
refitted in V9. Earlier neural reproduction evidence remains versioned V8.
Protected V8 neural/core/fusion/recipe hashes and neural-history bytes match.
Nine actual PBS records document requests and observed resource use.
Source reports contain all five folds per checkpoint; current manifests
record native/staff loading and local/HPC hash agreement. Formal report
writing remains deferred.

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
