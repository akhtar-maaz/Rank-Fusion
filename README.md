# A3 — Rank Fusion and Learning to Rank

Hand-written fusion/MSE/RankNet, an offline gradient-boosted ranking entry,
and the unchanged adaptive transfer recipe. Current submission: V9.
Formal report writing remains deferred.

## Data declaration and final evaluation

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

| Raw local DEV metric | V8 | V9 |
|---|---|---|
| Fusion nDCG@10 | 0.307785 | 0.307785 |
| Fusion MAP@10 | 0.128917 | 0.128917 |
| Tree nDCG@10 | 0.498374 | 0.498374 |
| Pointwise nDCG@10 | 0.431708 | 0.431708 |
| Pairwise nDCG@10 | 0.449922 | 0.449922 |

Latest staff feedback is evaluated V8 submission 271:
rank 9/31, held-out pairwise
0.4400 (displayed staff DEV 0.4500),
C2 0.3790, live score 0.4576/0.65,
7/7 conformance and 10/10 correctness (`results/leaderboard_v8.json`).
All V9 model artifacts are byte-identical to evaluated V8. No model-score improvement is expected from a new upload; keeping the already-uploaded V8 submission is valid.
Track B uses 0.005 rounding bands before
class percentile ranking; local raw changes alone establish no new rank.

All **148 tests** pass locally and on the college server. Artifacts match
HPC hashes and pristine staff neural loading passes under Python 3.12.
A promoted tree has a byte-identical independent full-TRAIN refit; a retained
tree preserves exact V8 bytes and is copied, not refitted. Protected neural
and core files preserve V8 hashes. See `results/execution.json`.

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

The [frozen protocol](docs/EXPERIMENTS_V9.md) specifies two XE_NDCG_MART
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

Use the recorded runtime/versions for exact tree reproduction. V9 requires
no neural training command: evaluated V8 neural artifacts are shipped unchanged.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/smoke_test.sh
python scripts/train_model.py --train data/train --config results/lambdamart_cv.json --out runs/reproduced_v9/model.txt
# Only evaluate DEV after the TRAIN choice is frozen:
python scripts/evaluate_models.py --features data/dev/vali.txt
python scripts/package_submission.py --out dist/rank-fusion-submission-v9.zip
```

The reproduction command refits the frozen tree method. Actual V9 scheduled
jobs refit only if promoted and otherwise copy the exact V8 file; the executed
command is recorded in `results/lambdamart_reproducibility.json`. Reproducing
the original pairwise method is a separate **V8** operation using its original
selection, fit script and recorded CPU Torch runtime; it is not V9 training.

On Linux, install the CPU Torch wheel if appropriate:
`python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu`.
On macOS, install OpenMP; scripts importing Torch and LightGBM share Torch's
runtime through `DYLD_LIBRARY_PATH`, as the smoke script arranges for its
temporary copy. Real data, environments, Git metadata and intermediate fits
stay outside the submission; original starter paths are retained.

Obtain the official archive from [Microsoft Research](https://www.microsoft.com/en-us/research/project/mslr/)
(MD5 `97c5d4e7c171e475c91d7031e4fd8e79`) and use
`python scripts/prepare_mslr.py --archive /path/to/MSLR-WEB10K.zip --out data`.
Preparation opens only released Fold1 TRAIN/validation entries. See
[docs/HPC.md](docs/HPC.md) for scheduled execution details.

`prepare()` loads the native artifact and raises if missing/corrupt. Candidate
rankings return unique supplied IDs with stable ties. Frozen DEV preparation
and maximum reranking times are 0.555s and
0.491s, within the 60s/2s limits.

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
