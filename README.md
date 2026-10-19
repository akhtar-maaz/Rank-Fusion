# A3 — Rank Fusion and Learning to Rank

Hand-written fusion, pointwise MSE and pairwise RankNet, with an offline-trained
LightGBM LambdaMART entry and adaptive transfer recipe. The current submission
is v6. Formal report writing remains deferred.

## Results and data declaration

Every shipped MSLR model is fitted on **Fold1 TRAIN only**: 723,412 original
rows, 6,000 original queries and 136 raw features. Model and stopping choices
use TRAIN folds only. DEV is evaluated once after configurations are frozen,
over 235,259 rows and 2,000 queries. No DEV fitting, tuning or early stopping,
official test split, other fold, undisclosed benchmark, external relevance
label or distillation target is used for MSLR training or selection.

| Raw DEV metric | v5 | v6 |
|---|---:|---:|
| Fusion nDCG@10 | 0.307785 | 0.307785 |
| Fusion MAP@10 | 0.128917 | 0.128917 |
| LTR nDCG@10 | 0.497649 | 0.497932 |
| Pointwise nDCG@10 | 0.431708 | 0.431708 |
| Pairwise nDCG@10 | 0.444853 | 0.447643 |

Track B rounds raw scores to 0.005 bands before class percentile ranking.
A raw gain alone does not establish a leaderboard-score gain. User-reported
v5 feedback is in `results/leaderboard_v5.json`: fourth of 21 teams, C2
0.3724, pairwise held-out 0.4333, live score 0.4695/0.65, 7/7 conformance
and 10/10 correctness. V6's C2/live score still requires staff evaluation.

All **96 tests** pass locally and on the college server. Actual frozen
DEV scoring uses the submission interface and meets per-call limits.
Promoted artifacts have independent byte-identical full-TRAIN refits;
retained artifacts preserve their v5 hashes. Neural files load with the
pristine staff MLP loader under Python 3.12. See `results/execution.json`,
`results/artifacts.json` and the reproducibility records.

## Final tree model

The promotion rule selected `v5_plus_3_800_weight0.25`.

| Member | Features | Trees | Weight | Augmentation |
|---:|---|---:|---:|---|
| 1 | query_rank | 800 | 0.09375 | none |
| 2 | query_normalized | 200 | 0.09375 | none |
| 3 | query_rank | 800 | 0.1875 | half_query_dropout_noise |
| 4 | query_all | 800 | 0.375 | half_query_mixed |
| 5 | query_all | 800 | 0.25 | full_query_mixed |

| Member | Leaves | Minimum leaf | L2 | Feature fraction | Seed | Truncation |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 127 | 100 | 10 | 0.8 | 0 | 30 |
| 2 | 127 | 100 | 10 | 0.8 | 0 | 30 |
| 3 | 127 | 100 | 10 | 0.8 | 0 | 13 |
| 4 | 127 | 100 | 10 | 0.8 | 0 | 30 |
| 5 | 127 | 100 | 10 | 0.8 | 0 | 30 |

Every member uses learning rate 0.05, row fraction 0.8 with frequency 1,
unlimited depth, deterministic row-wise fitting and four threads. Fixed
tree counts replace early stopping. The native artifact has
3400 trees, 408 internal features
and 136 raw inputs, with `query_all` metadata.
Split indices are remapped and leaf outputs weighted in one ordinary
LightGBM text model; constituent files are unnecessary for grading.

`query_rank` contains raw values and tied-midrank percentiles.
`query_normalized` contains query min-max values and percentiles.
`query_all` contains raw, percentile and min-max blocks. Finite constants
map to 0.5; missing values remain missing. Each transformation sees only
that query's candidates. Sorting temporaries use at most 16 columns.

## Tree augmentation and TRAIN-only selection

Augmentation retains originals and appends a separate query group with
copied labels, chosen by SHA256 of the opaque query ID. `half_query_mixed`
uses seed 20261020 and copies about half the fit queries; `full_query_mixed`
uses seed 20261021 and copies every fit query. Both choose affine, dropout
or noise by hash, independently of relevance labels. V4's dropout/noise
scheme and seed 20261019 remain unchanged. Augmentation follows splitting,
so validation queries never enter fitting. Exact full-TRAIN counts appear
in `results/execution.json`.

The frozen [v6 protocol](docs/EXPERIMENTS_V6.md) specifies 49 settings:
one exact v5 control, 12 new single-model checkpoints and 36 fixed averages
with v5. All 49 settings contain five complete folds, with no runtime
exclusions. Splits group by query, seed 0. Training metrics score original
clean queries only.

Keep settings within 0.0025 clean nDCG of the best complete mean and no
more than 0.0025 below v5 on any diagnostic. Maximize half clean plus half
the mean of affine/dropout/noise, with ties favoring fewer trees. Require
at least 0.0005 mixture gain or preserve v5. No partial-fold mean enters
selection. The exact v5 control reproduces previous TRAIN-CV results.

| TRAIN-CV condition | v5 | v6 |
|---|---:|---:|
| clean | 0.487398 | 0.487343 |
| affine | 0.474040 | 0.476724 |
| feature_dropout | 0.476076 | 0.476436 |
| noise | 0.472108 | 0.473732 |

Diagnostic mean changes from 0.474075 to
0.475631; mixture from 0.480737 to
0.481487. Selected clean sample SD is
0.010004, training mean 0.758586,
and validation/training ratio 0.642436.
The exported native model matches explicit held-out TRAIN averaging on
every fold, maximum score error
2.93e-14, with identical nDCG.

Diagnostics use positive affine changes, 10% query-column loss and Gaussian
noise with SD 5% of query-feature SD, selection seed 1729. Their definitions
are public in `scripts/robustness.py`. They are **not the staff C2 construction**.
Reused-fold adaptive selection and the training/validation gap limit confidence;
these estimates are not an unbiased new held-out test or a hidden-score guarantee.

| Post-selection perturbation seed 271828 | v5 | v6 |
|---|---:|---:|
| affine | 0.474720 | 0.477178 |
| feature_dropout | 0.477312 | 0.477835 |
| noise | 0.471619 | 0.472791 |
| Mean | 0.474550 | 0.475935 |

This post-selection check did not choose or change the artifact. It measures
sensitivity to perturbation randomness on reused TRAIN validation queries.

## Fusion and neural models

Fusion keeps the v2 weighted CombSUM: per-query min-max normalization;
BM25 whole-document 0.25, BM25 title 0.25, LMIR.DIR 0.5 and LMIR.JM 0.
Fusion and pointwise artifact hashes remain unchanged. Pointwise remains
136→128→64→1 ReLU, fitted with six full-TRAIN MSE epochs at learning rate
0.001, selected originally on internal TRAIN fold 0.

Pairwise uses the same bounded plain MLP. The promotion rule selected
`longer_warmup`, which uses 10 supervised
MSE warm-up epochs at 0.001, then 2 nDCG-weighted RankNet
epochs at 0.0001, with a fresh Adam optimizer. Batch size is
1024, seed 0, and at most 300
distinct strictly ordered pairs per query per epoch. Weights are absolute
nDCG@10 swap changes; graded `ranknet_loss` stays plain stable softplus.
No model scores or outside labels become training targets.

| Pairwise TRAIN-CV candidate | Mean | Sample SD | Train mean |
|---|---:|---:|---:|
| v5_pairwise_control | 0.431179 | 0.008981 | 0.448472 |
| more_pairs | 0.433317 | 0.007512 | 0.458083 |
| slow_finetune | 0.432611 | 0.008253 | 0.449355 |
| longer_warmup | 0.435046 | 0.007693 | 0.455417 |

All four neural configurations use fixed epoch counts on all five TRAIN
folds, with fold-local standardization and no validation-based stopping.
The highest mean must improve at least 0.001 and win at least three folds
against v5; otherwise keep v5. The control's fold 0 reproduces its earlier
0.444760168455 result.
The selected mean is 0.435046, compared
with control 0.431179. Released DEV was
excluded from the choice. Full results are in `results/neural_selection_v6.json`.

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

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/smoke_test.sh
python scripts/train_model.py --train data/train --config results/lambdamart_cv.json
python scripts/fit_pairwise_v6.py --train data/train --out submission/pairwise_model.pt --history-out runs/pairwise_refit.json
# Only evaluate DEV after the configuration is frozen:
python scripts/evaluate_models.py --features data/dev/vali.txt
python scripts/run_recipe.py
python scripts/package_submission.py --out dist/rank-fusion-submission-v6.zip
```

On Linux, install the CPU-only Torch wheel first if appropriate:
`python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu`.
On macOS install OpenMP with `brew install libomp`; the smoke script shares
Torch's OpenMP runtime and trains only a temporary copy. Other macOS scripts
importing both Torch and LightGBM need `DYLD_LIBRARY_PATH` pointing to
`torch/lib` in the active environment.

Obtain the official archive from [Microsoft Research](https://www.microsoft.com/en-us/research/project/mslr/)
(MD5 `97c5d4e7c171e475c91d7031e4fd8e79`) and run
`python scripts/prepare_mslr.py --archive /path/to/MSLR-WEB10K.zip --out data`.
Preparation opens only Fold1 TRAIN and validation entries. Real data,
downloads, environments and intermediate runs stay out of Git and ZIPs;
original repository files and toy data are retained. More execution details
and neural reproduction commands are in [docs/HPC.md](docs/HPC.md).

`prepare()` loads the artifact and raises if missing/corrupt. Rankings return
unique supplied candidate IDs with stable ties. Frozen DEV preparation and
maximum reranking times are 6.193s and
0.465s, within the 60s/2s limits.

## AI-use disclosure and code provenance

OpenAI Codex assisted with reading the specification, implementation, tests,
HPC execution, experiment interpretation, documentation and packaging.
The unchanged starter supplies the harness, MLP, standardizer, metrics and
grouped-fold helper. Fusion, MSE and RankNet follow the assignment and primer.
Percentiles, swap weighting, diagnostic perturbations, training augmentation,
native tree averaging and adaptive recipe logic are implemented directly; no external fusion or
neural ranking trainer is copied. LightGBM's permitted `lambdarank` objective
trains the trees. Consulted references include the official dataset page,
[LightGBM 4.6 parameter documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html),
and the [IIT Delhi HPC guide](https://supercomputing.iitd.ac.in/).
See [docs/SUBMISSION_NOTES.md](docs/SUBMISSION_NOTES.md) for the declarations.
