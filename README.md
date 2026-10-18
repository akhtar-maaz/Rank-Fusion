# A3 — Rank Fusion and Learning to Rank

Hand-written fusion, pointwise MSE and pairwise RankNet, with offline-trained
LightGBM LambdaMART and an adaptive transfer recipe. The current submission
is v3. Report writing remains deferred until submission feedback.

## Results and data declaration

Every submitted MSLR model is fitted on **Fold1 TRAIN only**: 723,412 rows,
6,000 queries and 136 raw features. Hyperparameters, features, epochs and
stopping choices use TRAIN folds only. DEV is evaluated after freezing the
configuration, over 235,259 rows and 2,000 queries. No DEV training, tuning
or early stopping is used. No held-out split, other fold, undisclosed
benchmark, external relevance label or distillation target is accessed.

| Raw DEV metric | v2 | v3 |
|---|---:|---:|
| Fusion nDCG@10 | 0.307785 | 0.307785 |
| Fusion MAP@10 | 0.128917 | 0.128917 |
| LambdaMART nDCG@10 | 0.491177 | 0.494193 |
| Pointwise nDCG@10 | 0.431708 | 0.431708 |
| Pairwise nDCG@10 | 0.444853 | 0.444853 |

The neural artifacts and fusion configuration are byte-identical to v2.
User-reported v2 feedback is retained in `results/leaderboard_v2.json`:
7/7 conformance, 10/10 correctness, held-out pairwise 0.4333, C2 0.3495,
live score 0.3317/0.65 and sixth place at that time. V3's C2/live result
requires staff evaluation; C1 is computed after the deadline. Local
diagnostics below do not estimate either hidden track.

All **76 tests** pass in the temporary-copy smoke harness. Final checks
include actual DEV scoring through the submission interface, staff-compatible
neural loading under Python 3.12, native-model averaging equivalence, and an
independent full-TRAIN refit that is byte-identical. See
`results/artifacts.json`, `results/execution.json`, and
`results/lambdamart_reproducibility.json`.

## Final tree model

V3 averages two independently TRAIN-fitted LambdaMART scorers:

| Features | Trees | Weight | Leaves | Minimum leaf size | L2 |
|---|---:|---:|---:|---:|---:|
| query_rank | 800 | 0.5 | 127 | 100 | 10 |
| query_normalized | 200 | 0.5 | 127 | 100 | 10 |

Both use learning rate 0.05, feature fraction 0.8, row fraction 0.8 with
frequency 1, LambdaRank truncation 30, unlimited depth, seed 0,
deterministic row-wise fitting, and four threads. Full TRAIN fits use fixed
tree counts without early stopping.

`query_rank` uses raw values plus within-query tied-midrank percentiles.
`query_normalized` uses within-query min-max values plus those percentiles.
Finite constants map to 0.5, missing values remain missing, and no labels
or other queries enter feature computation. Positive affine invariance of
the normalized representation holds in exact arithmetic; finite-precision
rounding can change a few ranks. Sorting uses blocks of at most 16 columns.

The average is exported as **one native LightGBM text model**, with
1000 trees and 408
internal features (raw, percentile and min-max blocks). It still accepts
136 raw features. Export remaps split indices and scales leaf outputs by
0.5. No constituent files are needed at grading time. Saved fold models
agree with explicit averaging to maximum absolute error 9.33e-15.
The feature-name metadata also supports legacy raw and query-rank models,
including arbitrary-width models emitted by the C1 recipe.

## TRAIN-only selection and generalization

The earlier 49-configuration grouped five-fold sweep remains in
`results/lambdamart_cv_v2.json`. V3 reports eight complete five-fold settings:
control and normalized features at 200/400/800 trees, plus fixed equal-weight
400+200 and 800+200 blends. Reported folds never control early stopping.
Two heavier regularization probes were stopped for runtime and excluded;
no partial-fold mean enters selection. The control reproduces the previous
clean CV score exactly. All reported fold scores are in
`results/lambdamart_cv.json`; the protocol and its documented bounded changes
are in [docs/EXPERIMENTS_V3.md](docs/EXPERIMENTS_V3.md).

Keep settings within 0.0025 clean nDCG of the highest complete mean, then
maximize `0.5 clean + 0.5 mean(three diagnostics)`. Require at least 0.001
improvement over v2 in this mixture; exact ties prefer fewer trees.
The choice is frozen before full fitting and final DEV scoring.

| TRAIN-CV condition | v2 control | Selected v3 |
|---|---:|---:|
| clean | 0.484133 | 0.485550 |
| affine | 0.446501 | 0.473141 |
| feature_dropout | 0.465942 | 0.472943 |
| noise | 0.452528 | 0.459265 |

Diagnostics apply query-local positive affine scale/shift changes, 10%
whole-column loss, and Gaussian noise with SD 5% of query-feature SD.
Their definitions and seeds are public in `scripts/robustness.py`. They are
**not the staff C2 construction**. The selected clean mean is
0.485550, sample SD 0.008929;
training mean is 0.719851, validation/training ratio
0.674514 (v2: 0.619071).
The gap remains substantial. These are adaptive selection estimates,
not an unbiased new held-out evaluation.

## Fusion and neural models

RRF, CombSUM and CombMNZ remain hand-written and independently tested. The
deployed rule is v2 weighted CombSUM after per-query min-max normalization:
BM25 whole document 0.25, BM25 title 0.25, LMIR Dirichlet 0.50, LMIR JM 0.
Constants contribute zero. TRAIN selected it among 41 settings using the
same 0.8 raw nDCG + 0.2 raw MAP proxy. Textbook fusion functions retain their
unweighted definitions.

Both neural scorers use the same 136 → 128 → 64 → 1 ReLU MLP and full-TRAIN
standardizer. Pointwise uses six MSE epochs at learning rate 0.001. Pairwise
uses six supervised MSE warm-up epochs at 0.001, then two weighted RankNet
epochs at 0.0001, with a fresh Adam optimizer. The extra weighted loss uses
absolute nDCG@10 swap changes; graded `ranknet_loss` remains plain stable
softplus. Sample up to 300 distinct, strictly ordered same-query pairs per
query and epoch. Batch size is 1024; seed is 0. Internal TRAIN fold 0 selects
epoch counts; fixed full-TRAIN refits produce the shipped artifacts. Both
are unchanged in v3 and load with the pristine staff MLP loader.

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
# Only evaluate DEV after the configuration is frozen:
python scripts/evaluate_models.py --features data/dev/vali.txt
python scripts/run_recipe.py
python scripts/package_submission.py --out dist/rank-fusion-submission-v3.zip
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

`prepare()` only loads the artifact and raises if missing/corrupt. Rankings
return unique supplied candidate IDs, with stable ties. Final DEV preparation
and maximum reranking call times are 0.377s
and 0.123s, within the 60s/2s limits.

## AI-use disclosure and code provenance

OpenAI Codex assisted with reading the specification, implementation, tests,
HPC execution, experiment interpretation, documentation and packaging.
The unchanged starter supplies the harness, MLP, standardizer, metrics and
grouped-fold helper. Fusion, MSE and RankNet follow the assignment and primer.
Percentiles, swap weighting, diagnostic perturbations, native tree averaging
and adaptive recipe logic are implemented directly; no external fusion or
neural ranking trainer is copied. LightGBM's permitted `lambdarank` objective
trains the trees. Consulted references include the official dataset page,
[LightGBM 4.6 parameter documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html),
and the [IIT Delhi HPC guide](https://supercomputing.iitd.ac.in/).
See [docs/SUBMISSION_NOTES.md](docs/SUBMISSION_NOTES.md) for the declarations.
