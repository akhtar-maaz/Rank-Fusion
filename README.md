# A3 — Rank Fusion and Learning to Rank

Hand-written fusion, pointwise MSE and pairwise RankNet, with offline-trained
LightGBM LambdaMART and an adaptive transfer recipe. The current submission
is v4. Report writing remains deferred.

## Results and data declaration

Every submitted MSLR model is fitted on **Fold1 TRAIN only**: 723,412 rows,
6,000 queries and 136 raw features. All model choices use TRAIN folds only.
DEV is evaluated after freezing the configuration, over 235,259 rows and
2,000 queries. No DEV fitting, tuning or early stopping is used. No held-out
split, other fold, undisclosed benchmark, external relevance label or
distillation target is accessed.

| Raw DEV metric | v3 | v4 |
|---|---:|---:|
| Fusion nDCG@10 | 0.307785 | 0.307785 |
| Fusion MAP@10 | 0.128917 | 0.128917 |
| LambdaMART nDCG@10 | 0.494193 | 0.496894 |
| Pointwise nDCG@10 | 0.431708 | 0.431708 |
| Pairwise nDCG@10 | 0.444853 | 0.444853 |

Both LTR values round to 0.495 under Track B's 0.005 grading bands, so the
raw DEV gain alone does not raise its percentile. C2 still needs staff scoring.

The neural artifacts, fusion configuration and C1 recipe retain their v3
hashes. User-reported v3 feedback is in `results/leaderboard_v3.json`:
7/7 conformance, 10/10 correctness, pairwise held-out 0.4333, C2 0.3625,
live score 0.4093/0.65 and seventh of 15 teams in that snapshot. V4's hidden
C2/live result requires staff evaluation; C1 is computed after the deadline.

All **83 tests** pass in the temporary-copy smoke harness. Checks include
actual DEV scoring through the submission interface, staff-compatible neural
loading under Python 3.12, native averaging equivalence, and an independent
full-TRAIN refit that is byte-identical. See `results/artifacts.json`,
`results/execution.json` and `results/lambdamart_reproducibility.json`.

## Final tree model

The frozen choice is `v3_plus_3_800`:

| Features | Trees | Weight | Truncation | Training augmentation |
|---|---:|---:|---:|---|
| query_rank | 800 | 0.25 | 30 | none |
| query_normalized | 200 | 0.25 | 30 | none |
| query_rank | 800 | 0.5 | 13 | dropout/noise copies |

Every member uses 127 leaves, minimum leaf size 100, L2 10, learning rate
0.05, feature fraction 0.8, row fraction 0.8 with frequency 1, unlimited depth,
seed 0, deterministic row-wise fitting and four threads. Full TRAIN fits use
fixed tree counts without early stopping.

`query_rank` contains raw values plus within-query tied-midrank percentiles.
`query_normalized` contains min-max values plus those percentiles. Finite
constants map to 0.5 and missing values remain missing. Each transformation
uses only that query's candidates. Percentile sorting uses blocks of at most
16 columns. The native artifact has 1800 trees and 408
internal features, accepts 136 raw inputs, and uses `query_all` metadata.
Score averages use remapped split indices and weighted leaf outputs in one
ordinary LightGBM text model; constituent files are unnecessary for grading.

Augmented fits retain every original training query and append a perturbed
copy of about half, with distinct group IDs. A fixed SHA256 rule chooses
the query and dropout versus noise; labels are copied without inspection.
When applied to full TRAIN, the rule adds 2,990 queries / 362,923 rows
(1,446 dropout queries and 1,544 noise queries). Augmentation runs after
splitting TRAIN and uses a different seed from validation diagnostics.
See `scripts/augmentation.py` and the
frozen [v4 protocol](docs/EXPERIMENTS_V4.md).

## TRAIN-only selection and generalization

V4 compares 36 complete five-fold settings: 12 fixed averages
of cached v3 scorers, 12 new single-model checkpoints, and 12 averages of
those checkpoints with v3. The new fits use truncation 13, two query-local
representations and optional augmentation, at 200/400/800 trees. There is no
early stopping on reported folds. All four candidates complete all five
query-grouped folds (seed 0). Earlier 49-setting and eight-setting sweeps
remain in `results/lambdamart_cv_v2.json` and `results/lambdamart_cv_v3.json`.

Retain settings within 0.0025 clean nDCG of the best complete mean; maximize
`0.5 clean + 0.5 mean(affine, feature_dropout, noise)`. Require at least 0.0005
improvement over v3's mixture. Exact ties favor fewer total trees. The v3
control reproduces its earlier TRAIN-CV result. Selection is frozen before
full fitting and final DEV scoring.

| TRAIN-CV condition | v3 | v4 |
|---|---:|---:|
| clean | 0.485550 | 0.486043 |
| affine | 0.473141 | 0.467205 |
| feature_dropout | 0.472943 | 0.475050 |
| noise | 0.459265 | 0.468511 |

The selected diagnostic mean is 0.470255, versus
0.468450 for v3; the mixture changes from
0.477000 to 0.478149.
Clean sample SD is 0.010684, training mean
0.743278 and validation/training ratio
0.653918. The generalization gap increases relative to v3. Noise and feature-loss
diagnostics improve, while affine robustness declines. The clean gain is
small relative to the fold-to-fold variation. These are adaptive selection
estimates, not an unbiased new held-out test.

Diagnostics apply positive affine changes, 10% query-column loss and Gaussian
noise with SD 5% of query-feature SD. Their public definitions and seeds are
in `scripts/robustness.py`; they are **not the staff C2 construction**.
The exported winner agrees with explicit held-out TRAIN score averaging
on every fold to maximum absolute error
2.31e-14.

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
are unchanged in v4 and load with the pristine staff MLP loader.

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
python scripts/package_submission.py --out dist/rank-fusion-submission-v4.zip
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
return unique supplied candidate IDs with stable ties. Final DEV preparation
and maximum reranking times are 0.723s and
0.219s, within the 60s/2s limits.

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
