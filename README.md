# A3 — Rank Fusion and Learning to Rank

Hand-written fusion, pointwise MSE and pairwise RankNet, with an offline-trained
LightGBM LambdaMART entry and adaptive transfer recipe. Current submission: v7.1.
Formal report writing remains deferred.

## Results and data declaration

Every shipped MSLR model is fitted on **Fold1 TRAIN only**: 723,412 original rows,
6,000 original queries and 136 raw features. Model and stopping choices use TRAIN
folds only. Released DEV (235,259 rows, 2,000 queries) is evaluated once after
both configurations are frozen. It is never used for MSLR fitting, tuning or
early stopping. No official test split, other MSLR fold, external relevance
label, distillation target or undisclosed benchmark is used in MSLR training
or selection. No attempt is made to identify the undisclosed C1 datasets.
The separate C1 recipe uses only the TRAIN and validation supplied by staff;
its supplied validation selects capacity and stopping. Website feedback is
user-provided aggregate feedback only.

| Raw DEV metric | v6 | v7 |
|---|---|---|
| Fusion nDCG@10 | 0.307785 | 0.307785 |
| Fusion MAP@10 | 0.128917 | 0.128917 |
| LTR nDCG@10 | 0.497932 | 0.498374 |
| Pointwise nDCG@10 | 0.431708 | 0.431708 |
| Pairwise nDCG@10 | 0.447643 | 0.449440 |

Track B rounds raw scores to 0.005 bands before class percentile ranking.
The latest user-reported evaluation is v7.1 submission 243: seventh of 29
scored teams, held-out pairwise 0.4383, C2 0.3790, live score 0.4493/0.65,
7/7 conformance and **10/10 correctness** (`results/leaderboard_v7_1.json`).
One additional team is queued. Compared with v6,
held-out pairwise increased by 0.0016 and C2 by 0.0043. Class membership and
competitors also changed, so the live-score difference does not isolate the
model's improvement. The v7.1 staff recheck restored the seeded-training check.

The v7 training run passed **112 tests** locally and on the college server.
The v7.1 source passes **120 tests** locally. Final artifacts
match the HPC hashes and load through the pristine staff neural loader under
Python 3.12. Promoted artifacts have byte-identical independent fresh fits;
each retained artifact preserves the exact v6 hash. Tree and neural promotion
decisions are independent. See `results/execution.json` and reproducibility records.

## Seeded training correction in v7.1

Submission 220 reported different weights from two same-seed training runs;
submission 243 confirms all ten correctness checks pass after the correction.
Controlled overlapping fits reproduced a shared PyTorch RNG interference
failure; the staff worker's exact execution pattern was not provided.
Each fit now initializes its linear weights and biases with a private CPU
generator, preserving the original `nn.Linear` initialization order, and
requests deterministic algorithms with four CPU threads.

Four interference regressions fail the old source and pass the correction.
Eighteen synthetic old/new fits preserve weights, histories and standardizers
exactly. All shipped model artifacts, fusion and C1 recipe remain byte-identical
to v7 commit `50fea4f`; their recorded DEV and held-out scores remain the
v7 measurements. See `results/reproducibility_fix_v7_1.json`. No new model
selection or real-data fitting was performed for this correction.

## Final tree model

The frozen rule promoted a new tree configuration; current choice is `v6_plus_2_400_weight0.25`.

| Member | Features | Trees | Weight | Augmentation (seed) |
|---|---|---|---|---|
| 1 | query_rank | 800 | 0.0703125 | none |
| 2 | query_normalized | 200 | 0.0703125 | none |
| 3 | query_rank | 800 | 0.140625 | half_query_dropout_noise (20261019) |
| 4 | query_all | 800 | 0.28125 | half_query_mixed (20261020) |
| 5 | query_all | 800 | 0.1875 | full_query_mixed (20261021) |
| 6 | query_all | 400 | 0.25 | full_query_mixed (20261021) |

| Member | Leaves | Minimum leaf | L2 | Feature fraction | Seed | Truncation | Additional regularization |
|---|---|---|---|---|---|---|---|
| 1 | 127 | 100 | 10 | 0.8 | 0 | 30 | none |
| 2 | 127 | 100 | 10 | 0.8 | 0 | 30 | none |
| 3 | 127 | 100 | 10 | 0.8 | 0 | 13 | none |
| 4 | 127 | 100 | 10 | 0.8 | 0 | 30 | none |
| 5 | 127 | 100 | 10 | 0.8 | 0 | 30 | none |
| 6 | 127 | 100 | 10 | 0.8 | 0 | 30 | extra_trees; extra_seed=0 |

All members use learning rate 0.05, row fraction 0.8 with frequency 1,
unlimited depth, deterministic row-wise fitting and four threads. Counts are
fixed, without early stopping. One ordinary native LightGBM model contains
3800 trees, 408 internal features and 136 raw inputs.
The exporter remaps feature indices and weights tree outputs; grading needs
no constituent artifacts. Native held-out TRAIN predictions match explicit
averaging on every fold, maximum error 2.4e-14,
with identical nDCG.

`query_rank` uses raw values and tied-midrank percentiles; `query_normalized`
uses query min-max values and percentiles; `query_all` uses all three blocks.
Constants map to 0.5 and missing entries stay missing. Transformations see
only the current query's candidates, with sorting temporaries bounded to 16 columns.

## Tree augmentation and TRAIN selection

The frozen [v7 protocol](docs/EXPERIMENTS_V7.md) compares 49 planned settings:
the exact v6 control, 12 new checkpoints and 36 fixed v6/new averages with
new-model weights 0.25/0.5/0.75. There are 49 complete
reported settings; runtime exclusions: [].
Every reported setting has all five query-grouped TRAIN folds, split seed 0.
Original clean training queries provide the reported training metrics.

Four new fits test full mixed augmentation with truncation 13, a split-gain
prior favoring rank/min-max over raw columns, randomized tree thresholds,
and broader augmentation. Full mixed seed 20261021 appends one copy per fit
query chosen as affine/dropout/noise. Broader seed 20261022 uses affine scales
0.25–4 and shifts ±0.5 feature range, 20% column dropout, or Gaussian noise
SD 10% of query-feature SD. Historical half-query augmentation schemes remain
unchanged. Originals remain, copied labels are not inspected, and augmented
copies have separate group IDs. All augmentation follows the query split.
Actual full-TRAIN counts are in `results/execution.json`.

Selection diagnostics remain positive affine scales 0.5–2/shifts ±0.25 range,
10% query-column dropout and Gaussian noise SD 5%, with seed 1729. These
are public generic diagnostics, **not the staff C2 construction**. Keep
settings within 0.0025 clean nDCG of the best complete mean and within 0.0025
of v6 on each diagnostic. Maximize half clean plus half mean diagnostic;
ties prefer fewer trees. That winner must improve mixture by 0.0005,
diagnostic mean by 0.001, and paired fold mixture in at least three folds.
If any gate fails, retain v6 without searching a runner-up.

The highest eligible candidate was `v6_plus_2_400_weight0.25`: mixture gain +0.000805618, diagnostic gain +0.001394126, and positive paired mixture gains in 5/5 folds. Gate outcomes (mixture/diagnostic/paired) are True/True/True.

| TRAIN-CV condition | v6 | v7 |
|---|---|---|
| clean | 0.487343 | 0.487560 |
| affine | 0.476724 | 0.479000 |
| feature_dropout | 0.476436 | 0.477154 |
| noise | 0.473732 | 0.474921 |

Diagnostic mean: 0.475631 → 0.477025;
mixture: 0.481487 → 0.482292.
Selected clean sample SD is 0.009312, training mean
0.732114, and validation/training ratio 0.665962.
Repeated adaptive use of these TRAIN folds and the generalization gap limit
confidence; these are not unbiased fresh held-out estimates.

| Post-selection seed 391729 | v6 | v7 |
|---|---|---|
| affine | 0.475103 | 0.476787 |
| feature_dropout | 0.476955 | 0.477275 |
| noise | 0.473841 | 0.475193 |
| Mean | 0.475300 | 0.476418 |

This descriptive check did not choose or change the artifact. It reuses TRAIN validation queries and is not a new held-out label test.

## Fusion and neural models

Fusion retains query min-max CombSUM: BM25 whole-document 0.25, BM25 title
0.25, LMIR.DIR 0.5, LMIR.JM 0. Pointwise retains 136→128→64→1 ReLU,
six MSE epochs at 0.001, originally selected on internal TRAIN fold 0.
Fusion, pointwise and the C1 recipe preserve v6 hashes.

The independent neural rule promoted the selected pairwise average. Frozen choice: `pairwise_seeds_0_41`;
shipped plain-MLP architecture: **136→256→128→1 ReLU**.

| Member seed | Hidden sizes | Weight | MSE epochs | RankNet epochs | RankNet LR | Pair cap | Batch |
|---|---|---|---|---|---|---|---|
| 0 | 128→64 | 0.5 | 10 | 2 | 0.0001 | 300 | 1024 |
| 41 | 128→64 | 0.5 | 10 | 2 | 0.0001 | 300 | 1024 |

Each member trains on relevance labels only, with MSE warm-up at 0.001,
then a fresh Adam optimizer for nDCG-weighted RankNet. Same-query strictly
ordered pairs are distinct within the cap per query/epoch. Weights are
absolute nDCG@10 swap changes; graded `ranknet_loss` remains plain stable
softplus. No model scores become training targets.

Three fixed candidates are evaluated on the same five TRAIN folds: v6's
seed 0, equal seeds 0/17, and equal seeds 0/41. Each fold fits three distinct
seeds once and reuses seed 0. Standardization fits only training rows and
is identical across same-fold seeds. V6's control reproduces training and
validation scores on all five folds within 1e-10. The better average must
gain at least 0.001 mean nDCG and improve at least three folds; otherwise
the exact v6 artifact is preserved. No epoch is selected using validation.

| Pairwise TRAIN-CV candidate | Mean | Sample SD | Train mean |
|---|---|---|---|
| v6_pairwise_control | 0.435046 | 0.007693 | 0.455417 |
| pairwise_seeds_0_17 | 0.437120 | 0.008205 | 0.457867 |
| pairwise_seeds_0_41 | 0.437999 | 0.008828 | 0.458611 |

Selected mean: 0.437999; control:
0.435046. Proposed average mean gain:
+0.002953, improved folds 5/5.
The two selected 128/64 members compile by concatenated first layers, block-diagonal second layers and weighted output rows into the shipped ordinary 256/128 MLP. It fits staff's width/depth bounds and needs no custom inference loader.
The full-TRAIN export differs from the explicit arithmetic score average by at most 1.43e-06; full-TRAIN exported/explicit nDCG are 0.461587306/0.461587306. Selection used the exported artifact's score, so float32 compilation effects are included.

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
python scripts/cv_pairwise_ensemble_v7.py --mode fit --train data/train/train.txt --specs configs/v7_neural_candidates.json --selection results/neural_selection_v7.json --baseline submission/pairwise_model.pt --fresh --cache-dir runs/reproduction_v7/neural_cache --out runs/reproduction_v7/pairwise_model.pt --history-out runs/reproduction_v7/pairwise_history.json
# The neural command independently refits a promoted choice or copies a retained artifact.
# Only evaluate DEV after both configurations are frozen:
python scripts/evaluate_models.py --features data/dev/vali.txt
python scripts/run_recipe.py
python scripts/package_submission.py --out dist/rank-fusion-submission-v7.1.zip
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
maximum reranking times are 3.335s and
0.508s, within the 60s/2s limits.

## AI-use disclosure and code provenance

OpenAI Codex assisted with reading the specification, implementation, tests,
HPC execution, experiment interpretation, documentation and packaging.
The unchanged starter supplies the harness, MLP, standardizer, metrics and
grouped-fold helper. Fusion, MSE and RankNet follow the assignment and primer.
Percentiles, swap weighting, diagnostic perturbations, training augmentation,
native tree averaging, bounded plain-MLP averaging and adaptive recipe logic are implemented directly; no external fusion or
neural ranking trainer is copied. LightGBM's permitted `lambdarank` objective
trains the trees. Consulted references include the official dataset page,
[LightGBM 4.6 parameter documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html),
and the [IIT Delhi HPC guide](https://supercomputing.iitd.ac.in/).
See [docs/SUBMISSION_NOTES.md](docs/SUBMISSION_NOTES.md) for the declarations.
