# A3 — Rank Fusion and Learning to Rank

Hand-written fusion, pointwise MSE and pairwise RankNet, with an offline-trained
LightGBM LambdaMART entry and adaptive transfer recipe. The current submission
is v5. Formal report writing remains deferred.

## Results and data declaration

Every submitted MSLR model uses **Fold1 TRAIN only**: 723,412 original rows,
6,000 original queries and 136 raw features. Selection, feature choices and
stopping choices use TRAIN folds only. DEV is evaluated after freezing the
configuration, over 235,259 rows and 2,000 queries. No DEV fitting, tuning or
early stopping is used. No official test split, other fold, undisclosed benchmark,
external relevance label or distillation target is accessed.

| Raw DEV metric | v4 | v5 |
|---|---:|---:|
| Fusion nDCG@10 | 0.307785 | 0.307785 |
| Fusion MAP@10 | 0.128917 | 0.128917 |
| LambdaMART nDCG@10 | 0.496894 | 0.497649 |
| Pointwise nDCG@10 | 0.431708 | 0.431708 |
| Pairwise nDCG@10 | 0.444853 | 0.444853 |

Track B rounds raw scores to 0.005 bands before class percentile ranking;
a raw gain alone does not establish a leaderboard-score gain. The neural
artifacts, fusion configuration and C1 recipe retain their v4 hashes.
User-reported v4 feedback is in `results/leaderboard_v4.json`: 7/7 conformance,
10/10 correctness, pairwise held-out 0.4333, C2 0.3698, live score 0.4290/0.65,
and sixth of 21 teams in that snapshot. V5 needs staff C2/live evaluation.

All **91 tests** pass locally and on the college server. Checks include
actual DEV scoring through the submission interface, staff-compatible neural
loading under Python 3.12, native export equivalence and an independent
full-TRAIN refit that is byte-identical. See `results/artifacts.json`,
`results/execution.json` and `results/lambdamart_reproducibility.json`.

## Final tree model

The frozen choice is `v4_plus_1_800_weight0.5`:

| Member | Features | Trees | Weight | Augmentation |
|---:|---|---:|---:|---|
| 1 | query_rank | 800 | 0.125 | none |
| 2 | query_normalized | 200 | 0.125 | none |
| 3 | query_rank | 800 | 0.25 | dropout/noise |
| 4 | query_all | 800 | 0.5 | affine/dropout/noise |

| Member | Leaves | Minimum leaf size | L2 | Feature fraction | Truncation |
|---:|---:|---:|---:|---:|---:|
| 1 | 127 | 100 | 10 | 0.8 | 30 |
| 2 | 127 | 100 | 10 | 0.8 | 30 |
| 3 | 127 | 100 | 10 | 0.8 | 13 |
| 4 | 127 | 100 | 10 | 0.8 | 30 |

All members use learning rate 0.05, row fraction 0.8 with frequency 1,
unlimited depth, seed 0, deterministic row-wise fitting and four threads.
Full-TRAIN fits use fixed tree counts without early stopping. The saved
native model has 2600 trees and 408 internal
features, accepts 136 raw inputs and uses `query_all` metadata.
Weighted averages remap split indices and scale leaf outputs in one ordinary
LightGBM text model. Constituent files are unnecessary at grading time.

`query_rank` contains raw values plus within-query tied-midrank percentiles.
`query_normalized` contains query min-max values plus percentiles.
`query_all` contains raw, percentile and min-max blocks. Finite constants
map to 0.5 and missing values remain missing. Each transformation sees only
that query's candidates; percentile sorting uses blocks of at most 16 columns.

## Augmentation and TRAIN-only selection

Augmented fits retain all original training queries and append one perturbed
copy of about half, with separate group IDs. Labels are copied without
inspection. V4's dropout/noise augmentation and seed 20261019 are unchanged.
The new mixed scheme uses seed 20261020 and a fixed SHA256 rule to choose
copying and affine/dropout/noise. Training seeds differ from diagnostic seed
1729. Augmentation runs after splitting, so a validation query never enters
fitting. When applied to full TRAIN, the deterministic rules add:

- `half_query_dropout_noise`: 2,990 copied queries / 362,923 copied rows.
- `half_query_mixed`: 2,984 copied queries / 356,838 copied rows.

The frozen [v5 protocol](docs/EXPERIMENTS_V5.md) declares 54 settings: six
cached averages, 12 new single-model checkpoints, and 36 new/v4 averages.
The result record contains 54 complete five-fold rows.
All four new candidates complete every fold. All splits group by query
(seed 0), and reported folds never control early stopping.
Training-fold metrics use original clean queries, excluding
the appended copies. Earlier sweeps remain in the versioned result records.

Keep settings within 0.0025 clean nDCG of the best complete mean. No individual
diagnostic may regress more than 0.0025 relative to v4. Among eligible settings,
maximize `0.5 clean + 0.5 mean(affine, feature_dropout, noise)`; require a gain
of at least 0.0005 over v4. Exact ties favor fewer total trees. The v4 control
reproduces its previous TRAIN-CV values. Selection is frozen before full
fitting and one final DEV evaluation.

| TRAIN-CV condition | v4 | v5 |
|---|---:|---:|
| clean | 0.486043 | 0.487398 |
| affine | 0.467205 | 0.474040 |
| feature_dropout | 0.475050 | 0.476076 |
| noise | 0.468511 | 0.472108 |

The diagnostic mean changes from 0.470255 to
0.474075; the mixture from 0.478149
to 0.480737. Selected training mean is
0.756822, clean sample SD 0.009563,
and validation/training ratio 0.644007.
The remaining gap and reused-fold adaptive selection limit confidence.
These estimates are not an unbiased new held-out test.

Diagnostics apply positive affine changes, 10% query-column loss and Gaussian
noise with SD 5% of query-feature SD. Definitions are in `scripts/robustness.py`;
they are **not the staff C2 construction** and do not establish a hidden-score
gain. The exported winner matches explicit held-out TRAIN averaging on every
fold, with maximum score error
2.04e-14 and identical
nDCG values.

After selection was frozen, perturbation seed 271828 gave:

| Post-selection seed 271828 | v4 | v5 |
|---|---:|---:|
| affine | 0.468370 | 0.474720 |
| feature_dropout | 0.475824 | 0.477312 |
| noise | 0.468689 | 0.471619 |
| Mean | 0.470961 | 0.474550 |

Every diagnostic improves under this second seed. This check did not choose
or change the artifact. It measures sensitivity to perturbation randomness
on reused TRAIN validation queries, not a new held-out label test.

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
are unchanged in v5 and load with the pristine staff MLP loader.

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
python scripts/package_submission.py --out dist/rank-fusion-submission-v5.zip
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
and maximum reranking times are 3.409s and
0.359s, within the 60s/2s limits.

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
