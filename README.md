# A3 — Rank Fusion and Learning to Rank

Python implementation of hand-written RRF, CombSUM, CombMNZ, pointwise regression,
and RankNet, plus LightGBM LambdaMART training and a transfer recipe.

## Trained submission

All three artifacts in `submission/` are trained on the real MSLR-WEB10K
Fold1 TRAIN split: 723,412 rows, 6,000 queries, 136 features. All model and
fusion selection uses TRAIN only. DEV is reserved for final evaluation.
The report is deferred; frozen measurements are saved in [results/](results/README.md).

Final raw DEV scores, over 2,000 queries and 235,259 rows:

| Ranker | nDCG@10 |
|---|---:|
| Single BM25 feature | 0.288487 |
| CombSUM fusion | 0.305636 |
| Pointwise MLP | 0.431708 |
| Pairwise RankNet MLP | 0.421922 |
| LambdaMART | 0.482608 |

Fusion MAP@10 is 0.128488. These are raw local metrics, not class percentile
scores. Hidden transfer and robustness scores require staff evaluation.
The pairwise model clears the required BM25 floor. Both neural artifacts load
with the unchanged staff-compatible loader under Python 3.12.

All 53 tests pass locally and on an IIT Delhi HPC compute node. The final
four-CPU, 8 GB job completed in 7 minutes 4 seconds, with reported peak memory
about 1.6 GB. An independent LambdaMART fit produced an identical SHA-256.

## Setup and checks

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/smoke_test.sh
```

On macOS, install OpenMP with `brew install libomp`. The smoke script makes
LightGBM share PyTorch's bundled OpenMP runtime, and tests a temporary copy:
it cannot overwrite the actual submitted models.

## Data

Use only MSLR-WEB10K Fold1 TRAIN and DEV. The official archive is linked from
[Microsoft Research](https://www.microsoft.com/en-us/research/project/mslr/).
Its published MD5 is `97c5d4e7c171e475c91d7031e4fd8e79`.

```bash
python scripts/prepare_mslr.py --archive /path/to/MSLR-WEB10K.zip --out data
# Alternatively, use an already prepared Fold1 directory:
python scripts/prepare_mslr.py --fold-dir /path/to/Fold1 --out data
```

Archive preparation opens only `Fold1/train.txt` and `Fold1/vali.txt`;
it never opens the held-out file or another fold. Real data, downloads,
environments, and intermediate runs are ignored by Git.

## Reproducible training

Use the committed CV selection to reproduce the submitted tree model:

```bash
python scripts/train_model.py --train data/train --config results/lambdamart_cv.json
python scripts/train_neural.py --train data/train --kind both --hidden 128 64 \
    --epochs 30 --patience 5 --pairs-per-query 300 --history-out results/neural_history.json
python scripts/select_fusion.py --train data/train/train.txt --out results/fusion_selection.json
# Run only after configurations and models are frozen:
python scripts/evaluate_models.py --features data/dev/vali.txt --out results/dev_metrics.json
python scripts/run_recipe.py
```

To reproduce the CV search itself:

```bash
python scripts/cv_lambdamart.py --train data/train --out runs/cv_initial.json
python scripts/cv_lambdamart.py --train data/train --config runs/cv_initial.json \
    --trees 800 1200 --out runs/cv_extension.json
python scripts/collect_cv.py runs/cv_initial.json runs/cv_extension.json \
    --out results/lambdamart_cv.json
```

The initial grid is leaves {15, 31}, minimum leaf size {50, 150}, L2 {1, 10},
and tree counts {50, 100, 200, 400}. Its winner was at the upper tree boundary,
so that configuration alone was extended to 800 and 1200 trees on the same
folds. No reported CV fold controls early stopping. Selection maximizes mean
validation nDCG@10, with ties preferring fewer trees and leaves.

## Final configurations

**Fusion.** Hand-written RRF, CombSUM, and CombMNZ are all implemented.
CombSUM sums the four channels after per-query min-max normalization; constant
channels contribute zero. It won the predeclared TRAIN comparison against
CombMNZ and RRF with c in {0, 10, 30, 60, 100}, using
0.8 × raw nDCG@10 + 0.2 × raw MAP@10 as the selection proxy.
The selected method is stored in `submission/fusion_config.json`.

**LambdaMART.** LightGBM `lambdarank`, 1200 trees, 31 leaves, learning rate
0.05, minimum leaf size 50, L2 1, feature fraction 0.8, bagging fraction 0.8
with frequency 1, seed 0, deterministic row-wise training, four threads.
The final fit uses all TRAIN and a fixed tree count, without early stopping.
For these selected regularization settings, the five-fold results are:

| Trees | Training nDCG@10 | Validation mean | Validation sample SD | Validation / training |
|---:|---:|---:|---:|---:|
| 50 | 0.469148 | 0.450289 | 0.007897 | 0.9598 |
| 100 | 0.491625 | 0.462202 | 0.007034 | 0.9402 |
| 200 | 0.518155 | 0.470066 | 0.007449 | 0.9072 |
| 400 | 0.553717 | 0.473145 | 0.007315 | 0.8545 |
| 800 | 0.605034 | 0.473991 | 0.007695 | 0.7834 |
| 1200 | 0.643263 | 0.475014 | 0.007563 | 0.7384 |

All 34 configurations and their fold scores are in
`results/lambdamart_cv.json`. Training accuracy grows much faster than
validation accuracy; the widening gap is retained explicitly. We selected the
best validation mean in this bounded search while retaining limited leaf
capacity, shrinkage, and row/feature sampling. These are model-selection
estimates rather than private held-out scores.

**Neural models.** Both use the same 136 → 128 → 64 → 1 ReLU MLP,
TRAIN-fitted feature standardization, Adam at 0.001, minibatches of 1024, and
seed 0. The losses are plain MSE and unweighted stable RankNet softplus.
Pair sampling is within query, without replacement, capped at 300 per query
per epoch, without enumerating every possible pair. Internal TRAIN fold 0
selects stopping epochs with a maximum of 30 and patience 5: six epochs for
pointwise and one for pairwise. Each model is then refit on all TRAIN for its
fixed selected epoch count. DEV never controls stopping.

**Features and transfer recipe.** The tree model uses the 136 provided
features directly; no extra features or external training data are used.
Neural models standardize those same features. The dimension-agnostic
transfer recipe uses 15 leaves below 200 training queries, otherwise 31;
minimum leaf size is `max(20, min(200, rows // 2000))`. It uses L2 10,
learning rate 0.05, feature fraction 0.9 below 100 features (otherwise 0.8),
bagging fraction 0.9, and at most 500 trees with patience 40 on the benchmark's
supplied validation file. It preserves NaNs for LightGBM's missing-value
handling. Its design is informed by TRAIN CV; synthetic transfer sets verify
the interface, not performance on the undisclosed benchmarks.

Training uses a two-pass float32 loader to keep memory bounded. Preparation
loads artifacts and raises for missing or corrupt models. Inference preserves
candidate order on ties and returns only unique supplied candidates.
On the final compute-node DEV evaluation, preparation took 0.287 seconds;
maximum fusion and reranking calls took 0.0012 and 0.0418 seconds, below the
60-second preparation and 2-second per-call caps.

For HPC submission and offline dependency installation, see [docs/HPC.md](docs/HPC.md).
The grading harness is unchanged; the ranking interface preserves its exact
signatures and loads artifacts without training.

After committing changes, build the upload file with:

```bash
python scripts/package_submission.py
```

This writes `dist/rank-fusion-submission.zip` from tracked repository files,
retaining the original layout and toy datasets while excluding local real
data, downloads, environments, and intermediate runs.

## Training-data declaration

| Artifact | Trained on | Selection / early stopping |
|---|---|---|
| `model.txt` | Full Fold1 TRAIN only | TRAIN-only grouped five-fold CV; final fit has no early stopping |
| `pointwise_model.pt` | Full Fold1 TRAIN only | Internal TRAIN fold 0; six-epoch full refit |
| `pairwise_model.pt` | Full Fold1 TRAIN only | Internal TRAIN fold 0; one-epoch full refit |
| `recipe.py` | Supplied benchmark TRAIN when invoked | Fixed adaptive design; supplied validation for early stopping |

DEV is evaluated after selection and is never used to fit, tune, or early-stop
any submitted MSLR model. No held-out labels or other real datasets are used.

## AI-use disclosure

OpenAI Codex assisted with reading the specification, implementing and reviewing
the algorithms, writing tests and training scripts, configuring and running HPC
experiments, interpreting measurements, and preparing repository documentation
and the submission archive.

## Code provenance

The original starter supplies the unchanged harness, MLP, and standardizer.
Fusion and neural-loss formulas follow the assignment PDF and starter primer;
no external fusion or ranking-training implementation was copied. The
[Microsoft dataset page](https://www.microsoft.com/en-us/research/project/mslr/)
and [IIT Delhi HPC guide](https://supercomputing.iitd.ac.in/) were consulted
for dataset access and scheduled execution.
