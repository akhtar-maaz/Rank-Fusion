# A3 — Rank Fusion and Learning to Rank

Hand-written fusion, pointwise MSE, and pairwise RankNet, with offline-trained
LightGBM LambdaMART and a dimension-agnostic transfer recipe. The current
artifacts are the second-round models; the report is deferred until submission
feedback. Frozen measurements are retained in [results/](results/README.md).

## Current results

All submitted MSLR models are fitted on Fold1 TRAIN only: 723,412 rows,
6,000 queries, 136 input features. All selection uses TRAIN folds. DEV is
evaluated only after configurations are frozen, over 2,000 queries and
235,259 rows.

| Raw DEV metric | First submission | Current submission |
|---|---:|---:|
| Fusion nDCG@10 | 0.305636 | 0.307785 |
| Fusion MAP@10 | 0.128488 | 0.128917 |
| LambdaMART nDCG@10 | 0.482608 | 0.491177 |
| Pointwise nDCG@10 | 0.431708 | 0.431708 |
| Pairwise nDCG@10 | 0.421922 | 0.444853 |

The BM25 floor is 0.288487; pairwise clears it. These are raw
DEV metrics, not class percentiles or held-out Track P/C2 scores. The first
upload's user-reported website feedback is in `results/leaderboard_v1.json`.
Current C2, held-out P, live score, and post-deadline C1 require staff evaluation.

All 64 tests pass locally and in Ubuntu/Python 3.12 CI. Final validation also
runs on four IIT Delhi HPC CPUs with 8 GB. Both neural artifacts load with the
unchanged staff-compatible loader under Python 3.12. Artifact hashes are in
`results/artifacts.json`; the independent tree refit is checked byte-for-byte
in `results/lambdamart_reproducibility.json`.

## Methods and features

**Fusion.** RRF, CombSUM, and CombMNZ are hand-written and independently
testable. The deployed rule is weighted CombSUM after per-query min-max
normalization: BM25 whole document 0.25, BM25 title 0.25, LMIR Dirichlet 0.50,
and LMIR JM 0. Constant channels contribute zero. These weights won a TRAIN-only
comparison of the original seven rules and 34 additional quarter-step simplex
settings, using 0.8 × raw nDCG@10 + 0.2 × raw MAP@10 as a selection proxy.
The textbook fusion functions keep their original unweighted definitions.

**Tree features.** The MSLR model uses the original 136 values plus 136
within-query feature percentiles. Each percentile is a tied midrank divided
by the number of finite documents minus one; constant finite columns map to
0.5, and NaNs remain missing. This uses candidate features only, without labels.
Queries are transformed independently, including inside CV. Feature names in
the LightGBM text artifact identify the transform, so `rerank_with_ltr` maps
the 136 input values to the expected 272 model features. Legacy raw-feature
models remain supported. Feature sorting uses blocks of at most 16 columns
to bound temporary memory for large queries.

**LambdaMART.** The selected configuration has 800 trees,
127 leaves, no explicit depth limit, learning rate 0.05, minimum leaf size 100,
L2 10, feature fraction 0.8, bagging fraction 0.8 with frequency 1, LambdaRank
truncation 30, seed 0, deterministic row-wise training, and four threads.
The final fit uses all TRAIN and a fixed tree count, with no early stopping.

**Neural models.** Both use the same 136 → 128 → 64 → 1 ReLU MLP,
TRAIN-fitted standardization, batch size 1024, Adam, and seed 0. Pointwise uses
MSE at learning rate 0.001 for six selected epochs. Pairwise first receives six
supervised MSE warm-up epochs at 0.001, then two selected fine-tuning epochs at
0.0001 with a separate RankNet loss weighted by the absolute nDCG@10 change
from swapping a pair's current ranks. The graded `ranknet_loss` remains plain,
stable, and unweighted. Pairs are sampled within query without replacement,
at most 300 per query per epoch, without enumerating every possible pair.
All targets are relevance labels. Internal TRAIN fold 0 selects epochs;
each stage is repeated on full TRAIN for the fixed chosen counts.

**Transfer recipe.** The same query-relative transform handles arbitrary
feature widths without knowing their meanings. Regularized LambdaMART uses
15 leaves below 200 training queries, otherwise 31; minimum leaf size is
`max(20, min(200, rows // 2000))`. It keeps learning rate 0.05, L2 10, feature
fraction 0.9 below 100 original features (otherwise 0.8), row fraction 0.9,
truncation 15, four threads, and at most 500 trees with patience 40 on the
supplied validation file. NaNs use LightGBM's missing-value handling.
Synthetic 40-, 90-, and 250-feature checks cover portability and label ranges;
their scores do not estimate the undisclosed benchmarks.

## Selection and generalization

The initial grouped five-fold TRAIN sweep evaluated 34 configurations. The
second round adds 15 complete configurations, including larger trees,
query-relative features, and depth/truncation controls. All 49 rows and
their fold scores are retained in `results/lambdamart_cv.json`. Incomplete
compute-limited experiments are excluded. No reported CV fold controls early
stopping. [Experiment protocol](docs/EXPERIMENTS_V2.md) records the bounded
stages and all five pairwise variants.

For the selected tree settings:

| Trees | Training nDCG@10 | Validation mean | Validation sample SD | Validation / training |
|---:|---:|---:|---:|---:|
| 200 | 0.618379 | 0.482170 | 0.009994 | 0.7797 |
| 400 | 0.694359 | 0.483968 | 0.009643 | 0.6970 |
| 800 | 0.782033 | 0.484133 | 0.008215 | 0.6191 |

Best validation mean rises from 0.475014 to 0.484133.
Larger raw-feature trees did not improve it; query-relative features did.
Selection maximizes mean validation nDCG, with ties preferring fewer trees
and leaves. The training/validation gap remains substantial: the chosen
validation/training ratio is 0.619071. We favor the higher
validation mean within this bounded search and retain the gap explicitly.
These are selection estimates, not private robustness measurements. Neural
selection nDCG rises from 0.426320 to 0.444760 on the same internal TRAIN fold.

## Setup and checks

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/smoke_test.sh
```

On Linux, install PyTorch's CPU wheel first if CUDA is unnecessary:
`python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu`.
On macOS, install OpenMP with `brew install libomp`. The smoke script shares
PyTorch's bundled OpenMP runtime and tests a temporary copy, preserving the
submitted models. For other macOS scripts importing both Torch and LightGBM,
set `DYLD_LIBRARY_PATH` to the `torch/lib` directory in the active environment.

## Data and reproduction

Use only MSLR-WEB10K Fold1 TRAIN and DEV. The official archive is linked from
[Microsoft Research](https://www.microsoft.com/en-us/research/project/mslr/)
and has MD5 `97c5d4e7c171e475c91d7031e4fd8e79`.

```bash
python scripts/prepare_mslr.py --archive /path/to/MSLR-WEB10K.zip --out data
python scripts/train_model.py --train data/train --config results/lambdamart_cv.json
python scripts/train_neural.py --train data/train --kind pointwise --hidden 128 64 \
    --epochs 30 --patience 5 --lr 0.001 --history-out runs/reproduce_pointwise.json
python scripts/train_neural.py --train data/train --kind pairwise --hidden 128 64 \
    --epochs 15 --patience 4 --lr 0.0001 --warmup-epochs 6 --pair-weighting ndcg \
    --pairs-per-query 300 --history-out runs/reproduce_pairwise.json
python scripts/select_fusion.py --train data/train/train.txt --weighted \
    --out runs/reproduce_fusion.json
# Evaluate only after model configurations are frozen:
python scripts/evaluate_models.py --features data/dev/vali.txt --out results/dev_metrics.json
python scripts/run_recipe.py
python scripts/package_submission.py --out dist/rank-fusion-submission-v2.zip
```

See [docs/HPC.md](docs/HPC.md) for the full CV and scheduled refit sequence.
Archive preparation opens only Fold1 TRAIN and validation entries. Held-out
data and other folds are never opened. Real data, downloaded packages,
environments, and intermediate runs are excluded from Git and the submission
ZIP. Original starter files and toy datasets are retained. The harness is
unchanged; preparation only loads artifacts and raises for missing/corrupt
models. Rankings preserve candidate order on ties and return unique supplied
candidates only. Final measured preparation and maximum reranking times are
0.283 and 0.095 seconds, within the
60-second / 2-second caps.

## Training-data declaration

| Artifact | Fitted on | Selection / stopping |
|---|---|---|
| `model.txt` | Full Fold1 TRAIN only | Grouped five-fold TRAIN CV; fixed final tree count |
| `pointwise_model.pt` | Full Fold1 TRAIN only | Internal TRAIN fold 0; six-epoch full refit |
| `pairwise_model.pt` | Full Fold1 TRAIN only | Internal TRAIN fold 0; six MSE warm-up plus two weighted-pairwise full-refit epochs |
| `recipe.py` | Supplied benchmark TRAIN when invoked | Fixed adaptive design and supplied validation for early stopping |

DEV is never used to fit, tune, or early-stop a submitted MSLR model. No
external relevance labels, held-out labels, or distillation targets are used.

## AI-use disclosure and code provenance

OpenAI Codex assisted with reading the specification, implementing and
reviewing the algorithms, writing tests and training scripts, configuring
and running HPC experiments, interpreting measurements, and preparing
repository documentation and the upload archive.

The original starter provides the unchanged harness, MLP, standardizer,
metrics, and grouped-fold helper. Fusion, MSE, and RankNet formulas follow
the assignment and starter primer; query percentiles and metric swap weights
are implemented directly. No external fusion or neural ranking trainer was
copied. LightGBM's permitted `lambdarank` objective implements the tree ranker.
The Microsoft dataset page and [IIT Delhi HPC guide](https://supercomputing.iitd.ac.in/)
were consulted for data access and scheduled execution. Full declarations
are also in [docs/SUBMISSION_NOTES.md](docs/SUBMISSION_NOTES.md).
