# Submission notes

## Training-data declaration

| Artifact | Trained on | Tuned on | Early stopping |
|---|---|---|---|
| LambdaMART `model.txt` | Full MSLR-WEB10K Fold1 TRAIN | Query-grouped five-fold CV within TRAIN, seed 0 | None; fixed 1200-tree final fit |
| `pointwise_model.pt` | Full Fold1 TRAIN | Internal TRAIN fold 0 | Select six epochs internally, then fixed six-epoch full TRAIN refit |
| `pairwise_model.pt` | Full Fold1 TRAIN | Internal TRAIN fold 0 | Select one epoch internally, then fixed one-epoch full TRAIN refit |
| `recipe.py` when invoked | The supplied benchmark TRAIN file | Fixed adaptive design informed by MSLR TRAIN CV; synthetic transfer checks | Supplied benchmark validation, patience 40, at most 500 trees |

The released DEV partition is used only to evaluate the frozen MSLR models.
It is never used for fitting, hyperparameter selection, or stopping. Neural
training uses relevance labels only. No real dataset other than the released
Fold1 TRAIN and DEV partitions is used. The official archive preparation opens
only its Fold1 TRAIN and validation entries; no held-out or other-fold data
is opened. No attempt is made to identify the undisclosed benchmarks.

## Fusion method and why

CombSUM sums all four provided base channels after min-max normalization
within each query. Constant channels contribute zero. It outperformed
CombMNZ and RRF with c in {0, 10, 30, 60, 100} in a predeclared TRAIN-only
comparison using 0.8 times raw nDCG@10 plus 0.2 times raw MAP@10 as a local
selection proxy. Full measurements are in `results/fusion_selection.json`.
RRF and both score-based textbook functions remain independently testable.
All ties preserve candidate order.

## LambdaMART hyperparameters and selection

The selected model has 1200 trees, 31 leaves, learning rate 0.05, minimum leaf
size 50, L2 1, feature fraction 0.8, bagging fraction 0.8, and bagging frequency
1. It uses seed 0, deterministic row-wise LightGBM training, and four threads.
The 136 provided features are used directly, without additional features.

The initial grouped five-fold TRAIN grid combines leaves {15, 31}, minimum
leaf size {50, 150}, L2 {1, 10}, and trees {50, 100, 200, 400}. The initial
winner was at 400 trees, so that configuration alone was extended to 800 and
1200 trees on the same folds. None of the reported CV folds controls early
stopping. Selection maximizes mean validation nDCG@10; ties prefer fewer
trees and leaves. All 34 rows and their five fold scores are retained in
`results/lambdamart_cv.json`, with a compact table in the root README.

## Final configuration and generalization tradeoff

The selected CV validation mean is 0.475014, sample SD 0.007563, training mean
0.643263, and validation/training ratio 0.738444. Extra trees increase training
accuracy much faster than validation accuracy. We retain the highest mean
validation result in the bounded sweep, with 31-leaf capacity, shrinkage,
and row/feature sampling, and document the increasing gap. Final fitting uses
full TRAIN only. Its independent reproduction is byte-identical.

Both neural scorers use the same 136 → 128 → 64 → 1 ReLU network,
TRAIN-fitted standardization, Adam at 0.001, batch size 1024, and seed 0.
Pointwise uses MSE; pairwise uses the plain stable RankNet loss with at most
300 distinct preference pairs per query per epoch. Internal TRAIN fold 0
chooses six pointwise epochs and one pairwise epoch from a maximum of 30,
with patience 5, before full TRAIN refits. Both artifacts use the unchanged
staff-loadable format and satisfy the architecture bounds.

Final DEV nDCG@10 is 0.482608 for LambdaMART, 0.431708 for pointwise, and
0.421922 for pairwise. Single-feature BM25 scores 0.288487. These are raw
metrics; class percentiles and private-track scores are unavailable locally.
The report is deferred until after submission feedback.

## AI-use disclosure

OpenAI Codex assisted with reading the assignment and starter files,
implementing and reviewing the algorithms, writing tests and training scripts,
configuring and running IIT Delhi HPC experiments, interpreting measurements,
and preparing repository documentation and the submission archive.

## Code provenance

The original starter provides the unchanged harness, MLP architecture,
standardizer, metrics, and grouped-fold helper. Fusion, squared-error, and
RankNet formulas follow the assignment PDF and starter primer. No external
fusion or neural ranking-training implementation was copied. LightGBM's
permitted `lambdarank` objective implements the tree ranker.

The [Microsoft Research dataset page](https://www.microsoft.com/en-us/research/project/mslr/)
and [IIT Delhi HPC guide](https://supercomputing.iitd.ac.in/) were consulted
for official data access and scheduled execution. Data provenance and all
split usage are stated above and in the root README.
