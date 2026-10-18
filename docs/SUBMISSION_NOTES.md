# Submission notes

## Training-data declaration

| Artifact | Fitted on | Tuned / stopped on |
|---|---|---|
| LambdaMART `model.txt` | Full MSLR-WEB10K Fold1 TRAIN | Grouped five-fold TRAIN CV; fixed 800-tree final fit |
| `pointwise_model.pt` | Full Fold1 TRAIN | Internal TRAIN fold 0; six-epoch full refit |
| `pairwise_model.pt` | Full Fold1 TRAIN | Internal TRAIN fold 0; six MSE warm-up epochs plus two weighted-pairwise refit epochs |
| `recipe.py` when invoked | Supplied benchmark TRAIN | Fixed adaptive design; supplied validation for early stopping |

DEV is used only for final evaluation after selection. No held-out labels,
other dataset folds, external relevance labels, or distillation targets are
used. Official archive preparation opens only Fold1 TRAIN and validation.
No attempt is made to identify the undisclosed benchmarks.

## Fusion method

Weighted CombSUM uses within-query min-max scores, with weights 0.25 for BM25
whole document, 0.25 for BM25 title, 0.50 for LMIR Dirichlet, and 0 for LMIR JM.
It is selected on TRAIN from 41 fixed settings using the raw 0.8 nDCG + 0.2 MAP
proxy. Textbook RRF, CombSUM, and CombMNZ remain separately testable.

## Features and LambdaMART configuration

The 136 raw MSLR features are augmented with 136 label-free within-query
percentiles, using tied midranks and preserving NaNs. The text model identifies
this transformation through feature names. The transfer recipe applies the
same transformation to its arbitrary input width, with bounded sorting memory.

Final MSLR LambdaMART: 800 trees, 127 leaves, unlimited depth,
learning rate 0.05, minimum leaf size 100, L2 10, feature fraction 0.8, bagging
fraction 0.8/frequency 1, truncation 30, seed 0, deterministic row-wise training,
four threads. All 49 complete configurations and five-fold scores are in
`results/lambdamart_cv.json`; no reported fold controls early stopping.

## Final configuration and tradeoff

The selected validation mean is 0.484133, sample SD
0.008215, training mean 0.782033, and validation/training
ratio 0.619071. Larger raw-feature trees failed to improve on
the first-round result; query-relative features improved it. We choose the
highest validation mean in the bounded search and explicitly retain the
substantial training gap. This favors measured validation accuracy over
the smaller candidates' narrower gaps; hidden robustness remains unmeasured
locally. DEV plays no part in that choice. Independent refitting checks exact
tree-artifact reproduction.

Both neural models keep the same 136 → 128 → 64 → 1 ReLU architecture,
TRAIN-fitted standardization, Adam, batch size 1024, and seed 0. Pointwise MSE
uses six full-TRAIN epochs at 0.001. Pairwise uses six supervised MSE warm-up
epochs at 0.001, then two epochs at 0.0001 with separate absolute-nDCG-swap
weighted RankNet training and up to 300 distinct pairs per query per epoch.
The graded `ranknet_loss` remains plain and unweighted. Internal TRAIN fold 0
selects the fine-tuning epoch and configuration before full refits.

Final DEV nDCG@10: LambdaMART 0.491177, pointwise
0.431708, pairwise 0.444853. The BM25 floor is
0.288487. These are raw DEV results, not private-track scores. The report
is deferred until submission feedback.

## AI-use disclosure

OpenAI Codex assisted with reading the assignment and starter files,
implementing and reviewing algorithms, writing tests and training scripts,
configuring and running IIT Delhi HPC experiments, interpreting measurements,
and preparing documentation and the submission archive.

## Code provenance

The starter supplies the unchanged harness, MLP architecture, standardizer,
metrics, and grouped folds. Fusion, squared-error, and RankNet formulas follow
the assignment and primer. Query percentiles and nDCG swap weights are
implemented directly; no external fusion or neural ranking trainer was copied.
LightGBM's permitted `lambdarank` objective implements the tree ranker.

The [Microsoft Research dataset page](https://www.microsoft.com/en-us/research/project/mslr/)
and [IIT Delhi HPC guide](https://supercomputing.iitd.ac.in/) were consulted for
data access and scheduled execution.
