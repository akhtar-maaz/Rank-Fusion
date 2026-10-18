# Submission notes — v3

## Training-data declaration

| Artifact | Fitted on | Selection / stopping |
|---|---|---|
| `model.txt` | Full Fold1 TRAIN only | Five-fold grouped TRAIN CV; fixed 800+200-tree refit |
| `pointwise_model.pt` | Full Fold1 TRAIN only | Internal TRAIN fold 0; fixed six-epoch refit |
| `pairwise_model.pt` | Full Fold1 TRAIN only | Internal TRAIN fold 0; fixed six MSE plus two weighted RankNet epochs |
| `recipe.py` | Supplied benchmark TRAIN only when staff call it | Supplied validation selects two capacities and stopping point |

DEV is evaluated only after configurations are frozen. It is never used for
fitting, tuning or stopping the shipped MSLR models. No held-out labels,
other MSLR folds, external relevance labels or distillation targets are used.
No attempt is made to identify or inspect undisclosed C1 datasets. Website
feedback is user-provided aggregate feedback only.

## Final configuration and tradeoff

The native LightGBM artifact averages an 800-tree raw-plus-percentile
LambdaMART and a 200-tree min-max-plus-percentile LambdaMART, each weighted
0.5. Both use 127 leaves, minimum leaf size 100, L2 10, learning rate 0.05,
feature and row fractions 0.8, truncation 30, seed 0, deterministic row-wise
fitting and four threads. Its 1,000 native trees accept 136 raw inputs via
408 label-free query-local features. Selection retains clean CV accuracy
within 0.0025 of the best complete candidate and favors the declared clean /
diagnostic mixture. Clean CV improves from 0.484133 to 0.485550;
diagnostics improve from 0.454990 to 0.468450. The
validation/training ratio improves from 0.619071 to 0.674514.
The remaining gap and sample SD 0.008929 limit what
these adaptive selection estimates establish. No hidden C2 gain is claimed.

## Transfer policy and limits

Two supplied-validation-selected leaf counts adapt to training query count:
15/31, 31/63 or 63/127 at cutoffs 200 and 2,000. Features are raw plus query
percentiles, regardless of their meaning. Minimum leaf size scales with row
count. Each fit has a deterministic 200–500-tree cap based on rows × raw
width, patience 40, L2 10, learning rate 0.05, truncation 13 and four threads.
The near-maximum TRAIN-derived 300-column check finishes in
208.2s at 4.19 GiB,
with caveats recorded in `results/recipe_budget_v3.json`. Synthetic checks
prove portability; the unknown datasets remain staff-only.

## Verification and reproducibility

76 tests pass. The actual frozen DEV run uses `rerank_with_ltr`; neural
artifacts load with the unchanged staff-compatible loader under Python 3.12.
The independent full-TRAIN tree refit is byte-identical; hashes and execution
records are in `results/`. Fusion and both neural artifacts retain v2 hashes.
The averaging exporter is checked against explicit arithmetic averaging
before and after native save/load. Original starter files remain in the ZIP;
real data, wheels, environments, Git metadata and intermediate runs do not.

## AI-use disclosure

OpenAI Codex assisted with specification reading, algorithms, implementation,
tests, environment setup, scheduled HPC training, diagnostics, interpretation,
documentation and packaging. The student must understand and explain these
methods during the course walkthrough. Report writing remains deferred.

## Code-provenance statement

The original starter supplies the unchanged harness, network, standardizer,
grouped folds and metrics. Hand-written fusion, MSE and RankNet follow the
assignment and primer. Query transforms, metric swap weights, perturbations,
native numeric-tree export and adaptive capacity/budget rules are implemented
directly. No external fusion library, neural ranking trainer, solution code,
other student's code, or extra labeled dataset is used. LightGBM 4.6's
permitted ranking objective implements the tree learner. Official references:
[MSLR](https://www.microsoft.com/en-us/research/project/mslr/),
[LightGBM parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html),
[IIT Delhi HPC](https://supercomputing.iitd.ac.in/).
