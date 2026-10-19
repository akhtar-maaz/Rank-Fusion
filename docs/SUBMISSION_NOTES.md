# Submission notes — v5

## Training-data declaration

| Artifact | Fitted on | Selection / stopping |
|---|---|---|
| `model.txt` | Full Fold1 TRAIN only | Five-fold grouped TRAIN CV; fixed 2600-tree native model |
| `pointwise_model.pt` | Full Fold1 TRAIN only | Internal TRAIN fold 0; fixed six-epoch refit |
| `pairwise_model.pt` | Full Fold1 TRAIN only | Internal TRAIN fold 0; fixed six MSE plus two weighted RankNet epochs |
| `recipe.py` | Supplied benchmark TRAIN only when staff call it | Supplied validation selects two capacities and stopping point |

DEV is evaluated only after configurations are frozen. It is never used for
fitting, tuning or stopping the shipped MSLR models. No official test labels,
other MSLR folds, external relevance labels or distillation targets are used.
No attempt is made to identify or inspect undisclosed C1 datasets. Website
feedback is user-provided aggregate feedback only.

## Final configuration and tradeoff

The frozen TRAIN-selected configuration is `v4_plus_1_800_weight0.5`:

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

All members use learning rate 0.05, row fraction 0.8, seed 0 and four
deterministic fitting threads. The native model accepts 136 raw features.
Any augmentation retains original training queries and appends deterministic
perturbed copies after splitting TRAIN. Validation queries never enter fitting.
The full protocol and all 54 complete settings are recorded.
Clean CV changes from 0.486043 to 0.487398;
the diagnostic mean from 0.470255 to 0.474075.
Selected sample SD is 0.009563; validation/training ratio
is 0.644007. The per-diagnostic regression guard
and clean guard apply before the mixture threshold. The remaining gap and
adaptive selection limit confidence. No hidden C2 gain is claimed.

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

91 tests pass. The actual frozen DEV run uses `rerank_with_ltr`; neural
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
training augmentation, native numeric-tree export and adaptive capacity/budget
rules are implemented directly. No external fusion library, neural ranking trainer, solution code,
other student's code, or extra labeled dataset is used. LightGBM 4.6's
permitted ranking objective implements the tree learner. Official references:
[MSLR](https://www.microsoft.com/en-us/research/project/mslr/),
[LightGBM parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html),
[IIT Delhi HPC](https://supercomputing.iitd.ac.in/).
