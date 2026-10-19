# Submission notes — v6

## Training-data declaration

| Artifact | Fitted on | Selection / stopping |
|---|---|---|
| `model.txt` | Full Fold1 TRAIN only | Five-fold grouped TRAIN CV; fixed 3400-tree native model |
| `pointwise_model.pt` | Full Fold1 TRAIN only | Internal TRAIN fold 0; fixed six-epoch refit |
| `pairwise_model.pt` | Full Fold1 TRAIN only | Five-fold fixed-epoch TRAIN assessment; 10 MSE plus 2 weighted RankNet epochs |
| `recipe.py` | Supplied benchmark TRAIN only when staff call it | Supplied validation selects two capacities and stopping point |

DEV is evaluated only after configurations are frozen. It is never used for
fitting, tuning or stopping the shipped MSLR models. No official test labels,
other MSLR folds, external relevance labels or distillation targets are used.
No attempt is made to identify or inspect undisclosed C1 datasets. Website
feedback is user-provided aggregate feedback only.

## Final configuration and tradeoff

The promotion rule selected `v5_plus_3_800_weight0.25`.

| Member | Features | Trees | Weight | Augmentation |
|---:|---|---:|---:|---|
| 1 | query_rank | 800 | 0.09375 | none |
| 2 | query_normalized | 200 | 0.09375 | none |
| 3 | query_rank | 800 | 0.1875 | half_query_dropout_noise |
| 4 | query_all | 800 | 0.375 | half_query_mixed |
| 5 | query_all | 800 | 0.25 | full_query_mixed |

| Member | Leaves | Minimum leaf | L2 | Feature fraction | Seed | Truncation |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 127 | 100 | 10 | 0.8 | 0 | 30 |
| 2 | 127 | 100 | 10 | 0.8 | 0 | 30 |
| 3 | 127 | 100 | 10 | 0.8 | 0 | 13 |
| 4 | 127 | 100 | 10 | 0.8 | 0 | 30 |
| 5 | 127 | 100 | 10 | 0.8 | 0 | 30 |

The [v6 protocol](EXPERIMENTS_V6.md) records all 49
complete tree settings and four complete fixed-epoch neural candidates.
Clean tree CV changes from 0.487398 to 0.487343;
the diagnostic mean from 0.474075 to 0.475631.
Selected clean SD is 0.010004, training mean
0.758586, and validation/training ratio 0.642436.
All augmentation runs after splitting; labels are copied without inspection.
The generalization gap and reused-fold adaptive estimates limit confidence.
No hidden C2 gain is established by these diagnostics.

Pairwise `longer_warmup` uses 10 MSE warm-up epochs,
then 2 nDCG-weighted RankNet epochs at 0.0001,
batch size 1024 and at most 300 pairs
per query. Network architecture stays 136→128→64→1 ReLU. Five-fold fixed
TRAIN CV selected this configuration, without DEV use.

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

96 tests pass. The actual frozen DEV run uses `rerank_with_ltr`; neural
artifacts load with the unchanged staff-compatible loader under Python 3.12.
Both promoted artifacts have byte-identical independent full-TRAIN refits.
Hashes and execution records are in `results/`. Fusion and pointwise retain
v2 hashes; the C1 recipe retains its v3 hash.
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
