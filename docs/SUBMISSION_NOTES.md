# Submission notes — v8

## Training-data declaration

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

| Artifact | Fit data | Selection / stopping |
|---|---|---|
| `model.txt` | Full Fold1 TRAIN only | Five grouped TRAIN folds; frozen 3800-tree native configuration |
| `pointwise_model.pt` | Full Fold1 TRAIN only | Original internal TRAIN fold 0; fixed six MSE epochs |
| `pairwise_model.pt` | Full Fold1 TRAIN only | Five fixed-epoch TRAIN folds; `warmup20_seeds_0_41` |
| `recipe.py` | Only supplied benchmark TRAIN | Supplied validation chooses capacity and stopping |

## Final configuration and tradeoff

The independent rules retained the exact v7.1 tree artifact and promoted the selected pairwise average.

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

| TRAIN-CV condition | v7 | v8 |
|---|---|---|
| clean | 0.487560 | 0.487560 |
| affine | 0.479000 | 0.479000 |
| feature_dropout | 0.477154 | 0.477154 |
| noise | 0.474921 | 0.474921 |

The highest eligible candidate was `v7_plus_1_600_weight0.25`: mixture gain +0.000720009, diagnostic gain +0.000896714, and positive paired mixture gains in 4/5 folds. Gate outcomes (mixture/diagnostic/paired) are True/False/True.

The [frozen v8 protocol](EXPERIMENTS_V8.md) records all 49
complete tree settings and five complete neural candidates. Reused-fold
selection and the training/validation gap limit confidence. Augmentation
occurs only after splitting and copies relevance labels without inspection.
No local diagnostic establishes a hidden C2 gain.

Shipped pairwise architecture: 136→256→128→1 ReLU.

| Member seed | Hidden sizes | Weight | MSE epochs | RankNet epochs | RankNet LR | Pair cap | Batch |
|---|---|---|---|---|---|---|---|
| 0 | 128→64 | 0.5 | 20 | 2 | 0.0001 | 300 | 1024 |
| 41 | 128→64 | 0.5 | 20 | 2 | 0.0001 | 300 | 1024 |

Two selected 128/64 members compile by concatenated first layers, block-diagonal second layers and weighted output rows into the shipped ordinary 256/128 MLP. The network fits staff's width/depth limits and needs no custom inference loader.
Selection uses exported-artifact nDCG for both candidate types.
The full-TRAIN compiled export differs from the explicit arithmetic score average by at most 9.54e-07; full-TRAIN exported/explicit nDCG are 0.473319001/0.473334219. Selection uses the exported artifact, including float32 compilation effects.

## Reproduction

Tree: `python scripts/train_model.py --train data/train --config results/lambdamart_cv.json --out runs/reproduced_v8/model.txt`.
For a retained tree, the final job preserved exact v7.1 bytes instead of refitting.

Pairwise (fresh member fits if promoted; exact shipped copy if retained).
Use the recorded sources/runtime for exact reproduction: Linux CPU Torch
2.8.0+cpu and the versions in `results/execution.json`, four threads:

```bash
python scripts/cv_pairwise_ensemble_v8.py --mode fit --train data/train/train.txt --specs configs/v8_neural_candidates.json --selection results/neural_selection_v8.json --control-report results/neural_selection_v7.json --baseline submission/pairwise_model.pt --fresh --cache-dir runs/reproduction_v8/neural_cache --out runs/reproduction_v8/pairwise_model.pt --history-out runs/reproduction_v8/pairwise_history.json
```

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

All 142 tests pass locally and on the college server. Actual frozen DEV
evaluation meets prepare/rerank limits. Artifacts match HPC hashes and load
through the pristine staff neural loader under Python 3.12. Independent
fresh fits reproduce promoted artifacts byte-for-byte; retained artifacts
preserve v7.1 bytes. Pointwise, fusion and the C1 recipe stay unchanged.
Execution evidence covers ten actual scheduled job records with per-job
requested/observed resources. No real data, environments, Git metadata or
intermediate fits enter the ZIP. Original starter files remain present.

## AI-use disclosure

OpenAI Codex assisted with specification reading, algorithms, implementation,
tests, environment setup, scheduled HPC training, diagnostics, interpretation,
documentation and packaging. The student must understand and explain these
methods during the course walkthrough. Report writing remains deferred.

## Code-provenance statement

The original starter supplies the unchanged harness, network, standardizer,
grouped folds and metrics. Hand-written fusion, MSE and RankNet follow the
assignment and primer. Query transforms, metric swap weights, perturbations,
training augmentation, native numeric-tree export, bounded plain-MLP averaging and adaptive capacity/budget
rules are implemented directly. No external fusion library, neural ranking trainer, solution code,
other student's code, or extra labeled dataset is used. LightGBM 4.6's
permitted ranking objective implements the tree learner. Official references:
[MSLR](https://www.microsoft.com/en-us/research/project/mslr/),
[LightGBM parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html),
[IIT Delhi HPC](https://supercomputing.iitd.ac.in/).
