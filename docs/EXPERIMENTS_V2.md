# Second-round TRAIN-only experiments

The first upload passes 7/7 conformance and 10/10 correctness checks. Its
user-reported aggregate scores are retained in `results/leaderboard_v1.json`.
The shipped artifacts remain the first-round models until these experiments
finish and a candidate wins on TRAIN validation.

## Tree ablations

Every row uses grouped five-fold TRAIN CV, seed 0, fixed tree checkpoints
400/800/1600, learning rate 0.05, feature and row fractions 0.8, four threads.
No reported fold controls early stopping.

| Candidate | Features | Leaves | Minimum leaf size | L2 |
|---:|---|---:|---:|---:|
| 0 | Original 136 | 63 | 50 | 1 |
| 1 | Original 136 | 127 | 100 | 10 |
| 2 | Original 136 plus query percentiles | 63 | 50 | 1 |
| 3 | Original 136 plus query percentiles | 127 | 100 | 10 |

Percentiles use tied midranks within each query, computed without labels.
Each query is transformed independently before CV splitting, so features
never combine information from training and validation queries. The LightGBM
text model carries feature names identifying this transform; the same
transform is applied within `rerank_with_ltr`. Plain recipe models continue
to receive their original features. These properties have regression tests.

Compare the best mean validation result with the first-round best, 0.475014.
Keep the first-round artifact if no candidate improves. These are selection
estimates, not unbiased hidden-test estimates.

## Pairwise ablations

All candidates keep the original 136 → 128 → 64 → 1 MLP and staff artifact
format. Use seed 0, 300 pairs per query, batch size 1024, at most 20 epochs,
patience 5 on internal TRAIN fold 0, followed by fixed-epoch full TRAIN refit.

| Candidate | Loss used for training | Learning rate |
|---:|---|---:|
| 0 | Plain RankNet | 0.0003 |
| 1 | RankNet weighted by absolute nDCG@10 swap change | 0.0003 |
| 2 | RankNet weighted by absolute nDCG@10 swap change | 0.001 |

Weights are recomputed from current within-query ranks each epoch, with the
same exponential gain and cutoff as the metric. The weighted loss is a
separate function; the graded `ranknet_loss` remains plain and unweighted.
Compare internal selection nDCG against the original 0.426320. Keep the
original pairwise artifact if no candidate improves. The pointwise artifact
keeps the same architecture and its original TRAIN-only fit.

## Fusion ablation and final evaluation

The original seven fusion settings are compared with 34 additional weighted
CombSUM settings: nonnegative quarter-step channel weights summing to one,
with the equal-weight duplicate omitted. Selection uses TRAIN only and the
same raw 0.8 nDCG + 0.2 MAP proxy.

Freeze selected artifacts and fusion before final DEV evaluation. No hidden
labels are used. Hidden robustness construction is unknown, and an improved
TRAIN CV score does not establish an improved C2 or class-percentile score.
