# Second-round TRAIN-only experiments

The first upload passes 7/7 conformance and 10/10 correctness checks. Its
user-reported aggregate scores are retained in `results/leaderboard_v1.json`.
The second-round artifacts are selected by the completed TRAIN-only protocol
below. Final measurements and all reported fold scores are retained in
`results/`; the first-round records remain available with `_v1` suffixes.

## Tree ablations

Every row uses grouped five-fold TRAIN CV, seed 0, learning rate 0.05,
feature and row fractions 0.8, and four threads. Candidates 0–2 use fixed
tree checkpoints 400/800/1600; the widest candidate 3 uses 200/400/800 to
bound its compute cost. The incomplete initial candidate-3 run is discarded.
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
transform is applied within `rerank_with_ltr`. Legacy raw-feature text models
continue to receive their original features. These properties have regression
tests.

Compare the best mean validation result with the first-round best, 0.475014.
Keep the first-round artifact if no candidate improves. These are selection
estimates, not unbiased hidden-test estimates.

The first completed comparison (candidate 0) peaks at 0.474716, below the
baseline, while its training score rises strongly. A bounded second stage
therefore tests two 31-leaf configurations with maximum depth 6 and LambdaRank
truncation 10, at 400/800/1600 trees on the same five TRAIN folds: candidate 4
uses raw features, minimum leaf size 50, L2 1; candidate 5 uses query-rank
features, minimum leaf size 100, L2 10. No DEV result is used for this decision.
Candidate 5 is subsequently cancelled for runtime before completing all five
folds. Its partial run is excluded; candidate 4 must complete all five folds
before promotion. The final comparison then contains 49 complete configurations.

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

After candidate 1 improved internal selection nDCG to 0.433502, a bounded
second stage tests six supervised MSE warm-up epochs before pairwise training:
candidate 3 uses nDCG weighting at learning rate 0.0001; candidate 4 uses plain
RankNet at 0.0003. Six warm-up epochs come from the original pointwise TRAIN
selection. Each uses at most 15 fine-tuning epochs with patience 4. Warm-up
is fitted separately on each training split, then repeated on full TRAIN
during refit. It uses relevance labels directly, with a fresh Adam optimizer
for fine-tuning. No DEV or other model's predictions enter training.

## Fusion ablation and final evaluation

The original seven fusion settings are compared with 34 additional weighted
CombSUM settings: nonnegative quarter-step channel weights summing to one,
with the equal-weight duplicate omitted. Selection uses TRAIN only and the
same raw 0.8 nDCG + 0.2 MAP proxy.

Freeze selected artifacts and fusion before final DEV evaluation. No hidden
labels are used. Hidden robustness construction is unknown, and an improved
TRAIN CV score does not establish an improved C2 or class-percentile score.

## Transfer recipe

The matched 63-leaf comparison supports the query-relative transform:
at 400 trees it changes TRAIN-CV validation mean from 0.474716 to 0.482138.
The updated transfer recipe therefore adds the same label-free percentiles
for arbitrary input widths, retaining its previous regularization, adaptive
leaf sizes, and 500-tree / patience-40 budget. Feature sorting processes at
most 16 columns at once to bound temporary memory for large queries. NaNs
stay missing, and finite tied values receive identical midranks. Finite MSLR
feature values are bit-identical to those used by the in-flight CV jobs.

The three synthetic dimension/label checks pass; they establish portability
rather than hidden-benchmark performance. No undisclosed benchmark is sought
or accessed.
