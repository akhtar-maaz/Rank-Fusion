# A3 — Rank Fusion and Learning to Rank

Current submission: V12. The formal report remains deferred.

## Data declaration

The shipped tree was fitted on released **Fold1 TRAIN and DEV**: 723,412
TRAIN rows/6,000 queries plus 235,259 DEV rows/2,000 queries, 958,671 original
rows/8,000 split-namespaced queries and 136 raw features. The assignment
permits disclosed DEV fitting. Tree DEV accuracy is **in-sample** and cannot
establish unseen-query improvement. Tree status: exact evaluated V10 artifact retained; no V12 final tree fit.
Pairwise status: fresh V12 full-TRAIN fit, independently reproduced with identical bytes. Retained artifacts are preserved copies,
not newly fitted models.

All V12 tuning, selection and stopping use released TRAIN only, five
query-grouped folds, seed 0. There is no new DEV-based tuning, early stopping
or post-fit choice. No official test, other MSLR fold, outside relevance
label, distillation target or undisclosed benchmark was accessed. The
formal assignment report remains deferred.

| Artifact | Fitting data | Selection and stopping |
|---|---|---|
| `model.txt` | TRAIN + DEV; retained V10 bytes | TRAIN-only CV; fixed 3,800 trees; no early stopping |
| `pointwise_model.pt` | TRAIN only; exact retained bytes | Original TRAIN fold 0; six MSE epochs; no V12 fit |
| `pairwise_model.pt` | TRAIN only; fresh V12 fit | TRAIN-only fixed-epoch exported-MLP selection; no DEV fitting |
| `fusion_config.json` | No trained scorer; exact V11 weights retained | V11 TRAIN-only fixed grid; no V12 change |
| `recipe.py` | Staff-supplied benchmark TRAIN | Unchanged C1 recipe; supplied validation chooses capacity/stopping |

## V12 TRAIN selection and final evaluation

The [frozen protocol](docs/EXPERIMENTS_V12.md) preceded measurements. Complete
TRAIN-only choices were audited, committed and pushed at
`e93ba47c1e223bab1b11df22a9a9f5d5551070cb` before final fits or the new DEV run.
See `results/v12_selection_audit.json`, `results/lambdamart_cv_v12.json`,
`results/neural_selection_v12.json`, `results/cv_v12/`, and
`results/neural_v12/` for all parts, controls, gates and native-export checks.

Tree comparison included four balanced-domain candidates and 33 fixed settings.
Each fit retains original queries plus affine, feature-dropout and noisy
versions after query splitting, with labels copied unchanged. Strengths 1/2,
feature fractions .8/.5, seed 20261024 and 200/400-tree prefixes were fixed.
The exact historical TRAIN-only control was reproduced. Tree promotion:
**False**. Public diagnostic scenarios are not reconstructions of staff C2.

Neural comparison freshly fitted all 50 members across five folds. The exact
V8 warm-up-20, seeds-0/41 control reproduced its archived exported metrics,
member histories and comparable artifact/state/statistics identities. The four
new fixed configurations use warm-up 30/40 then two RankNet epochs, or
warm-up 20/30 then four RankNet epochs. The proposed exported-MLP mean gain
was 0.001102825, with 4 positive paired
folds; the frozen .001/three-fold gate gives promotion **True**.
No runner-up was substituted.

| Final local DEV metric | V12 value | Interpretation |
|---|---|---|
| Fusion nDCG@10 | 0.309012995 | Unchanged TRAIN-selected V11 weights |
| Fusion MAP@10 | 0.129671266 | Unchanged TRAIN-selected V11 weights |
| Tree nDCG@10 | 0.682399113 | In-sample TRAIN+DEV tree fit |
| Pointwise nDCG@10 | 0.431707743 | Retained TRAIN-only fit |
| Pairwise nDCG@10 | 0.452027035 | TRAIN-only fit; final DEV never selects it |

Latest supplied feedback is V11 submission 558: rank 10/35, B .6824,
held-out P .4400, C2 .3795, live .4797/.65, 7/7 conformance and 10/10
correctness (`results/leaderboard_v11.json`). V12 has no staff evaluation at
packaging time. Neither local DEV nor TRAIN diagnostics predict a private
score, class percentile or rank.

## Shipped tree method and provenance

The native weighted tree export contains 3,800 trees and 408
internal features for 136 raw inputs. Frozen fit record: `results/tree_fit_v10.json`.
Current `results/lambdamart_cv.json` separates V12 TRAIN selection from
the final fitting provenance. SHA256: `f2fa40faac67aaab067bb43eb169fd7d7db5a13c5eeb0a6d248c88f941e44864`.

| Member | Feature mode | Trees | Weight | Augmentation |
|---|---|---|---|---|
| 1 | query_rank | 800 | 0.0703125 | none |
| 2 | query_normalized | 200 | 0.0703125 | none |
| 3 | query_rank | 800 | 0.140625 | {'name': 'half_query_dropout_noise', 'seed': 20261019} |
| 4 | query_all | 800 | 0.28125 | {'name': 'half_query_mixed', 'seed': 20261020} |
| 5 | query_all | 800 | 0.1875 | {'name': 'full_query_mixed', 'seed': 20261021} |
| 6 | query_all | 400 | 0.25 | {'name': 'full_query_mixed', 'seed': 20261021} |

The six retained members use LambdaRank with 127 leaves, minimum leaf 100,
L2 10, learning rate .05, feature/bagging fractions .8, bagging frequency 1,
seed 0, four threads and fixed counts. Truncation is 30 except member 3 uses
13; member 6 uses extra trees with seed 0. Their original exact configuration
is `configs/v10_tree.json`. Query transforms
inspect only the current candidates, preserve missing values and give .5
to constants. Native export was checked against the explicit weighted
average on all five TRAIN validation folds. Final independent-fit evidence
belongs to V12 only when refitted; retained V10 records remain historical.

## Neural models, fusion and graded source

Pairwise is a plain staff-loadable 136→256→128→1 ReLU MLP compiled from
equally weighted seeds 0 and 41, each 136→128→64→1. Each member uses MSE
warm-up at .001 followed by fresh-Adam nDCG-weighted handwritten RankNet
at .0001, pair cap 300 and batch size 1,024. The exact selected warm-up
and epoch counts are in `results/pairwise_refit_v12.json`. The shipped members use
30 MSE warm-up epochs and 2 RankNet epochs. Compilation includes float32 effects
and is checked against arithmetic averaging at atol 2e-6/rtol 2e-5.

Pointwise remains 136→128→64→1 with six MSE epochs at .001. Its artifact,
history and selection are unchanged. Corrected deterministic initialization,
all graded neural losses/training functions and C1 recipe preserve evaluated
source bytes. Fusion remains handwritten minmax CombSUM with weights .3
BM25 whole-document, .3 BM25 title, .4 LMIR.DIR and 0 LMIR.JM; required
unweighted RRF/CombSUM/CombMNZ remain implemented and tested.

## Verification

All 176 tests pass locally and on the college server. All
9 actual V12 PBS jobs succeeded; complete requested/observed
resources, final-fit status and source CI are in `results/execution.json`.
Source CI [39775028153](https://github.com/akhtar-maaz/Rank-Fusion/actions/runs/39775028153) passed.
Local/HPC content hashes agree and ranking is deterministic. Pristine staff
loading passes on local Python 3.12.14; HPC uses
Python 3.9.21. Preparation took 5.680s,
maximum fusion 0.001s and reranking
0.488s, within the 60s/2s limits.
All seven earlier evaluated ZIPs and all protected artifacts remain intact.

## Adaptive C1 recipe



`submission/recipe.py` reads only its two supplied files, handles arbitrary
feature width and labels 0–4, and uses raw values plus query percentiles.
It validates two capacities: 15/31 leaves below 200 training queries,
31/63 below 2,000, otherwise 63/127. Validation selects capacity and the
best iteration; exact ties keep fewer leaves.

Both fits use learning rate 0.05, L2 10, minimum leaf size
`max(20, min(200, training_rows // 2000))`, feature fraction 0.9 below 100
raw features (otherwise 0.8), row fraction 0.9, truncation 13, seed supplied
by staff, four threads and patience 40. Each fit's deterministic tree ceiling
is `max(200, min(500, 30_000_000_000 // (training_rows * raw_width)))`.
There are no wall-clock-based selection decisions. Native missing-value
handling preserves NaNs.

Three constructed TRAIN-only nested holdouts improve mean outer nDCG from
0.448871 to
0.450507. These single-holdout method
checks do not estimate hidden-benchmark performance. Records are in
`results/recipe_domains_v3/` and `results/recipe_selection_v3.json`.
The wide compute check fits 499,918 rows × 300 derived
columns, validates 142,390 rows, finishes in
208.2 seconds and uses
4.19 GiB peak RSS. It repeats/rescales
TRAIN columns in memory; parsing uses the original 136-feature file.
It is a budget check, not an unknown-benchmark simulation. The provided
40/90/250-feature synthetic portability tests also pass.

## Setup and reproduction

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/smoke_test.sh
```

V12 experiments use the frozen configurations and scripts named in
`docs/EXPERIMENTS_V12.md`; historical controls need their original fold
caches. The runtime for exact college reproduction is Python 3.9.21;
source, configuration, input hashes and versions are recorded per fit.
No fitting happens during grading. Real data, environments, `runs/`, `dist/`
and Git metadata are excluded from the committed archive. Original starter
paths are retained. See [docs/HPC.md](docs/HPC.md).

## AI-use disclosure and code provenance



OpenAI Codex assisted with reading the specification, implementation, tests,
HPC execution, experiment interpretation, documentation and packaging.
The unchanged starter supplies the harness, MLP, standardizer, metrics and
grouped-fold helper. Fusion, MSE and RankNet follow the assignment and primer.
Percentiles, swap weighting, diagnostic perturbations, training augmentation,
native tree averaging, bounded plain-MLP averaging and adaptive recipe logic are implemented directly; no external fusion or
neural ranking trainer is copied. LightGBM's permitted ranking objectives `lambdarank` and `rank_xendcg`
train the tree candidates; final member objectives are documented above. Consulted references include the official dataset page,
[LightGBM 4.6 parameter documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html),
and the [IIT Delhi HPC guide](https://supercomputing.iitd.ac.in/).
See [docs/SUBMISSION_NOTES.md](docs/SUBMISSION_NOTES.md) for the declarations.
