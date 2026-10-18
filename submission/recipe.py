"""
submission/recipe.py -- your training recipe (assignment Section 4.1, Track C1).

Staff call train() ONCE per hidden benchmark (two undisclosed flat-feature LTR datasets) on that
benchmark's train and vali files, in an isolated directory, then evaluate the model it writes on the benchmark's
held-out split through YOUR prepare() / rerank_with_ltr(). Contract:

  * signature: train(train_path, vali_path, out_model_path, seed=0) -> None
  * the files are flat-feature SVMLight (label qid:N 1:v ... D:v), D unknown (at most ~300), labels integers
    0..4 (possibly fewer levels), at most ~500k training rows; feature meanings are NOT given
  * write a LightGBM text model (or whatever your prepare() loads) to out_model_path
  * 20 minutes wall-clock PER BENCHMARK on 4 CPU threads; no network; read only the two input files; deterministic given seed
  * a recipe that raises, times out, or writes a model prepare() cannot load scores 0 on that benchmark

The provided default is a fixed-hyperparameter LambdaMART. Replace the hyperparameters with choices that
transfer: the point of Track C1 is that settings picked by cross-validation on MSLR TRAIN should carry over to
a different dataset -- e.g. express tree count / leaf size / min_data_in_leaf as functions of the number of
queries and features rather than hard-coding numbers tuned on MSLR, or run a small CV inside the time limit.

Test it locally on the synthetic transfer sets:  python scripts/run_recipe.py
"""
from submission.data import group_sizes, load_ltr
from submission.features import tree_features, tree_feature_names


def fit_arrays(X, y, q, Xv, yv, qv, seed=0):
    """Choose capacity using only the supplied validation split, with fixed work.

    Two fits of at most 200--500 trees each, with 40-round patience. No timing-based
    stopping decisions: identical inputs and seed follow the same search.
    """
    import lightgbm as lgb

    width = X.shape[1]
    max_rounds = max(200, min(500, 30_000_000_000 // max(1, len(y) * width)))
    names = tree_feature_names(width, "query_rank")
    X = tree_features(X, q, "query_rank")
    Xv = tree_features(Xv, qv, "query_rank")
    n_queries = len(group_sizes(q))
    params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
        "learning_rate": 0.05,
        "min_data_in_leaf": max(20, min(200, len(y) // 2000)),
        "lambda_l2": 10.0,
        "feature_fraction": 0.9 if width < 100 else 0.8,
        "bagging_fraction": 0.9, "bagging_freq": 1, "seed": seed,
        "lambdarank_truncation_level": 13,
        "deterministic": True, "force_row_wise": True, "verbose": -1, "num_threads": 4,
    }
    training = lgb.Dataset(X, label=y, group=group_sizes(q), feature_name=names)
    validation = lgb.Dataset(Xv, label=yv, group=group_sizes(qv), reference=training, feature_name=names)
    capacities = (15, 31) if n_queries < 200 else (31, 63) if n_queries < 2000 else (63, 127)
    best, best_score, history = None, -float("inf"), []
    for leaves in capacities:
        booster = lgb.train(dict(params, num_leaves=leaves), training, num_boost_round=max_rounds,
                            valid_sets=[validation], callbacks=[lgb.early_stopping(40, verbose=False)])
        score = booster.best_score["valid_0"]["ndcg@10"]
        history.append({"leaves": leaves, "trees": booster.best_iteration, "maximum_trees": max_rounds,
                        "validation_lightgbm_ndcg@10": score})
        if score > best_score + 1e-12:
            best, best_score = booster, score
    return best, history


def train(train_path: str, vali_path: str, out_model_path: str, seed: int = 0) -> None:
    (X, y, q), (Xv, yv, qv) = load_ltr(train_path, vali_path, allow_nan=True)
    booster, _ = fit_arrays(X, y, q, Xv, yv, qv, seed)
    # Save only the supplied-validation-selected iteration.
    booster.save_model(out_model_path)
