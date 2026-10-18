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


def train(train_path: str, vali_path: str, out_model_path: str, seed: int = 0) -> None:
    import lightgbm as lgb

    (X, y, q), (Xv, yv, qv) = load_ltr(train_path, vali_path, allow_nan=True)
    width = X.shape[1]
    names = tree_feature_names(width, "query_rank")
    X = tree_features(X, q, "query_rank")
    Xv = tree_features(Xv, qv, "query_rank")
    n_queries = len(group_sizes(q))
    params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
        "num_leaves": 15 if n_queries < 200 else 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": max(20, min(200, len(y) // 2000)),
        "lambda_l2": 10.0,
        "feature_fraction": 0.9 if width < 100 else 0.8,
        "bagging_fraction": 0.9, "bagging_freq": 1, "seed": seed,
        "lambdarank_truncation_level": 15,
        "deterministic": True, "force_row_wise": True, "verbose": -1, "num_threads": 4,
    }
    training = lgb.Dataset(X, label=y, group=group_sizes(q), feature_name=names)
    validation = lgb.Dataset(Xv, label=yv, group=group_sizes(qv), reference=training, feature_name=names)
    booster = lgb.train(params, training, num_boost_round=500, valid_sets=[validation],
                        callbacks=[lgb.early_stopping(40, verbose=False)])
    # Save only the validation-selected iteration; never access an evaluation split.
    booster.save_model(out_model_path)
