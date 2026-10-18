"""Track C1 contract: submission/recipe.py trains a loadable model on a flat-feature file of ANY dimension
and label range, deterministically given the seed."""
import glob
import os

import pytest

from harness.recipe_eval import run_recipe_and_evaluate

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "toy_transfer")
SETS = sorted(glob.glob(os.path.join(ROOT, "*")))


def test_transfer_sets_present():
    assert len(SETS) == 3


@pytest.mark.parametrize("d", SETS, ids=[os.path.basename(s) for s in SETS])
def test_recipe_trains_and_scores(tmp_path, d):
    vali = os.path.join(d, "vali.txt")
    r = run_recipe_and_evaluate(os.path.join(d, "train.txt"), vali, vali, str(tmp_path / "m.txt"), seed=0)
    assert r["ndcg@10"] > 0.5 and r["num_eval_queries"] == 50


def test_recipe_deterministic_given_seed(tmp_path):
    d = SETS[0]
    vali = os.path.join(d, "vali.txt")
    a = run_recipe_and_evaluate(os.path.join(d, "train.txt"), vali, vali, str(tmp_path / "a.txt"), seed=3)
    b = run_recipe_and_evaluate(os.path.join(d, "train.txt"), vali, vali, str(tmp_path / "b.txt"), seed=3)
    assert a["ndcg@10"] == b["ndcg@10"]


def test_recipe_preserves_native_missing_values(tmp_path):
    train, vali = tmp_path / "train.txt", tmp_path / "vali.txt"
    for path, queries in ((train, range(40)), (vali, range(40, 50))):
        path.write_text("".join(f"{d % 3} qid:{q} 1:{d % 3} 2:nan # synthetic\n"
                                for q in queries for d in range(10)))
    result = run_recipe_and_evaluate(str(train), str(vali), str(vali), str(tmp_path / "model.txt"))
    assert result["ndcg@10"] > 0.9
