import copy

import pytest

from scripts.select_neural import choose_candidate


def _report(score, final_dev):
    return {"arguments": {"train": "data/train", "seed": 0, "folds": 5, "hidden": [128, 64], "dev": None},
            "models": {"pairwise": {"best_epoch": 1, "selection_history": [{"epoch": 1, "dev_ndcg@10": score}],
                                      "full_train_history": [{"epoch": 1}], "final_dev_ndcg@10": final_dev}}}


def test_neural_promotion_uses_internal_train_scores_only():
    assert choose_candidate([_report(0.4, 0.9), _report(0.45, 0.1)]) == 1
    assert choose_candidate([_report(0.4, 0.1), _report(0.39, 0.9)]) == 0


def test_neural_promotion_rejects_different_validation_folds():
    first = _report(0.4, 0.1)
    second = copy.deepcopy(first)
    second["arguments"]["seed"] = 1
    with pytest.raises(ValueError, match="incomparable"):
        choose_candidate([first, second])
