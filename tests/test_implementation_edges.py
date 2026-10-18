"""Edge cases beyond the hand-checkable starter examples."""
import numpy as np
import pytest

from submission import fusion, pointwise_pairwise as pp, ranking
from submission.data import group_sizes, load_ltr


def test_fusion_ties_negative_scores_and_constant_channels():
    base = {"b": {"x": -2, "constant": 8}, "a": {"x": -2, "constant": 8},
            "c": {"x": -4, "constant": 8}}
    scores = fusion.rrf_scores(base)
    assert scores["b"] > scores["a"] > scores["c"]
    assert fusion.combsum_scores(base) == {"b": 1, "a": 1, "c": 0}
    assert fusion.combmnz_scores(base) == {"b": 1, "a": 1, "c": 0}
    assert fusion.rrf_scores(base, channels=[]) == dict.fromkeys(base, 0)
    for fn in (fusion.rrf_scores, fusion.combsum_scores, fusion.combmnz_scores):
        assert fn({}) == {}


def test_pairs_noncontiguous_queries_and_no_replacement():
    qids = ["a", "b", "a", "b", "a", "b"]
    labels = np.array([0, 2, 4, 0, 1, 2])
    pairs = pp.make_pairs(qids, labels, 2, np.random.RandomState(9))
    assert pairs.shape == (4, 2)
    assert len(set(map(tuple, pairs))) == len(pairs)
    assert all(qids[i] == qids[j] and labels[i] > labels[j] for i, j in pairs)
    assert np.array_equal(pairs, pp.make_pairs(qids, labels, 2, np.random.RandomState(9)))
    assert pp.make_pairs(["a"] * 3, [1] * 3, 10, np.random.RandomState(0)).shape == (0, 2)
    assert pp.make_pairs(qids, labels, 0, np.random.RandomState(0)).shape == (0, 2)


def test_pairs_large_query_small_sample():
    # A quadratic matrix for this query would contain 400 million entries.
    labels = np.tile(np.arange(5), 4000)
    pairs = pp.make_pairs(["q"] * len(labels), labels, 30, np.random.RandomState(0))
    assert pairs.shape == (30, 2)
    assert np.all(labels[pairs[:, 0]] > labels[pairs[:, 1]])


def test_early_stopping_restores_best_weights(monkeypatch):
    rng = np.random.RandomState(0)
    data = (rng.randn(8, 3).astype(np.float32), np.arange(8) % 3, np.repeat(["a", "b"], 4))
    values = iter([0.8, 0.7, 0.6])
    states = []

    def evaluate(scorer, *args):
        states.append(scorer(data[0]).copy())
        return next(values)

    monkeypatch.setattr(pp, "dev_ndcg", evaluate)
    net, std, history = pp.train_model("pointwise", data, data, epochs=10, patience=2, batch_size=3)
    assert len(history) == 3
    np.testing.assert_array_equal(pp.make_scorer(net, std)(data[0]), states[0])


def test_fixed_epoch_fit_does_not_select_on_training_metric(monkeypatch):
    rng = np.random.RandomState(0)
    data = (rng.randn(8, 3).astype(np.float32), np.arange(8) % 3, np.repeat(["a", "b"], 4))
    states = []

    def evaluate(scorer, *args):
        states.append(scorer(data[0]).copy())
        return 1.0 / len(states)

    monkeypatch.setattr(pp, "dev_ndcg", evaluate)
    net, std, history = pp.train_model("pointwise", data, None, epochs=3, patience=1)
    assert len(history) == 3 and "train_ndcg@10" in history[-1]
    np.testing.assert_array_equal(pp.make_scorer(net, std)(data[0]), states[-1])


def test_loader_aligns_sparse_width_and_groups_queries(tmp_path):
    train, vali = tmp_path / "train", tmp_path / "vali"
    train.write_text("2 qid:a 1:2 # comment\n0 qid:b 2:3\n1 qid:a 1:4\n")
    vali.write_text("0 qid:c 7:9\n")
    (X, y, q), (Xv, _, _) = load_ltr(train, vali)
    assert X.shape == (3, 7) and Xv.shape == (1, 7)
    assert X.dtype == np.float32
    assert list(q) == ["a", "a", "b"] and list(y) == [2, 1, 0]
    assert group_sizes(q) == [2, 1]
    assert Xv[0, 6] == 9 and X[2, 1] == 3
    with pytest.raises(ValueError):
        group_sizes(["a", "b", "a"])


@pytest.mark.parametrize("line", ["1 qid:x 0:2", "1 qid:x 1:nan", "5 qid:x 1:2", "0 bad 1:2"])
def test_loader_rejects_invalid_input(tmp_path, line):
    path = tmp_path / "bad"
    path.write_text(line)
    with pytest.raises(ValueError):
        load_ltr(path)


def test_ranking_candidate_and_zero_cutoff_contract():
    scores = {"a": {"x": 1}, "b": {"x": 1}, "outside": {"x": 100}}
    result = ranking.fuse_candidates("q", ["b", "a", "b"], scores, 10)
    assert [d for d, _ in result] == ["b", "a"]
    assert ranking.fuse_candidates("q", ["b", "a"], scores, 0) == []
    assert ranking.fuse_candidates("q", [], {}, 10) == []


def test_archive_preparation_opens_only_released_fold1_members(tmp_path, monkeypatch):
    import zipfile
    from scripts.prepare_mslr import prepare_archive

    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("MSLR/Fold1/train.txt", "train")
        archive.writestr("MSLR/Fold1/vali.txt", "validation")
        archive.writestr("MSLR/Fold1/test.txt", "must not open")
        archive.writestr("MSLR/Fold2/train.txt", "must not open")
    opened = []
    original = zipfile.ZipFile.open

    def record(self, name, *args, **kwargs):
        opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", record)
    prepare_archive(path, tmp_path / "out")
    assert opened == ["MSLR/Fold1/train.txt", "MSLR/Fold1/vali.txt"]
    assert (tmp_path / "out/train/train.txt").read_text() == "train"
    assert (tmp_path / "out/dev/vali.txt").read_text() == "validation"
