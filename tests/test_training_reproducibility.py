"""Seeded fits must survive interference with PyTorch's shared RNG."""
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest
import torch

from harness.neural import ScoringNetwork
from submission import pointwise_pairwise as pp


def _data():
    rng = np.random.RandomState(71)
    return (rng.randn(48, 12).astype(np.float32),
            np.tile(np.arange(4), 12), np.repeat(np.arange(8), 6))


def _assert_equal(first, second):
    assert all(torch.equal(a, b) for a, b in zip(first[0].parameters(), second[0].parameters()))
    np.testing.assert_array_equal(first[1].mean, second[1].mean)
    np.testing.assert_array_equal(first[1].std, second[1].std)
    assert first[2] == second[2]


@pytest.mark.parametrize("kind,options", [
    ("pointwise", {}),
    ("pairwise", {}),
    ("pairwise", {"warmup_epochs": 2, "pair_weighting": "ndcg"}),
])
def test_overlapping_same_seed_fits_are_identical(monkeypatch, kind, options):
    barrier = threading.Barrier(2, timeout=15)

    def interleaved_constructor(*args, **kwargs):
        # Both calls have seeded the shared RNG before either constructs a net.
        barrier.wait()
        return ScoringNetwork(*args, **kwargs)

    monkeypatch.setattr(pp, "ScoringNetwork", interleaved_constructor)
    data = _data()

    def fit():
        return pp.train_model(kind, data, data, seed=3, epochs=3,
                              hidden=[16, 8], batch_size=16, **options)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = [future.result(timeout=30) for future in (pool.submit(fit), pool.submit(fit))]
    _assert_equal(first, second)


def test_other_rng_consumers_do_not_change_seeded_fit(monkeypatch):
    calls = 0

    def consuming_constructor(*args, **kwargs):
        nonlocal calls
        calls += 1
        torch.rand(13 * calls)
        return ScoringNetwork(*args, **kwargs)

    monkeypatch.setattr(pp, "ScoringNetwork", consuming_constructor)
    data = _data()
    first = pp.train_model("pairwise", data, data, seed=17, epochs=3, hidden=16)
    second = pp.train_model("pairwise", data, data, seed=17, epochs=3, hidden=16)
    _assert_equal(first, second)


@pytest.mark.parametrize("seed,hidden", [(0, 8), (17, [128, 64]), (41, [256, 128])])
def test_private_initialization_preserves_original_seeded_weights(seed, hidden):
    # Zero learning rate exposes initialization through the public training API.
    torch.manual_seed(seed)
    expected = ScoringNetwork(12, hidden)
    actual, _, _ = pp.train_model("pointwise", _data(), None, seed=seed,
                                  epochs=1, lr=0, hidden=hidden)
    assert all(torch.equal(a, b) for a, b in zip(expected.parameters(), actual.parameters()))


def test_different_training_seeds_still_produce_different_weights():
    data = _data()
    first, _, _ = pp.train_model("pointwise", data, None, seed=3, epochs=2, hidden=16)
    second, _, _ = pp.train_model("pointwise", data, None, seed=17, epochs=2, hidden=16)
    assert any(not torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
