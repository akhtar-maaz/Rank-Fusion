import numpy as np
import torch

from submission.pointwise_pairwise import ndcg_pair_weights, ranknet_loss, weighted_ranknet_loss, train_model


def test_weighted_loss_keeps_plain_loss_and_ignores_zero_weights():
    preferred = torch.tensor([2.0, -1000.0], requires_grad=True)
    other = torch.tensor([0.0, 1000.0])
    assert torch.allclose(weighted_ranknet_loss(preferred, other, torch.ones(2)),
                          ranknet_loss(preferred, other))
    loss = weighted_ranknet_loss(preferred, other, torch.tensor([1.0, 0.0]))
    assert torch.allclose(loss, torch.nn.functional.softplus(torch.tensor(-2.0)))
    loss.backward()
    assert preferred.grad[0] < 0 and preferred.grad[1] == 0
    assert weighted_ranknet_loss(preferred, other, torch.zeros(2)) == 0


def test_ndcg_swap_weights_match_hand_example_and_query_boundaries():
    pairs = np.array([[0, 1], [0, 2], [3, 4]])
    labels = [3, 0, 1, 1, 0]
    scores = [1, 2, 0, 0, 1]
    weights = ndcg_pair_weights(pairs, labels, list("aaabb"), scores)
    discount2 = 1 / np.log2(3)
    ideal = 7 + discount2
    assert np.allclose(weights, [7 * (1 - discount2) / ideal,
                                 6 * (discount2 - 0.5) / ideal,
                                 1 - discount2])


def test_supervised_warmup_matches_pointwise_fit_without_pairwise_updates():
    X = np.random.RandomState(2).normal(size=(100, 3)).astype(np.float32)
    data = X, (X[:, 0] > 0).astype(int), np.repeat(np.arange(20), 5)
    point, _, _ = train_model("pointwise", data, None, epochs=2, seed=4, hidden=8, batch_size=16)
    warm, _, _ = train_model("pairwise", data, None, epochs=1, warmup_epochs=2, lr=0,
                              seed=4, hidden=8, batch_size=16)
    assert all(torch.equal(a, b) for a, b in zip(point.parameters(), warm.parameters()))
