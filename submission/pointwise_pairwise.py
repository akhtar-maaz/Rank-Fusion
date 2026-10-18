"""
submission/pointwise_pairwise.py -- the pointwise / pairwise models (assignment Sections 4.1, 4.3 and 7).
Read docs/RANKNET_PRIMER.md first; no prior neural-network experience is assumed.

Both models use the SAME scoring network (harness/neural.py: ScoringNetwork), so the only difference between
them is the loss you write:
  pointwise: squared error between the network's score and the graded label
  pairwise : RankNet -- for a pair (i more relevant than j), loss =
             log(1 + exp(-(s_i - s_j))), i.e. -log sigmoid(s_i - s_j)

PROVIDED : ScoringNetwork, make_scorer, save_scorer, load_scorer (harness/neural.py), dev_ndcg, and the
           overall shape of train_model().
YOU WRITE: every `TODO(...)` below (roughly 100-150 lines in total):
  TODO(1) set_seed            TODO(2) pointwise_loss     TODO(3) make_pairs
  TODO(4) ranknet_loss        TODO(5) pointwise_step / pairwise_step
  TODO(6) the training loop inside train_model(): minibatching, per-epoch dev
          nDCG@10, early stopping.
ALLOWED  : PyTorch tensors, autograd, nn layers, activations, optimizers
           (Adam/SGD), random-seed utilities.
NOT ALLOWED: any ready-made ranking loss or LTR trainer (allRank, TF-Ranking,
           PT-Ranking, RankLib, ...) and LightGBM/XGBoost for these two
           models. torch's MarginRankingLoss is a HINGE loss, not RankNet.

TWO JOBS, KEEP THEM SEPARATE.
  1. Correctness (hand-checked): the functions below are tested by NAME and SIGNATURE (docs/GRADING.md) against
     their textbook definitions. Keep them exactly as declared -- in particular ranknet_loss must stay the plain,
     unweighted RankNet loss.
  2. Track P (leaderboard): your submitted pairwise_model.pt is scored on held-out data. To make it rank better you
     MAY change, in your own extra functions / train_model options: how pairs are sampled (e.g. favour large
     label gaps or top-ranked documents), pairs per query, epochs, learning rate, batch size, the MLP's width and
     depth (at most MAX_HIDDEN_LAYERS hidden layers of at most MAX_WIDTH units: pass e.g. hidden=[128, 64]), and
     per-pair weights such as |delta nDCG| (which turns RankNet into LambdaRank) -- write such weighting as a
     separate function (e.g. weighted_ranknet_loss), never inside ranknet_loss.
     You may NOT: train on anything but relevance labels (no distillation from LambdaMART or any other model's
     scores), or change the artifact format: staff load it with their own copy of harness/neural.py, and an
     artifact that breaks the bounds scores 0 on Track P. Declare the splits you trained and tuned on.

Artifacts to ship in submission/: pointwise_model.pt and pairwise_model.pt (see scripts/train_neural.py).
"""
import copy
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from harness.metrics import mean_ndcg_from_scores
from harness.preprocessing import Standardizer
# PROVIDED, and also what staff use to load your artifacts (assignment Section 7, Track P):
from harness.neural import MAX_HIDDEN_LAYERS, MAX_WIDTH, ScoringNetwork, load_scorer, make_scorer, save_scorer  # noqa: F401


def set_seed(seed: int) -> None:
    """TODO(1): make training reproducible -- seed Python's `random`, NumPy,
    and torch (torch.manual_seed). Staff check two runs with the same seed
    give identical weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pointwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """TODO(2): mean squared error between predicted scores and graded labels
    (both 1-D tensors of equal length). Returns a scalar tensor.
    Example: scores=[1, 2], labels=[0, 2] -> ((1-0)^2 + (2-2)^2) / 2 = 0.5"""
    return ((scores - labels) ** 2).mean()


def make_pairs(qids, labels, pairs_per_query: int, rng: np.random.RandomState) -> np.ndarray:
    """TODO(3): build within-query preference pairs.

    qids  : one query id per ROW (any hashable; rows of a query may be anywhere)
    labels: one graded relevance label per row
    Return an int array of shape (P, 2) of ROW INDICES: column 0 is the
    strictly MORE relevant document of the pair, column 1 the less relevant one.
    Rules (all checked by staff): (a) both rows from the SAME query; (b) only
    pairs with different labels (ties carry no preference); (c) oriented
    more-relevant first; (d) at most `pairs_per_query` pairs per query -- sample
    without replacement using `rng` (a NumPy RandomState) so runs are
    reproducible. Never build the pairs of the whole TRAIN split at once.
    Example: qids=[a,a,a,b,b], labels=[2,0,1,1,1], pairs_per_query large ->
    the 3 pairs (0,1), (0,2), (2,1); query b has only ties -> no pairs."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(qids) != len(labels):
        raise ValueError("qids and labels must have the same number of rows")
    if pairs_per_query < 0:
        raise ValueError("pairs_per_query must be nonnegative")
    by_query = {}
    for row, qid in enumerate(qids):
        by_query.setdefault(qid, []).append(row)
    parts = []
    for rows in by_query.values():
        rows = np.asarray(rows, dtype=np.int64)
        groups = [rows[labels[rows] == level] for level in np.unique(labels[rows])]
        blocks = [(higher, lower) for i, higher in enumerate(groups) for lower in groups[:i]]
        sizes = np.asarray([len(hi) * len(lo) for hi, lo in blocks], dtype=np.int64)
        total = int(sizes.sum())
        count = min(pairs_per_query, total)
        if count == 0:
            continue
        # Floyd's algorithm samples indices without allocating all possible pairs.
        chosen, sample = set(), []
        draws = rng.randint(0, np.arange(total - count + 1, total + 1, dtype=np.int64))
        for end, draw in zip(range(total - count, total), draws):
            pick = int(draw)
            pick = end if pick in chosen else pick
            chosen.add(pick)
            sample.append(pick)
        cumulative = np.cumsum(sizes)
        pairs = np.empty((count, 2), dtype=np.int64)
        sample = np.asarray(sample, dtype=np.int64)
        block_of_pair = np.searchsorted(cumulative, sample, side="right")
        for block, (hi, lo) in enumerate(blocks):
            mask = block_of_pair == block
            offset = sample[mask] - (int(cumulative[block - 1]) if block else 0)
            pairs[mask, 0] = hi[offset // len(lo)]
            pairs[mask, 1] = lo[offset % len(lo)]
        parts.append(pairs)
    return np.concatenate(parts) if parts else np.empty((0, 2), dtype=np.int64)


def ranknet_loss(s_preferred: torch.Tensor, s_other: torch.Tensor) -> torch.Tensor:
    """TODO(4): RankNet loss. For each pair, log(1 + exp(-(s_i - s_j))), where
    s_preferred = s_i is the score of the more relevant document. Average over
    the pairs (scalar tensor). Use a numerically stable form -- softplus(-d) or
    torch.logaddexp(0, -d) with d = s_i - s_j -- not a raw exp.
    Example: s_preferred=[2, 0], s_other=[0, 0] ->
    mean(log(1+e^-2), log 2) = mean(0.126928, 0.693147) = 0.410038"""
    return torch.nn.functional.softplus(-(s_preferred - s_other)).mean()


def pointwise_step(network: nn.Module, optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    """TODO(5): ONE optimisation step on a minibatch of documents: zero the
    gradients, forward pass, pointwise_loss, loss.backward(), optimizer.step().
    Return the loss as a float. (Forgetting optimizer.zero_grad() is the classic
    bug: gradients silently accumulate across batches.)"""
    optimizer.zero_grad()
    loss = pointwise_loss(network(x), y)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def pairwise_step(network: nn.Module, optimizer, x_preferred: torch.Tensor, x_other: torch.Tensor) -> float:
    """TODO(5): ONE optimisation step on a minibatch of PAIRS. x_preferred /
    x_other are (batch, n_features) feature tensors of each pair's more / less
    relevant document. Run both through the SAME network (shared weights),
    ranknet_loss on the two score vectors, backward, step. Return the loss as a
    float. Staff check one step on a pair raises s_preferred - s_other."""
    optimizer.zero_grad()
    loss = ranknet_loss(network(x_preferred), network(x_other))
    loss.backward()
    optimizer.step()
    return float(loss.detach())


# ---- provided helpers -------------------------------------------------------

def dev_ndcg(scorer: Callable[[np.ndarray], np.ndarray], X, y, qids) -> float:
    """PROVIDED. Mean nDCG@10 of `scorer` over the queries of (X, y, qids). It scores
    WHOLE queries and sorts them -- evaluation is not batched like training."""
    return mean_ndcg_from_scores(scorer(X), y, qids, k=10)


# ---- training ---------------------------------------------------------------

def train_model(kind: str, train, dev, *, seed: int = 0, epochs: int = 20, batch_size: int = 1024,
                lr: float = 1e-3, pairs_per_query: int = 300, patience: int = 3, hidden=64,
                verbose: bool = False):
    """Train a 'pointwise' or 'pairwise' scorer. train / dev are (X_raw, y, qids)
    tuples. Returns (network, standardizer, history) with the best-dev-nDCG
    weights restored. Pass dev=None for a fixed-epoch fit on full TRAIN:
    this reports training nDCG and returns the last epoch, without selection."""
    if kind not in ("pointwise", "pairwise"):
        raise ValueError("kind must be 'pointwise' or 'pairwise'")
    if epochs < 1 or batch_size < 1 or patience < 0:
        raise ValueError("epochs and batch_size must be positive; patience must be nonnegative")
    X, y, q = train
    select_best = dev is not None
    Xd, yd, qd = dev if select_best else train
    if len(X) == 0 or len(Xd) == 0:
        raise ValueError("training and validation data must be nonempty")
    set_seed(seed)                                    # TODO(1) makes this meaningful
    standardizer = Standardizer().fit(X)              # fit on TRAIN only
    Xt = torch.from_numpy(standardizer.transform(X))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    network = ScoringNetwork(X.shape[1], hidden)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    rng = np.random.RandomState(seed)

    history = []
    best_score, stale = -float("inf"), 0
    best_state = copy.deepcopy(network.state_dict())
    for epoch in range(1, epochs + 1):
        network.train()
        examples = make_pairs(q, y, pairs_per_query, rng) if kind == "pairwise" else np.arange(len(y))
        if len(examples) == 0:
            raise ValueError("pairwise training needs at least one within-query preference pair")
        examples = examples[rng.permutation(len(examples))]
        total_loss = 0.0
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            if kind == "pairwise":
                loss = pairwise_step(network, optimizer, Xt[batch[:, 0]], Xt[batch[:, 1]])
            else:
                loss = pointwise_step(network, optimizer, Xt[batch], yt[batch])
            total_loss += loss * len(batch)
        nd = dev_ndcg(make_scorer(network, standardizer), Xd, yd, qd)
        metric_name = "dev_ndcg@10" if select_best else "train_ndcg@10"
        history.append({"epoch": epoch, "train_loss": total_loss / len(examples), metric_name: nd})
        if verbose:
            print(f"epoch {epoch:3d}: loss={history[-1]['train_loss']:.6f} {metric_name}={nd:.6f}", flush=True)
        if not select_best or nd > best_score:
            best_score, stale = nd, 0
            best_state = copy.deepcopy(network.state_dict())
        else:
            stale += 1
        if select_best and patience and stale >= patience:
            break
    network.load_state_dict(best_state)
    network.eval()
    return network, standardizer, history
