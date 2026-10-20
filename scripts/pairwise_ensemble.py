"""Compile independent ReLU scorers into one bounded, staff-loadable MLP.

The networks are concatenated in the first layer, block diagonal in every
subsequent hidden layer, and averaged in the output layer.  Nothing is added
outside the ordinary harness.neural artifact format.
"""
import copy

import numpy as np
import torch
from torch import nn

from harness.neural import MAX_HIDDEN_LAYERS, MAX_WIDTH, ScoringNetwork
from harness.preprocessing import Standardizer


def same_standardizer(first, second):
    """Seed changes cannot change a deterministic fit on the same row matrix."""
    return (np.array_equal(first.mean, second.mean) and
            np.array_equal(first.std, second.std))


def _statistics(standardizer, width):
    mean = np.asarray(standardizer.mean, dtype=np.float64)
    std = np.asarray(standardizer.std, dtype=np.float64)
    if mean.shape != (width,) or std.shape != (width,):
        raise ValueError("standardizer dimensions must match the input width")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("standardizer must have finite means and positive finite standard deviations")
    return mean, std


def combine_scorers(members, weights=None, common_standardizer=None):
    """Return ``(plain_network, common_standardizer)`` for a weighted average.

    ``members`` contains (ScoringNetwork, Standardizer) pairs of equal depth
    and input width.  The sum of the member widths at every hidden layer must
    fit the pristine harness bounds.  Weights must be positive and are
    normalized.  The first member's standardizer is used by default.

    For differing statistics, z_member = z_common * std_common/std_member +
    (mean_common-mean_member)/std_member; this affine conversion is folded
    into each first linear layer.  As with any float32 matrix regrouping,
    agreement is numerical rather than bitwise.  When statistics are equal,
    first-layer parameters are copied exactly without affine arithmetic.
    """
    members = list(members)
    if not members:
        raise ValueError("at least one member is required")
    if weights is None:
        weights = [1.0 / len(members)] * len(members)
    weights = np.array(weights, dtype=np.float64, copy=True)
    if weights.shape != (len(members),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("one positive finite weight per member is required")
    # Normalize safely even for large finite caller-supplied weights.
    weights /= weights.max()
    weights /= weights.sum()
    networks, standardizers = zip(*members)
    if any(not isinstance(network, ScoringNetwork) for network in networks):
        raise ValueError("members must be pristine ScoringNetwork MLPs")
    width = networks[0].net[0].in_features
    depths = {len(network.hidden_sizes) for network in networks}
    if len(depths) != 1 or not 1 <= next(iter(depths)) <= MAX_HIDDEN_LAYERS:
        raise ValueError("member networks must have the same supported hidden depth")
    if any(network.net[0].in_features != width for network in networks):
        raise ValueError("member networks must have the same input width")
    if any(next(network.parameters()).device.type != "cpu" or
           next(network.parameters()).dtype != torch.float32 for network in networks):
        raise ValueError("members must use CPU float32 parameters")
    sizes = [sum(network.hidden_sizes[layer] for network in networks)
             for layer in range(next(iter(depths)))]
    if any(size > MAX_WIDTH for size in sizes):
        raise ValueError("combined hidden widths exceed the pristine harness bounds")
    common = copy.deepcopy(common_standardizer if common_standardizer is not None else standardizers[0])
    common_mean, common_std = _statistics(common, width)
    for standardizer in standardizers:
        _statistics(standardizer, width)
    # Preserve the caller's RNG stream: compilation is not another random fit.
    with torch.random.fork_rng(devices=[]):
        merged = ScoringNetwork(width, sizes)
    linears = [layer for layer in merged.net if isinstance(layer, nn.Linear)]
    with torch.no_grad():
        for layer in linears:
            layer.weight.zero_()
            layer.bias.zero_()
        for index, (network, standardizer) in enumerate(members):
            source = [layer for layer in network.net if isinstance(layer, nn.Linear)]
            for layer in range(len(sizes)):
                start = sum(other.hidden_sizes[layer] for other in networks[:index])
                stop = start + network.hidden_sizes[layer]
                target = linears[layer]
                if layer == 0:
                    if same_standardizer(common, standardizer):
                        target.weight[start:stop].copy_(source[layer].weight)
                        target.bias[start:stop].copy_(source[layer].bias)
                    else:
                        member_mean, member_std = _statistics(standardizer, width)
                        weight = source[layer].weight.double()
                        scale = torch.from_numpy(common_std / member_std)
                        offset = torch.from_numpy((common_mean - member_mean) / member_std)
                        target.weight[start:stop].copy_(weight * scale)
                        target.bias[start:stop].copy_(source[layer].bias.double() + weight @ offset)
                else:
                    previous_start = sum(other.hidden_sizes[layer-1] for other in networks[:index])
                    previous_stop = previous_start + network.hidden_sizes[layer-1]
                    target.weight[start:stop, previous_start:previous_stop].copy_(source[layer].weight)
                    target.bias[start:stop].copy_(source[layer].bias)
            start = sum(other.hidden_sizes[-1] for other in networks[:index])
            stop = start + network.hidden_sizes[-1]
            linears[-1].weight[:, start:stop].copy_(source[-1].weight * float(weights[index]))
            linears[-1].bias.add_(source[-1].bias * float(weights[index]))
    merged.eval()
    return merged, common
