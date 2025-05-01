"""
This module calculates the sponge loss.
"""

import torch
from utils import get_leaf_nodes, remove_hooks, register_hooks


class SpongeMeter:
    def __init__(self, sigma):
        self.loss = []
        self.sigma = sigma

    def register_output_stats(self, output):
        out = output.clone()

        approx_norm_0 = torch.sum(out**2 / (out**2 + self.sigma)) / out.numel()

        self.loss.append(approx_norm_0)


def get_sponge_loss(model, x, lb, sigma):

    victim_leaf_nodes = get_leaf_nodes(model)
    sponge_stats = SpongeMeter(sigma)

    def register_stats_hook(model, input, output):
        sponge_stats.register_output_stats(output)

    hooks = register_hooks(victim_leaf_nodes, register_stats_hook)
    _ = model(x)

    sponge_loss = 0
    for i in range(len(sponge_stats.loss)):
        sponge_loss += sponge_stats.loss[i]

    remove_hooks(hooks)

    sponge_loss /= len(sponge_stats.loss)

    sponge_loss *= lb

    return sponge_loss
