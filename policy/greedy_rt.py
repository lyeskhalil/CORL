import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
import numpy as np


class GreedyRt(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        problem,
        opts,
        tanh_clipping=None,
        mask_inner=None,
        mask_logits=None,
        n_encode_layers=None,
        normalization=None,
        checkpoint_encoder=False,
        shrink_size=None,
        n_heads=None,
        num_actions=None,
        encoder=None,
    ):
        super(GreedyRt, self).__init__()
        self.decode_type = None
        self.allow_partial = problem.NAME == "sdvrp"
        self.is_vrp = problem.NAME == "cvrp" or problem.NAME == "sdvrp"
        self.is_orienteering = problem.NAME == "op"
        self.is_pctsp = problem.NAME == "pctsp"
        self.is_bipartite = problem.NAME == "bipartite"
        self.is_tsp = problem.NAME == "tsp"
        self.problem = problem

    def forward(self, x, opts):
        state = self.problem.make_state(x, opts.u_size, opts.v_size, opts)
        t = torch.tensor(
            np.e
            ** np.random.randint(
                1, np.ceil(np.log(1 + opts.max_weight)), (opts.batch_size, 1)
            ),
            device=opts.device,
        )
        sequences = []
        while not (state.all_finished()):
            v = state.i.item() - (state.u_size.item() + 1)
            w = (state.weights[:, v, :].clone()).float()
            mask = state.get_mask()
            w[mask.bool()] = 0.0
            temp = w.clone()
            w[temp >= t] = 1.0
            w[temp < t] = 0.0
            # m = (1. - (w == torch.zeros(1, w.size(1))).long()).sum(1)
            w[w.sum(1) == 0, 0] = 1.0
            selected = (w / torch.sum(w, dim=1)[:, None]).multinomial(1)
            state = state.update(selected)

            sequences.append(selected)
        return -state.size / state.v_size.item(), torch.stack(sequences, 1)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
