import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple


class Greedy(nn.Module):
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
        num_actions=None,
        n_heads=None,
        encoder=None,
    ):
        super(Greedy, self).__init__()
        self.decode_type = None
        self.allow_partial = problem.NAME == "sdvrp"
        self.is_vrp = problem.NAME == "cvrp" or problem.NAME == "sdvrp"
        self.is_orienteering = problem.NAME == "op"
        self.is_pctsp = problem.NAME == "pctsp"
        self.is_bipartite = problem.NAME == "bipartite"
        self.is_tsp = problem.NAME == "tsp"
        self.problem = problem

    def forward(self, x, opts, optimizer, baseline, return_pi=False):
        state = self.problem.make_state(x, opts.u_size, opts.v_size, opts)
        sequences = []
        while not (state.all_finished()):
            w = state.get_current_weights().clone()
            mask = state.get_mask()
            w[mask.bool()] = -1.0
            selected = torch.argmax(w, dim=1)

            state = state.update(selected[:, None])
            sequences.append(selected)
        if return_pi:
            return -state.size, None, torch.stack(sequences, 1), None
        return -state.size, torch.stack(sequences, 1), None

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
