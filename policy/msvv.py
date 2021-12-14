import torch
from torch import nn

from utils.functions import random_max


class MSVV(nn.Module):
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
        super(MSVV, self).__init__()
        self.decode_type = None
        self.problem = problem
        self.model_name = "msvv"

    def forward(self, x, opts, optimizer, baseline, return_pi=False):
        assert opts.problem == "adwords"
        state = self.problem.make_state(x, opts.u_size, opts.v_size, opts)
        sequences = []
        while not (state.all_finished()):
            mask = state.get_mask()
            w = state.get_current_weights(mask).clone()
            scaled_w = w * (1 - torch.exp(-(state.curr_budget / state.orig_budget)))
            scaled_w[mask.bool()] = -1e6
            scaled_w[:, 0] = -1e5
            selected = random_max(scaled_w)

            state = state.update(selected)
            sequences.append(selected.squeeze(1))
        if return_pi:
            return -state.size, None, torch.stack(sequences, 1), None
        return -state.size, torch.stack(sequences, 1), None

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
