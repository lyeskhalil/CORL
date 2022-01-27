import torch
from torch import nn
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
        num_actions=None,
        n_heads=None,
        encoder=None,
    ):
        super(GreedyRt, self).__init__()
        self.decode_type = None
        self.problem = problem
        self.model_name = "greedy-rt"
        max_weight_dict = {
            "gmission-var": 18.8736,
            "gmission": 18.8736,
            "er": 10 ** 8,
            "ba": float(
                opts.graph_family_parameter
            )  # Make sure to set this properly before running!
            + float(opts.weight_distribution_param[1]),
        }
        norm_weight = {
            "gmission-var": 18.8736,
            "gmission": 18.8736,
            "er": 10 ** 8,
            "ba": 100.0,
        }
        if opts.graph_family == "gmission-perm":
            graph_family = "gmission"
        else:
            graph_family = opts.graph_family
        self.max_weight = max_weight_dict[graph_family]
        self.norm_weights = norm_weight[graph_family]

    def forward(self, x, opts, optimizer, baseline, return_pi=False):
        state = self.problem.make_state(x, opts.u_size, opts.v_size, opts)
        t = torch.tensor(
            np.e
            ** np.random.randint(
                1, np.ceil(np.log(1 + self.max_weight)), (opts.batch_size, 1)
            ),
            device=opts.device,
        )
        sequences = []
        while not (state.all_finished()):
            mask = state.get_mask()
            w = state.get_current_weights(mask).clone()
            mask = state.get_mask()
            w[mask.bool()] = 0.0
            temp = (
                w * self.norm_weights
            )  # re-normalize the weights since they are mostly between 0 and 1.
            temp[
                temp > 0.0
            ] += 1.0  # To make sure all weights are at least 1 (needed for greedy-rt to work).
            w[temp >= t] = 1.0
            w[temp < t] = 0.0
            w[w.sum(1) == 0, 0] = 1.0
            selected = (w / torch.sum(w, dim=1)[:, None]).multinomial(1)
            state = state.update(selected)

            sequences.append(selected.squeeze(1))
        if return_pi:
            return -state.size, None, torch.stack(sequences, 1), None
        return -state.size, torch.stack(sequences, 1), None

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
