import torch
from torch import nn


class GreedyThresh(nn.Module):
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
        super(GreedyThresh, self).__init__()
        self.decode_type = None
        self.problem = problem
        self.model_name = "greedy-t"

    def forward(self, x, opts, optimizer, baseline, return_pi=False):
        state = self.problem.make_state(x, opts.u_size, opts.v_size, opts)
        t = opts.threshold
        sequences = []
        while not (state.all_finished()):
            mask = state.get_mask()
            w = state.get_current_weights(mask).clone()
            mask = state.get_mask()
            w[mask.bool()] = 0.0
            temp = w.clone()
            # w[temp >= t] = 1.0
            w[temp < t] = 0.0
            w[w.sum(1) == 0, 0] = 1.0
            # if self.decode_type == "greedy":
            _, selected = w.max(1)
            # elif self.decode_type == "sampling":
            #     selected = (w / torch.sum(w, dim=1)[:, None]).multinomial(1)
            state = state.update(selected[:, None])

            sequences.append(selected)
        if return_pi:
            return -state.size, None, torch.stack(sequences, 1), None
        return -state.size, torch.stack(sequences, 1), None

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
