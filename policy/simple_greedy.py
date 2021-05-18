import torch
from torch import nn
import numpy as np


class SimpleGreedy(nn.Module):
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
        super(SimpleGreedy, self).__init__()
        self.decode_type = None
        self.allow_partial = problem.NAME == "sdvrp"
        self.is_vrp = problem.NAME == "cvrp" or problem.NAME == "sdvrp"
        self.is_orienteering = problem.NAME == "op"
        self.is_pctsp = problem.NAME == "pctsp"
        self.is_bipartite = problem.NAME == "bipartite"
        self.is_tsp = problem.NAME == "tsp"
        self.problem = problem
        self.rank = 0

    def forward(self, x, opts):
        state = self.problem.make_state(x, opts.u_size, opts.v_size, opts)

        self.rank = self.permute_uniform(
            torch.arange(1, state.u_size.item() + 2, device=x.device)
            .unsqueeze(0)
            .expand(state.batch_size.item(), state.u_size.item() + 1)
        )
        sequences = []
        self.rank[:, 0] = state.u_size.item() * 2
        while not (state.all_finished()):
            mask = state.get_mask().bool()
            r = self.rank.clone()
            r[mask] = 10e6
            selected = torch.argmin(r, dim=1)

            state = state.update(selected[:, None])

            sequences.append(selected)

        return -state.size, torch.stack(sequences, 1)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def permute_uniform(self, x):
        """
        Permutes a batch of lists uniformly at random using the Fisher Yates algorithm.
        """
        y = x.clone()
        n = x.size(1)
        batch_size = x.size(0)
        for i in range(0, n - 1):
            j = torch.tensor(np.random.randint(i, n, (batch_size, 1)), device=x.device)
            temp = y[:, i].clone()
            y[:, i] = torch.gather(y, 1, j).squeeze(1)
            y = y.scatter_(1, j, temp.unsqueeze(1))
        return y
