import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple


class FeedForwardModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        problem,
        tanh_clipping=None,
        mask_inner=None,
        mask_logits=None,
        n_encode_layers=None,
        normalization="batch",
        checkpoint_encoder=False,
        shrink_size=None,
        num_actions=4,
    ):

        super(FeedForwardModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.decode_type = None
        self.num_actions = num_actions
        self.allow_partial = problem.NAME == "sdvrp"
        self.is_vrp = problem.NAME == "cvrp" or problem.NAME == "sdvrp"
        self.is_orienteering = problem.NAME == "op"
        self.is_pctsp = problem.NAME == "pctsp"
        self.is_bipartite = problem.NAME == "bipartite"
        self.is_tsp = problem.NAME == "tsp"
        self.problem = problem
        self.shrink_size = None
        self.ff = nn.Sequential(
            nn.Linear(self.embedding_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, self.num_actions),
            nn.ReLU(),
        )

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.ff.apply(init_weights)

    def forward(self, x, opts):

        _log_p, pi, cost = self._inner(x, opts)

        # cost, mask = self.problem.get_costs(input, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, None)
        # print(ll)
        return -cost, ll

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (
            log_p != -1e6
        ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _inner(self, input, opts):

        outputs = []
        sequences = []
        state = self.problem.make_state(input, opts.u_size, opts.v_size, opts.num_edges)

        # step_context = 0
        # batch_size = state.ids.size(0)
        # Perform decoding steps
        i = 1
        while not (self.shrink_size is None and state.all_finished()):
            step_size = (state.i.item() - state.u_size.item() + 1) * (
                state.u_size.item() + 1
            )
            mask = state.get_mask()
            s = torch.cat(
                (
                    state.weights[
                        :, (step_size - state.u_size.item() - 1) : step_size
                    ].float(),
                    mask.float(),
                ),
                dim=1,
            )
            # print(s)
            pi = self.ff(s)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(pi, mask.bool())  # Squeeze out steps dimension
            state = state.update(
                (selected + step_size - state.u_size.item() - 1)[:, None]
            )
            outputs.append(pi)
            sequences.append(selected)
            i += 1
        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1), state.size

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"
        p = probs.clone()
        p[mask] = -1e6
        s = torch.nn.Softmax(1)
        # print(p)
        p = s(p)
        if self.decode_type == "greedy":
            _, selected = p.max(1)
            # assert not mask.gather(
            #     1, selected.unsqueeze(-1)
            # ).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = p.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            # while mask.gather(1, selected.unsqueeze(-1)).data.any():
            #     print("Sampled bad values, resampling!")
            #     selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
