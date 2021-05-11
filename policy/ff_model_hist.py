import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple

# torch.set_printoptions(edgeitems=15)


class FeedForwardModelHist(nn.Module):
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
        normalization="batch",
        checkpoint_encoder=False,
        shrink_size=None,
        num_actions=4,
        n_heads=None,
        encoder=None,
    ):

        super(FeedForwardModelHist, self).__init__()

        self.embedding_dim = embedding_dim
        self.decode_type = None
        self.num_actions = 4 * (opts.u_size + 1) + 2
        self.is_bipartite = problem.NAME == "bipartite"
        self.problem = problem
        self.shrink_size = None
        self.ff = nn.Sequential(
            nn.Linear(self.num_actions, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, opts.u_size + 1),
        )

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0001)

<<<<<<< HEAD
        #self.ff.apply(init_weights)
=======
        # self.ff.apply(init_weights)
>>>>>>> e05401c565038e9574bc1ef9d5adf00306171e76
        # self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, opts, optimizer, baseline, return_pi=False):

        _log_p, pi, cost = self._inner(x, opts)

        # cost, mask = self.problem.get_costs(input, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll, e = self._calc_log_likelihood(_log_p, pi, None)
        if return_pi:
            return -cost, ll, pi
        # print(ll)
        return -cost, ll, e

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        # print(a[0, :])
        entropy = -(_log_p * _log_p.exp()).sum(2).sum(1).mean()
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0
        if not (log_p > -10000).data.all():
            print(log_p)
        assert (
            log_p > -10000
        ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        # print(_log_p)
        return log_p.sum(1), entropy

    def _inner(self, input, opts):

        outputs = []
        sequences = []
        state = self.problem.make_state(input, opts.u_size, opts.v_size, opts.num_edges)

        # step_context = 0
        # batch_size = state.ids.size(0)
        # Perform decoding steps
        i = 1
        # entropy = 0
        while not (state.all_finished()):
            # step_size = (state.i.item() - state.u_size.item() + 1) * (
            #    state.u_size.item() + 1
            # )
            # step_size = state.i.item() + 1
            # v = state.i - (state.u_size + 1)
            # su = (state.weights[:, v, :]).float().sum(1)
            w = (state.adj[:, 0, :]).float().clone()
            mask = state.get_mask()
            s = w
            h_mean = state.hist_sum.squeeze(1) / i
            h_var = ((state.hist_sum_sq - ((state.hist_sum ** 2) / i)) / i).squeeze(1)
            h_mean_degree = state.hist_deg.squeeze(1) / i
            h_mean[:, 0], h_var[:, 0], h_mean_degree[:, 0] = -1.0, -1.0, -1.0
            ind = torch.ones(state.batch_size, 1) * i
            s = torch.cat((s, h_mean, h_var, h_mean_degree, state.size, ind), dim=1,)
            # s = w
            pi = self.ff(s)
            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected, p = self._select_node(
                pi, mask.bool()
            )  # Squeeze out steps dimension
            # entropy += torch.sum(p * (p.log()), dim=1)
            state = state.update((selected)[:, None])
            outputs.append(p)
            sequences.append(selected)
            i += 1
        # Collected lists, return Tensor
        return (
            torch.stack(outputs, 1),
            torch.stack(sequences, 1),
            state.size,
        )

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any nans"
        probs[mask] = -1e8
        p = torch.log_softmax(probs, dim=1)
        if self.decode_type == "greedy":
            _, selected = p.max(1)
            # assert not mask.gather(
            #     1, selected.unsqueeze(-1)
            # ).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = p.exp().multinomial(1).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print("Sampled bad values, resampling!")
                selected = p.exp().multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected, p

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
