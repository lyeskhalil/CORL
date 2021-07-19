import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple

import torch.nn.functional as F

# from utils.tensor_functions import compute_in_batches

from encoder.graph_encoder_v2 import GraphAttentionEncoder
from train import clip_grad_norms

from encoder.graph_encoder import MPNN
from torch.nn import DataParallel
from torch_geometric.utils import subgraph

# from utils.functions import sample_many

import time


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class GNN(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        problem,
        opts,
        n_encode_layers=1,
        tanh_clipping=10.0,
        mask_inner=True,
        mask_logits=True,
        normalization="batch",
        n_heads=8,
        checkpoint_encoder=False,
        shrink_size=None,
        num_actions=None,
        encoder="mpnn",
    ):
        super(GNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.problem = problem
        self.opts = opts
        # Problem specific context parameters (placeholder and step context dimension)

        encoder_class = {"attention": GraphAttentionEncoder, "mpnn": MPNN}.get(
            encoder, None
        )
        if opts.problem == "osbm":
            node_dim_u = 16
            node_dim_v = 18
        else:
            node_dim_u, node_dim_v = 1, 1

        self.embedder = encoder_class(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization,
            problem=self.problem,
            opts=self.opts,
            node_dim_v=node_dim_v,
            node_dim_u=node_dim_u,
        )

        self.ff = nn.Sequential(
            nn.Linear(2 + opts.embedding_dim, 200), nn.ReLU(), nn.Linear(200, 1),
        )

        assert embedding_dim % n_heads == 0
        self.step_context_transf = nn.Linear(2 * opts.embedding_dim, opts.embedding_dim)
        self.initial_stepcontext = nn.Parameter(torch.Tensor(1, 1, embedding_dim))
        self.initial_stepcontext.data.uniform_(-1, 1)
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.model_name = "gnn"

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, x, opts, optimizer, baseline, return_pi=False):

        _log_p, pi, cost = self._inner(x, opts)

        # cost, mask = self.problem.get_costs(input, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll, e = self._calc_log_likelihood(_log_p, pi, None)
        if return_pi:
            return -cost, ll, pi, e
        # print(ll)
        return -cost, ll, e

    def _calc_log_likelihood(self, _log_p, a, mask):

        entropy = -(_log_p * _log_p.exp()).sum(2).sum(1).mean()
        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0
        if not (log_p > -10000).data.all():
            print(log_p.nonzero())
        assert (
            log_p > -10000
        ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        # print(log_p.sum(1))

        return log_p.sum(1), entropy

    def _inner(self, input, opts):

        outputs = []
        sequences = []

        state = self.problem.make_state(input, opts.u_size, opts.v_size, opts)

        batch_size = state.batch_size
        graph_size = state.u_size + state.v_size + 1
        i = 1
        while not (state.all_finished()):
            step_size = state.i + 1
            mask = state.get_mask()
            w = state.get_current_weights(mask)
            # Pass the graph to the Encoder
            node_features = state.get_node_features()
            nodes = torch.cat(
                (
                    torch.arange(0, opts.u_size + 1, device=opts.device),
                    state.idx[:i] + opts.u_size + 1,
                )
            )
            subgraphs = (
                (nodes.unsqueeze(0).expand(batch_size, step_size))
                + torch.arange(
                    0, batch_size * graph_size, graph_size, device=opts.device
                ).unsqueeze(1)
            ).flatten()  # The nodes of the current subgraphs
            graph_weights = state.get_graph_weights()
            edge_i, weights = subgraph(
                subgraphs,
                state.graphs.edge_index,
                graph_weights.unsqueeze(1),
                relabel_nodes=True,
            )
            embeddings = checkpoint(
                self.embedder,
                node_features,
                edge_i,
                weights.float(),
                torch.tensor(i),
                self.dummy,
            ).reshape(batch_size, step_size, -1)
            pos = torch.argsort(state.idx[:i])[-1]
            incoming_node_embeddings = embeddings[
                :, pos + state.u_size + 1, :
            ].unsqueeze(1)
            # print(incoming_node_embeddings)
            w = (state.adj[:, state.get_current_node(), :]).float()
            # mean_w = w.mean(1)[:, None, None].repeat(1, state.u_size + 1, 1)
            s = w.reshape(state.batch_size, state.u_size + 1, 1)
            idx = (
                torch.ones(state.batch_size, 1, 1, device=opts.device)
                * i
                / state.v_size
            )
            s = torch.cat(
                (
                    s,
                    idx.repeat(1, state.u_size + 1, 1),
                    incoming_node_embeddings.repeat(1, state.u_size + 1, 1),
                ),
                dim=2,
            )
            pi = self.ff(s).reshape(state.batch_size, state.u_size + 1)
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
        probs[mask] = -1e6
        p = torch.log_softmax(probs, dim=1)
        # print(p)
        if self.decode_type == "greedy":
            _, selected = p.max(1)
            # assert not mask.gather(
            #     1, selected.unsqueeze(-1)
            # ).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = p.exp().multinomial(1).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            # while mask.gather(1, selected.unsqueeze(-1)).data.any():
            #     print("Sampled bad values, resampling!")
            #     selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected, p
