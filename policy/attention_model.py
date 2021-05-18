import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple

import torch.nn.functional as F

# from utils.tensor_functions import compute_in_batches

from encoder.graph_encoder_v2 import GraphAttentionEncoder
from train import clip_grad_norms

#from encoder.graph_encoder import MPNN
from torch.nn import DataParallel
from torch_geometric.utils import subgraph

# from utils.functions import sample_many

import time


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def train_n_step(cost, ll, x, optimizers, baseline, opts):
    bl_val, bl_loss = baseline.eval(x, cost)

    # Calculate loss
    # print("\nCost: " , cost.item())
    reinforce_loss = ((cost.squeeze(1) - bl_val) * ll).mean()
    loss = reinforce_loss + bl_loss
    # print(loss.item())
    # Perform backward pass and optimization step
    # s = time.time()
    optimizers[0].zero_grad()
    loss.backward()
    # print(time.time() - s)
    clip_grad_norms(optimizers[0].param_groups, opts.max_grad_norm)
    optimizers[0].step()
    # optimizers[1].step()
    # print(time.time() - s)
    return


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """

    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key],
            )
        # return super(AttentionModelFixed, self).__getitem__(key)
        return self[key]


class AttentionModel(nn.Module):
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
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.is_bipartite = True
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.opts = opts
        # Problem specific context parameters (placeholder and step context dimension)
        step_context_dim = 0
        # node_dim = 0
        if self.is_bipartite:  # online bipartite matching
            step_context_dim = (
                embedding_dim * 1
            )  # Embedding of edges chosen and current node
            # node_dim = 1  # edge weight

            # Learned input symbols for first action
            self.W_placeholder = nn.Parameter(torch.Tensor(embedding_dim * 1))
            self.W_placeholder.data.uniform_(
                -1, 1
            )  # Placeholder should be in range of activations
        # self.init_embed = nn.Linear(node_dim, embedding_dim)

        encoder_class = {"attention": GraphAttentionEncoder}.get(
            encoder, None
        )

        self.embedder = encoder_class(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization,
            problem=self.problem,
            opts=self.opts,
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim)
        self.project_step_context = nn.Linear(
            step_context_dim + opts.u_size + 1, embedding_dim
        )
        self.get_edge_embed = nn.Linear(2 * embedding_dim, embedding_dim)
        # self.project_node_features = nn.Linear(1, embedding_dim)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim)
        # nn.init.xavier_uniform_(self.project_node_embeddings.weight)
        # nn.init.xavier_uniform_(self.project_node_features.weight)
        # nn.init.xavier_uniform_(self.project_fixed_context.weight)
        # nn.init.xavier_uniform_(self.project_step_context.weight)
        # nn.init.xavier_uniform_(self.get_edge_embed.weight)
        # nn.init.xavier_uniform_(self.project_out.weight)
        # self.init_parameters()
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, opts, optimizer, baseline, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        # if (
        #     self.checkpoint_encoder and self.training
        # ):  # Only checkpoint if we need gradients
        #     embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        # # else:
        #     embeddings, _ = self.embedder(self._init_embed(input))
        # s = time.time()
        _log_p, pi, cost = self._inner(input, opts, optimizer, baseline)
        # print(time.time() - s)
        # cost, mask = self.problem.get_costs(input, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, None)
        if return_pi:
            return -cost, ll, pi

        return -cost, ll

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def _calc_log_likelihood(self, _log_p, a, mask):
        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0
        assert (
            log_p > -1000
        ).data.all(), "Logprobs should not be -inf, check sampling procedure!"
        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, input):

        return self.init_embed(input)

    def _inner(self, input, opts, optimizer, baseline):

        outputs = []
        sequences = []

        state = self.problem.make_state(input, opts.u_size, opts.v_size, opts)
        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        # fixed = self._precompute(embeddings)
        step_context = 0
        batch_size = state.batch_size
        graph_size = state.u_size + state.v_size + 1
        i = 1

        while not (state.all_finished()):
            step_size = state.i + 1

            # Pass the graph to the Encoder
            node_features = (
                torch.cat(
                    (
                        torch.ones(1, device=opts.device),
                        torch.ones(opts.u_size, device=opts.device) * 2,
                        torch.ones(step_size - opts.u_size - 1, device=opts.device) * 3,
                    )
                )
                .unsqueeze(0)
                .expand(batch_size, step_size)
                .reshape(batch_size * step_size, 1)
            ).float()  # Collecting node features up until the ith incoming node
            subgraphs = (
                (
                    torch.arange(0, step_size, device=opts.device)
                    .unsqueeze(0)
                    .expand(batch_size, step_size)
                )
                + torch.arange(
                    0, batch_size * graph_size, graph_size, device=opts.device
                ).unsqueeze(1)
            ).flatten()  # The nodes of the current subgraphs
            edge_i, weights = subgraph(
                subgraphs,
                state.graphs.edge_index,
                state.graphs.weight.unsqueeze(1),
                relabel_nodes=True,
            )
            if i % opts.checkpoint_every == 0:
                embeddings = checkpoint(
                    self.embedder,
                    node_features,
                    edge_i,
                    weights.float(),
                    torch.tensor(i),
                    self.dummy,
                ).reshape(batch_size, step_size, -1)
            else:
                embeddings = self.embedder(
                    node_features, edge_i, weights.float(), i
                ).reshape(batch_size, step_size, -1)

            # context node embedding
            fixed = self._precompute(embeddings, step_size, opts, state)

            # Decoder
            log_p, mask = self._get_log_p(
                fixed, state, step_context, opts, embeddings[:, -1, :]
            )

            # Select a Node
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :].bool())

            # Update state information
            state = state.update(selected[:, None])
            s = (selected[:, None].repeat(1, fixed.node_embeddings.size(-1)))[
                :, None, :
            ]
            step_context = (
                step_context
                + (
                    self.get_edge_embed(
                        torch.cat(
                            (
                                torch.gather(fixed.node_embeddings, 1, s),
                                embeddings[:, -1, :].unsqueeze(1),
                            ),
                            dim=2,
                        )
                    )
                    #                    / 2
                    - step_context
                )
                / i
            )  # Incremental averaging of selected edges
            # Collect output of step
            # step_size = ((state.i.item() - state.u_size.item() + 1) * (state.u_size + 1))
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            if (
                (optimizer is not None)
                and opts.n_step
                and (i % opts.max_steps == 0 or state.all_finished())
            ):
                _log_p, pi, cost = (
                    torch.stack(outputs[i - opts.max_steps : i], 1),
                    torch.stack(sequences[i - opts.max_steps : i], 1),
                    -state.size / i,
                )

                # policy gradient
                ll = self._calc_log_likelihood(_log_p, pi, None)
                train_n_step(cost, ll, None, optimizer, baseline, opts)
                step_context = step_context.detach()
                # initial_embeddings = self.project_node_features(node_features).reshape(batch_size, graph_size, -1)
                # state = state._replace(size=state.size.detach())
            i += 1
        # Collected lists, return Tensor
        return (
            torch.stack(outputs, 1),
            torch.stack(sequences, 1),
            state.size,
        )

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any nans"
        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(
                1, selected.unsqueeze(-1)
            ).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, step_size, opts, state, num_steps=1):
        # calculate the mean of the embeddings of the edges
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]
        # embed_dim = embeddings.size()[-1]
        # u = opts.u_size + 1
        # v = step_size - u
        # embeddings = (
        #     embeddings.reshape(state.batch_size, u, v, embed_dim)
        #     .transpose(1, 2)
        #     .reshape(state.batch_size, u * v, embed_dim)
        # )
        # offset = u * (step_size - u - 1)
        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(
            torch.cat(
                (
                    embeddings[:, None, : opts.u_size + 1, :],
                    embeddings[:, None, -1, :].unsqueeze(2),
                ),
                dim=2,
            )
            # embeddings[:, None, : step_size, :]
        ).chunk(
            3, dim=-1
        )
        # print(embeddings[:, None, offset: offset + opts.u_size + 1, :])
        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous(),
        )
        return AttentionModelFixed(
            torch.cat(
                (
                    embeddings[:, : opts.u_size + 1, :],
                    embeddings[:, -1, :].unsqueeze(1),
                ),
                dim=1,
            ),
            # embeddings[:, offset : offset + u, :],
            fixed_context,
            *fixed_attention_node_data,
        )

    def _get_log_p(self, fixed, state, step_context, opts, curr_node, normalize=True):

        # Compute query = context node embedding
        projected_step_context = self.project_step_context(
            self._get_parallel_step_context(
                fixed.node_embeddings, state, step_context, curr_node
            )
        )
        query = fixed.context_node_projected + projected_step_context

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask

        mask = state.get_mask()[:, None, :]
        mask = torch.cat(
            (mask, torch.zeros(mask.size(0), 1, 1, device=mask.device).long()), dim=2
        )
        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(
            query, glimpse_K, glimpse_V, logit_K, mask
        )
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)
        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(
        self, embeddings, state, step_context, curr_node, from_depot=False
    ):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        # current_node = state.get_current_node()
        # batch_size = state.batch_size
        num_steps = 1

        if self.is_bipartite:  # Bipartite matching
            if (
                num_steps == 1
            ):  # We need to special case if we have only 1 step, may be the first or not
                if state.i == state.u_size + 1:
                    # First and only step, ignore prev_a (this is a placeholder)
                    return torch.cat(
                        (
                            # self.W_placeholder[None, None, :].expand(
                            # batch_size, 1, self.W_placeholder.size(-1)
                            # ),
                            curr_node.unsqueeze(1),
                            state.adj[:, 0, :].float().unsqueeze(1),
                        ),
                        dim=2,
                    )
                    # return self.W_placeholder[None, None, :].expand(
                    #     batch_size, 1, self.W_placeholder.size(-1)
                    # )
                #        return (curr_node.unsqueeze(1))
                else:
                    return torch.cat(
                        (
                            curr_node.unsqueeze(1),
                            state.adj[:, 0, :].float().unsqueeze(1),
                        ),
                        dim=2,
                    )  # add embedding of arriving node to context
                    # return step_context

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads
        mask = mask.bool()
        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(
            batch_size, num_steps, self.n_heads, 1, key_size
        ).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(
            glimpse_Q, glimpse_K.transpose(-2, -1)
        ) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[
                mask[None, :, :, None, :].expand_as(compatibility)
            ] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)
        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4)
            .contiguous()
            .view(-1, num_steps, 1, self.n_heads * val_size)
        )

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(
            -2
        ) / math.sqrt(final_Q.size(-1))
        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf
            logits[:, :, -1] = -math.inf
        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):

        # Bipartite
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous()
            .view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(
                v.size(0),
                v.size(1) if num_steps is None else num_steps,
                v.size(2),
                self.n_heads,
                -1,
            )
            .permute(
                3, 0, 1, 2, 4
            )  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )


class CachedLookup(object):
    def __init__(self, data):
        self.orig = data
        self.key = None
        self.current = None

    def __getitem__(self, key):
        assert not isinstance(key, slice), (
            "CachedLookup does not support slicing, "
            "you can slice the result of an index operation instead"
        )

        if torch.is_tensor(key):  # If tensor, idx all tensors by this tensor:

            if self.key is None:
                self.key = key
                self.current = self.orig[key]
            elif len(key) != len(self.key) or (key != self.key).any():
                self.key = key
                self.current = self.orig[key]

            return self.current

        return super(CachedLookup, self).__getitem__(key)
        # return self[key]
