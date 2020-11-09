import torch
import numpy as np
from torch import nn
import math
import torch.nn.functional as F


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, mask=None, weights=None):
        return input + self.module(input, mask=mask, weights=weights)


class SkipConnection1(nn.Module):
    def __init__(self, module):
        super(SkipConnection1, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        input_dim,
        problem=None,
        embed_dim=None,
        val_dim=None,
        key_dim=None,
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim
        # print(input_dim)
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.problem = problem
        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None, weights=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention
        # print(h.shape)
        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        if self.problem == "obm":
            compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        else:  # Need to encode edge weights
            # Get the full weight matrix of size (batch_size, graph_size, graph_size)
            v = graph_size - weights.size(2)
            u = weights.size(2)
            weights1 = torch.cat(
                (
                    torch.zeros((batch_size, u, u), device=weights.device),
                    weights[:, :v, :].transpose(1, 2).float(),
                ),
                dim=2,
            )
            weights2 = torch.cat(
                (
                    weights[:, :v, :].float(),
                    torch.zeros((batch_size, v, v), device=weights.device),
                ),
                dim=2,
            )
            weights = torch.cat((weights1, weights2), dim=1)

            # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
            compatibility = (
                self.norm_factor
                * torch.matmul(Q, K.transpose(2, 3))
                # * (weights + (weights == 0).float())
            )
        mask = (mask.float() - torch.diag_embed(torch.ones(batch_size, graph_size, device=mask.device))).bool()
        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(
                compatibility
            )
            compatibility[mask] = -1e10
        # attn = torch.softmax(compatibility, dim=-1)
        # attn = torch.softmax(compatibility * (weights + (weights == 0).float()), dim=-1)
        compatibility = compatibility.exp() * (weights + (weights == 0).float())
        attn = torch.nn.functional.normalize(compatibility, dim=-1, p=1)
        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        print(attn, weights)
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc
        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3)
            .contiguous()
            .view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim),
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()

        normalizer_class = {"batch": nn.BatchNorm1d, "instance": nn.InstanceNorm1d}.get(
            normalization, None
        )

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        embed_dim,
        problem=None,
        feed_forward_hidden=512,
        normalization="batch",
    ):
        super(MultiHeadAttentionLayer, self).__init__()
        self.attention = SkipConnection(
            MultiHeadAttention(
                n_heads, problem=problem, input_dim=embed_dim, embed_dim=embed_dim
            )
        )
        self.norm = Normalization(embed_dim, normalization)
        self.ff = SkipConnection1(
            nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, embed_dim),
            )
            if feed_forward_hidden > 0
            else nn.Linear(embed_dim, embed_dim)
        )
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, input, mask=None, weights=None):
        # print(input)
        h = self.attention(input, mask=mask, weights=weights)
        h = self.norm(h)
        h = self.ff(h)
        h = self.norm2(h)
        return h


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        n_heads,
        embed_dim,
        n_layers,
        problem,
        node_dim=None,
        normalization="batch",
        feed_forward_hidden=512,
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = (
            nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        )
        self.layers = nn.ModuleList(
            [
                *(
                    MultiHeadAttentionLayer(
                        n_heads,
                        embed_dim,
                        feed_forward_hidden=feed_forward_hidden,
                        normalization=normalization,
                        problem=problem,
                    )
                    for _ in range(n_layers)
                )
            ]
        )

    def forward(self, x, mask=None, weights=None):

        # assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = (
            self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1)
            if self.init_embed is not None
            else x
        )
        # print(h.shape)
        for layer in self.layers:
            h = layer(h, mask=mask, weights=weights)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


class GAT(nn.Module):
    def __init__(
            self,
            dropout,
            alpha,
            weights,
            n_heads,
            embed_dim,
            n_layers,
            problem,
            problem,
            node_dim=None,
    ):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [
            GraphAttentionLayer(node_dim, embed_dim, problem=problem, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(n_heads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

        # self.out_att = GraphAttentionLayer(
        #    nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False
        # )

    def forward(self, x, adj, weights):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, weights) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        #return F.log_softmax(x, dim=1)
        return x


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, problem, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.problem = problem
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, weights):
        Wh = torch.mm(
            h, self.W
        )  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        batch_size, graph_size, input_dim = h.size()
        v = graph_size - weights.size(2)
        u = weights.size(2)
        weights1 = torch.cat(
            (
                torch.zeros((batch_size, u, u), device=weights.device),
                weights[:, :v, :].transpose(1, 2).float(),
            ),
            dim=2,
        )
        weights2 = torch.cat(
            (
                weights[:, :v, :].float(),
                torch.zeros((batch_size, v, v), device=weights.device),
            ),
            dim=2,
        )
        weights = torch.cat((weights1, weights2), dim=1)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = attention.exp() * weights
        attention = F.normalize(attention, dim=1, p=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1
        )
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
