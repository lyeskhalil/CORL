import torch
import numpy as np
from torch import nn
import math
from torch_geometric.nn import NNConv
from torch_geometric.nn import BatchNorm
import torch.nn.functional as F


class MPNN(nn.Module):
    def __init__(
        self,
        n_heads,
        embed_dim,
        n_layers,
        problem,
        opts,
        dropout=0.1,
        alpha=0.01,
        node_dim=1,
        normalization="batch",
        feed_forward_hidden=512,
    ):
        super(MPNN, self).__init__()
        self.l1 = nn.Linear(1, embed_dim ** 2)
        self.node_embed = nn.Linear(node_dim, embed_dim)
        self.conv1 = NNConv(embed_dim, embed_dim, self.l1, aggr="mean")
        # self.norm = BatchNorm(embed_dim)
        self.node_dim = node_dim
        self.n_layers = n_layers

    def init_parameters(self, module):
        for name, param in module.named_parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index, edge_attribute, i, dummy):
        i = i.item()
        if i < self.n_layers:
            n_encode_layers = i + 1
        else:
            n_encode_layers = self.n_layers
        x = self.node_embed(x)

        for j in range(n_encode_layers):
            x = F.relu(x)
            x = self.conv1(x, edge_index, edge_attribute.float())

        # x = self.norm(x.view(-1, x.size(-1))).view(*x.size())

        return x
