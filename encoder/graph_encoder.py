import torch
import numpy as np
from torch import nn
import math
from torch_geometric.nn import NNConv
import torch.nn.functional as F


class MPNN(torch.nn.Module):
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
        self.conv1 = NNConv(node_dim, embed_dim, torch.nn.Linear(1, embed_dim))
        self.conv2 = NNConv(embed_dim, embed_dim, torch.nn.Linear(1, embed_dim ** 2))
        self.conv3 = NNConv(embed_dim, embed_dim, torch.nn.Linear(1, embed_dim ** 2))
        self.dropout = dropout
        self.node_dim = node_dim

    def forward(self, x, edge_index, edge_attribute):
        x = self.conv1(x, edge_index, edge_attribute)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attribute)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attribute)

        return x
