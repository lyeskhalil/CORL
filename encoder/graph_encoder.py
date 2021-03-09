import torch
import numpy as np
from torch import nn
import math
from torch_geometric.nn import NNConv
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
        #self.l2 = nn.Linear(1, embed_dim ** 2)
        #self.l3 = nn.Linear(1, embed_dim ** 2)
        self.conv1 = NNConv(embed_dim, embed_dim, self.l1, aggr='mean')
        self.norm = nn.BatchNorm1d(embed_dim)
        #self.conv2 = NNConv(embed_dim, embed_dim, self.l2, aggr='mean')
        #self.conv3 = NNConv(embed_dim, embed_dim, self.l3, aggr='mean')
        self.dropout = dropout
        self.node_dim = node_dim
        self.n_layers = n_layers
        nn.init.xavier_uniform_(self.conv1.root)
        nn.init.normal_(self.conv1.bias)
        nn.init.xavier_uniform_(self.node_embed.weight)
        nn.init.normal_(self.node_embed.bias)
        nn.init.normal_(self.norm.weight)
        nn.init.normal_(self.norm.bias)
        #nn.init.xavier_uniform_(self.conv2.root)
        #nn.init.xavier_uniform_(self.conv3.root)
        #nn.init.xavier_uniform_(self.l1.weight)
        #nn.init.xavier_uniform_(self.l2.weight)
        #nn.init.xavier_uniform_(self.l3.weight)
    def forward(self, x, edge_index, edge_attribute):
        x = self.node_embed(x)
        for i in range(self.n_layers):
            x = F.leaky_relu(x)
            x = self.conv1(x, edge_index, edge_attribute.float() / 100.)
        

        #x = F.leaky_relu(x)
        #x = self.norm(x.view(-1, x.size(-1))).view(*x.size())
        #x = F.dropout(x, p=self.dropout, training=self.training)
        #x = self.conv2(x, edge_index, edge_attribute)
        #x = F.leaky_relu(x)
        #x = self.conv3(x, edge_index, edge_attribute)

        return x
