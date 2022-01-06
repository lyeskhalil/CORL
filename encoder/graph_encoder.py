import torch
from torch import nn
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
        node_dim_u=1,
        node_dim_v=1,
        normalization="batch",
        feed_forward_hidden=512,
    ):
        super(MPNN, self).__init__()
        self.l1 = nn.Linear(1, embed_dim * embed_dim)  # TODO: Change back
        self.node_embed_u = nn.Linear(node_dim_u, embed_dim)
        if node_dim_u != node_dim_v:
            self.node_embed_v = nn.Linear(node_dim_v, embed_dim)
        else:
            self.node_embed_v = self.node_embed_u

        self.conv1 = NNConv(
            embed_dim, embed_dim, self.l1, aggr="mean"
        )  # TODO: Change back
        if n_layers > 1:
            self.l2 = nn.Linear(1, embed_dim ** 2)
            self.conv2 = NNConv(embed_dim, embed_dim, self.l2, aggr="mean")
        self.n_layers = n_layers
        self.problem = opts.problem
        self.u_size = opts.u_size
        self.node_dim_u = node_dim_u
        self.node_dim_v = node_dim_v
        self.batch_size = opts.batch_size

    def forward(self, x, edge_index, edge_attribute, i, dummy, opts):
        i = i.item()
        graph_size = opts.u_size + 1 + i
        if i < self.n_layers:
            n_encode_layers = i + 1
        else:
            n_encode_layers = self.n_layers
        x_u = x[:, : self.node_dim_u * (opts.u_size + 1)].reshape(
            self.batch_size, opts.u_size + 1, self.node_dim_u
        )
        x_v = x[:, self.node_dim_u * (opts.u_size + 1) :].reshape(
            self.batch_size, i, self.node_dim_v
        )
        x_u = self.node_embed_u(x_u)
        x_v = self.node_embed_v(x_v)
        x = torch.cat((x_u, x_v), dim=1).reshape(self.batch_size * graph_size, -1)

        for j in range(n_encode_layers):
            # x = F.relu(x) # TODO: Change back
            if j == 0:
                x = self.conv1(x, edge_index, edge_attribute.float())
                if n_encode_layers > 1:
                    x = F.relu(x)
            else:
                x = self.conv2(x, edge_index, edge_attribute.float())

        return x
