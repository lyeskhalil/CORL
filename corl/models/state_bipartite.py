import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
import numpy as np
import networkx as nx

"""
TODO: CODE BELOW SHOULD BE MODIFIED TO WORK FOR BIPARTITE
"""


class StateBipartite(NamedTuple):
    # Fixed input
    graphs: torch.Tensor  # full adjacency matrix of all graphs in a batch
    weights: torch.Tensor  # weights of all edges of each graph in a batch
    edges: torch.Tensor  # edges of each graph in a batch
    degree: torch.Tensor  # degree of each node in the V set
    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    # curr_edge: torch.Tensor  # current edge number
    matched_nodes: torch.Tensor  # Keeps track of nodes that have been matched
    picked_edges: torch.Tensor
    size: torch.Tensor  # size of current matching
    i: torch.Tensor  # Keeps track of step
    # mask: torch.Tensor  # mask for each step

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.matched_nodes

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(
            key, slice
        ):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                graphs=self.graphs[key],
                weights=self.weights[key],
                matched_nodes=self.matched_nodes[key],
                picked_edges=self.picked_edges[key],
                size=self.size[key],
                edges=self.edges[key],
                degree=self.degree[key],
            )
        return super(StateBipartite, self).__getitem__(key)

    @staticmethod
    def initialize(graphs, u_size, num_edges, visited_dtype=torch.uint8):

        batch_size, n_loc, _ = graphs.size()
        # size = torch.zeros(batch_size, 1, dtype=torch.long, device=graphs.device)
        return StateBipartite(
            graphs=torch.tensor(graphs[:, 0]),
            weights=torch.tensor(graphs[:, 1]),
            edges=torch.tensor(graphs[:, 2]),
            degree=torch.tensor(graphs[:, 3]),
            ids=torch.arange(batch_size, dtype=torch.int64, device=graphs.device)[
                :, None
            ],  # Add steps dimension
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            matched_nodes=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, u_size, dtype=torch.uint8, device=graphs.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(
                    batch_size,
                    1,
                    (n_loc + 63) // 64,
                    dtype=torch.int64,
                    device=graphs.device,
                )  # Ceil
            ),
            picked_edges=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, num_edges, dtype=torch.uint8, device=graphs.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(
                    batch_size,
                    1,
                    (n_loc + 63) // 64,
                    dtype=torch.int64,
                    device=graphs.device,
                )  # Ceil
            ),
            size=torch.zeros(batch_size, 1, device=graphs.device),
            i=torch.zeros(
                1, dtype=torch.int64, device=graphs.device
            ),  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.size

    def update(self, selected):
        # Update the state
        nodes = self.matched_nodes.scatter_(-1, selected[:, :-1], 1)
        selected_u = selected.T.expand(self.edges.shape[1], -1)
        selected_v = (self.i).T.expand(self.edges.shape[1], -1)
        mask = (selected_u == self.edges[:, 0]) & (selected_v == self.edges[:, 1])
        total_weights = self.size + torch.sum(self.weights * mask.T.long(), dim=1)
        edges = self.picked_edges + mask.long()
        # mask = edges.scatter_(-1, self.edges[:, :, 0] == )
        return self._replace(
            matched_nodes=nodes, size=total_weights, picked_edges=edges, i=self.i + 1,
        )

    def all_finished(self):
        # Exactly n steps
        return self.i.item() >= self.degree.size(-1)

    def get_current_node(self):
        return self.i

    def get_mask(self):
        return (
            self.matched_nodes > 0
        )  # Hacky way to return bool or uint8 depending on pytorch version

    def get_nn(self, k=None):
        # Insert step dimension
        # Nodes already visited get inf so they do not make it
        if k is None:
            k = self.loc.size(-2) - self.i.item()  # Number of remaining
        return (
            self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6
        ).topk(k, dim=-1, largest=False)[1]

    def get_nn_current(self, k=None):
        assert (
            False
        ), "Currently not implemented, look into which neighbours to use in step 0?"
        # Note: if this is called in step 0, it will have k nearest neighbours to node 0, which may not be desired
        # so it is probably better to use k = None in the first iteration
        if k is None:
            k = self.loc.size(-2)
        k = min(k, self.loc.size(-2) - self.i.item())  # Number of remaining
        return (self.dist[self.ids, self.prev_a] + self.visited.float() * 1e6).topk(
            k, dim=-1, largest=False
        )[1]

    def construct_solutions(self, actions):
        return actions
