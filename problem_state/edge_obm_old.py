import torch
from typing import NamedTuple
from torch_geometric.utils import to_dense_adj, subgraph

# from utils.boolmask import mask_long2bool, mask_long_scatter


class StateEdgeBipartite(NamedTuple):
    # Fixed input
    graphs: torch.Tensor  # full adjacency matrix of all graphs in a batch
    # adj: torch.Tensor # full adjacency matrix of all graphs in a batch
    weights: torch.Tensor  # weights of all edges of each graph in a batch
    # edges: torch.Tensor  # edges of each graph in a batch
    u_size: torch.Tensor
    v_size: torch.Tensor
    batch_size: torch.Tensor
    hist_sum: torch.tensor
    hist_sum_sq: torch.tensor
    hist_deg: torch.tensor
    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    adj: torch.Tensor
    # State
    # curr_edge: torch.Tensor  # current edge number
    matched_nodes: torch.Tensor  # Keeps track of nodes that have been matched
    # picked_edges: torch.Tensor
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
                size=self.size[key],
                u_size=self.u_size[key],
                v_size=self.v_size[key],
            )
        return super(StateEdgeBipartite, self).__getitem__(key)
        #return self[key]

    @staticmethod
    def initialize(
        input, u_size, v_size, num_edges, visited_dtype=torch.uint8,
    ):
        graph_size = u_size + v_size + 1
        batch_size = int(input.batch.size(0) / graph_size)
        adj = to_dense_adj(input.edge_index, input.batch, input.weight.unsqueeze(1))[
            :, u_size + 1 :, : u_size + 1
        ].squeeze(-1)

        #permute the nodes for data 
        # idx = torch.randperm(adj.shape[1])
        # adj = adj[:, idx, :].view(adj.size())
        # size = torch.zeros(batch_size, 1, dtype=torch.long, device=graphs.device)
        # adj = (input[0] == 0).float()
        # adj[:, :, 0] = 0.0
        return StateEdgeBipartite(
            graphs=input,
            adj=adj,
            u_size=u_size,
            v_size=v_size,
            weights=None,
            batch_size=batch_size,
            ids=None,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            matched_nodes=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size,
                    1,
                    u_size + 1,
                    dtype=torch.uint8,
                    device=input.batch.device,
                )
            ),
            hist_sum=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size,
                    1,
                    u_size + 1,
                    dtype=torch.uint8,
                    device=input.batch.device,
                )
            ),
            hist_deg=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size,
                    1,
                    u_size + 1,
                    dtype=torch.uint8,
                    device=input.batch.device,
                )
            ),
            hist_sum_sq=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size,
                    1,
                    u_size + 1,
                    dtype=torch.uint8,
                    device=input.batch.device,
                )
            ),
            size=torch.zeros(batch_size, 1, device=input.batch.device),
            i=u_size + 1,
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.size

    def update(self, selected):
        # Update the state
        nodes = self.matched_nodes.squeeze(1).scatter_(-1, selected, 1)
        # v = self.i - (self.u_size + 1)
        # graph_size = self.u_size + self.v_size + 1
        # offset = torch.arange(
        #    0,
        #    self.batch_size * (graph_size),
        #    graph_size,
        #    device=self.graphs.batch.device,
        # ).unsqueeze(1)
        # subgraphs = torch.cat(
        #    (
        #        (
        #            torch.arange(0, self.u_size + 1, device=self.graphs.batch.device)
        #            .unsqueeze(0)
        #            .expand(self.batch_size, self.u_size + 1)
        #        )
        #        + offset,
        #        torch.tensor(self.i, device=self.graphs.batch.device).expand(
        #            self.batch_size, 1
        #        )
        #        + offset,
        #    ),
        #    dim=1,
        # ).flatten()
        # edge_i, weights = subgraph(
        #    subgraphs,
        #    self.graphs.edge_index,
        #    self.graphs.weight.unsqueeze(1),
        #    relabel_nodes=True,
        # )
        # adj = to_dense_adj(
        #    edge_i,
        #    self.graphs.batch.reshape(self.batch_size, graph_size)[
        #        :, : self.u_size + 2
        #    ].flatten(),
        #    weights,
        # )[:, -1, : self.u_size + 1].squeeze(-1)
        # v = self.i - (self.u_size + 1)
        total_weights = self.size + self.adj[:, 0, :].gather(1, selected)
        hist_sum = self.hist_sum + self.adj[:, 0, :].unsqueeze(1)
        hist_sum_sq = self.hist_sum_sq + self.adj[:, 0, :].unsqueeze(1) ** 2
        hist_deg = self.hist_deg + (self.adj[:, 0, :].unsqueeze(1) != 0).float()
        # total_weights = self.size + adj.gather(1, selected)
        return self._replace(
            matched_nodes=nodes,
            size=total_weights,
            i=self.i + 1,
            adj=self.adj[:, 1:, :],
            hist_sum=hist_sum,
            hist_sum_sq=hist_sum_sq,
            hist_deg=hist_deg,
        )

    def all_finished(self):
        # Exactly v_size steps
        return (self.i - (self.u_size + 1)) >= self.v_size

    def get_current_node(self):
        return self.i

    def get_mask(self):
        """
        Returns a mask vector which includes only nodes in U that can matched.
        That is, neighbors of the incoming node that have not been matched already.
        """
        # v = self.i - (self.u_size + 1)
        # graph_size = self.u_size + self.v_size + 1
        # offset = torch.arange(
        #    0,
        #    self.batch_size * (graph_size),
        #    graph_size,
        #    device=self.graphs.batch.device,
        # ).unsqueeze(1)
        # subgraphs = torch.cat(
        #    (
        #        (
        #            torch.arange(0, self.u_size + 1, device=self.graphs.batch.device)
        #            .unsqueeze(0)
        #            .expand(self.batch_size, self.u_size + 1)
        #        )
        #        + offset,
        #         torch.tensor(self.i, device=self.graphs.batch.device).expand(
        #            self.batch_size, 1
        #        )
        #        + offset,
        #    ),
        #    dim=1,
        # ).flatten()
        # edge_i, weights = subgraph(
        #    subgraphs,
        #   self.graphs.edge_index,
        #    self.graphs.weight.unsqueeze(1),
        #    relabel_nodes=True,
        # )
        # mask = (
        #    1.0
        #    - to_dense_adj(
        #        edge_i,
        #        self.graphs.batch.reshape(self.batch_size, graph_size)[
        #            :, : self.u_size + 2
        #        ].flatten(),
        #    )[:, -1, : self.u_size + 1]
        # )
        mask = (self.adj[:, 0, :] == 0).float()
        mask[:, 0] = 0
        self.matched_nodes[
            :, 0
        ] = 0  # node that represents not being matched to anything can be matched to more than once
        return (
            self.matched_nodes.squeeze(1) + mask > 0
        ).long()  # Hacky way to return bool or uint8 depending on pytorch version
