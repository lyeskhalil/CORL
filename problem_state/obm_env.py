import torch
from typing import NamedTuple

# from utils.boolmask import mask_long2bool, mask_long_scatter


class StateBipartite(NamedTuple):
    # Fixed input
    graphs: torch.Tensor  # full adjacency matrix of all graphs in a batch
    # adj: torch.Tensor # full adjacency matrix of all graphs in a batch
    weights: torch.Tensor
    u_size: torch.Tensor
    v_size: torch.Tensor
    batch_size: torch.Tensor
    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

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
                matched_nodes=self.matched_nodes[key],
                size=self.size[key],
                u_size=self.u_size[key],
                v_size=self.v_size[key],
            )
        # return super(StateBipartite, self).__getitem__(key)
        return self[key]

    @staticmethod
    def initialize(
        input, u_size, v_size, num_edges, visited_dtype=torch.uint8,
    ):

        batch_size = len(input)
        # size = torch.zeros(batch_size, 1, dtype=torch.long, device=graphs.device)
        return StateBipartite(
            graphs=input,
            u_size=torch.tensor([u_size], device=input.device),
            v_size=torch.tensor([v_size], device=input.device),
            weights=None,
            batch_size=torch.tensor([batch_size], device=input.device),
            ids=torch.arange(batch_size, dtype=torch.int64, device=input.device)[
                :, None
            ],  # Add steps dimension
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            matched_nodes=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, u_size + 1, dtype=torch.uint8, device=input.device
                )
            ),
            size=torch.zeros(batch_size, 1, device=input.device),
            i=torch.ones(1, dtype=torch.int64, device=input.device)
            * (u_size + 1),  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.size

    def update(self, selected):
        # Update the state
        nodes = self.matched_nodes.squeeze(1).scatter_(-1, selected, 1)

        total_weights = self.size + (selected != 0).long()

        return self._replace(matched_nodes=nodes, size=total_weights, i=self.i + 1,)

    def all_finished(self):
        # Exactly n steps
        return (self.i.item() - (self.u_size.item() + 1)) >= self.v_size

    def get_current_node(self):
        return self.i.item()

    def get_mask(self):
        """
        Returns a mask vector which includes only nodes in U that can matched.
        That is, neighbors of the incoming node that have not been matched already.
        """
        mask = self.graphs[:, self.i.item(), : self.u_size.item() + 1]

        self.matched_nodes[
            :, 0
        ] = 0  # node that represents not being matched to anything can be matched to more than once
        return (
            self.matched_nodes.squeeze(1) + mask > 0
        ).long()  # Hacky way to return bool or uint8 depending on pytorch version

    # def get_nn(self, k=None):
    #     # Insert step dimension
    #     # Nodes already visited get inf so they do not make it
    #     if k is None:
    #         k = self.loc.size(-2) - self.i.item()  # Number of remaining
    #     return (
    #         self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6
    #     ).topk(k, dim=-1, largest=False)[1]

    # def get_nn_current(self, k=None):
    #     assert (
    #         False
    #     ), "Currently not implemented, look into which neighbours to use in step 0?"
    #     # Note: if this is called in step 0, it will have k nearest neighbours to node 0, which may not be desired
    #     # so it is probably better to use k = None in the first iteration
    #     if k is None:
    #         k = self.loc.size(-2)
    #     k = min(k, self.loc.size(-2) - self.i.item())  # Number of remaining
    #     return (self.dist[self.ids, self.prev_a] + self.visited.float() * 1e6).topk(
    #         k, dim=-1, largest=False
    #     )[1]

    # def construct_solutions(self, actions):
    #     return actions
