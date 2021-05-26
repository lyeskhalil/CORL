import torch
from typing import NamedTuple
from torch_geometric.utils import to_dense_adj, subgraph

# from utils.boolmask import mask_long2bool, mask_long_scatter


class StateEdgeBipartite(NamedTuple):
    # Fixed input
    graphs: torch.Tensor  # graphs objects in a batch
    u_size: int
    v_size: int
    batch_size: torch.Tensor
    hist_sum: torch.Tensor
    hist_sum_sq: torch.Tensor
    hist_deg: torch.Tensor
    min_sol: torch.Tensor
    max_sol: torch.Tensor
    sum_sol_sq: torch.Tensor
    num_skip: torch.Tensor

    adj: torch.Tensor
    # State
    matched_nodes: torch.Tensor  # Keeps track of nodes that have been matched
    size: torch.Tensor  # size of current matching
    i: int  # Keeps track of step

    @staticmethod
    def initialize(
        input, u_size, v_size, opts,
    ):
        graph_size = u_size + v_size + 1
        batch_size = int(input.batch.size(0) / graph_size)
        adj = to_dense_adj(input.edge_index, input.batch, input.weight.unsqueeze(1))[
            :, u_size + 1 :, : u_size + 1
        ].squeeze(-1)

        # permute the nodes for data
        if opts.model != "supervised" or not opts.eval:
            idx = torch.randperm(adj.shape[1])
            adj = adj[:, idx, :].view(adj.size())
        # size = torch.zeros(batch_size, 1, dtype=torch.long, device=graphs.device)
        # adj = (input[0] == 0).float()
        # adj[:, :, 0] = 0.0
        return StateEdgeBipartite(
            graphs=input,
            adj=adj,
            u_size=u_size,
            v_size=v_size,
            batch_size=batch_size,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            matched_nodes=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(batch_size, u_size + 1, device=input.batch.device,)
            ),
            hist_sum=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(batch_size, 1, u_size + 1, device=input.batch.device,)
            ),
            hist_deg=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(batch_size, 1, u_size + 1, device=input.batch.device,)
            ),
            hist_sum_sq=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(batch_size, 1, u_size + 1, device=input.batch.device,)
            ),
            min_sol=torch.zeros(batch_size, 1, device=input.batch.device),
            max_sol=torch.zeros(batch_size, 1, device=input.batch.device),
            sum_sol_sq=torch.zeros(batch_size, 1, device=input.batch.device),
            num_skip=torch.zeros(batch_size, 1, device=input.batch.device),
            size=torch.zeros(batch_size, 1, device=input.batch.device),
            i=u_size + 1,
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.size

    def get_current_weights(self):
        return self.adj[:, 0, :].float()

    def update(self, selected):
        # Update the state
        nodes = self.matched_nodes.scatter_(-1, selected, 1)
        nodes[
            :, 0
        ] = 0  # node that represents not being matched to anything can be matched to more than once
        selected_weights = (
            self.adj[:, 0, :].clone().gather(1, selected).to(self.adj.device)
        )
        skip = (selected == 0).float()
        num_skip = self.num_skip + skip
        if self.i == self.u_size + 1:
            min_sol = selected_weights
        else:
            m = self.min_sol.clone()
            m[m == 0.0] = 2.0
            selected_weights[skip.bool()] = 2.0
            min_sol = torch.minimum(m, selected_weights)
            selected_weights[selected_weights == 2.0] = 0.0
            min_sol[min_sol == 2.0] = 0.0

        max_sol = torch.maximum(self.max_sol, selected_weights)
        total_weights = self.size + selected_weights
        sum_sol_sq = self.sum_sol_sq + selected_weights ** 2

        hist_sum = self.hist_sum + self.adj[:, 0, :].unsqueeze(1)
        hist_sum_sq = self.hist_sum_sq + self.adj[:, 0, :].unsqueeze(1) ** 2
        hist_deg = self.hist_deg + (self.adj[:, 0, :].unsqueeze(1) != 0).float()

        return self._replace(
            matched_nodes=nodes,
            size=total_weights,
            i=self.i + 1,
            adj=self.adj[:, 1:, :],
            hist_sum=hist_sum,
            hist_sum_sq=hist_sum_sq,
            hist_deg=hist_deg,
            num_skip=num_skip,
            max_sol=max_sol,
            min_sol=min_sol,
            sum_sol_sq=sum_sol_sq,
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

        mask = (self.adj[:, 0, :] == 0).float()
        mask[:, 0] = 0
        self.matched_nodes[
            :, 0
        ] = 0  # node that represents not being matched to anything can be matched to more than once
        return (
            self.matched_nodes + mask > 0
        ).long()  # Hacky way to return bool or uint8 depending on pytorch version
