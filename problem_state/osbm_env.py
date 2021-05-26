import torch
from typing import NamedTuple
from torch_geometric.utils import to_dense_adj, subgraph

# from utils.boolmask import mask_long2bool, mask_long_scatter


class StateOSBM(NamedTuple):
    # Fixed input
    graphs: torch.Tensor  # graphs objects in a batch
    u_size: int
    v_size: int
    num_genres: int
    num_users: int
    batch_size: torch.Tensor
    hist_sum: torch.tensor
    hist_sum_sq: torch.tensor
    hist_deg: torch.tensor
    adj: torch.Tensor
    users: torch.Tensor
    u_features: torch.Tensor
    v_features: torch.Tensor
    # State
    matched_nodes: torch.Tensor  # Keeps track of nodes that have been matched
    size: torch.Tensor  # size of current matching
    i: int  # Keeps track of step

    @staticmethod
    def initialize(
        input, u_size, v_size, opts,
    ):
        num_genres = 15
        num_users = 200
        graph_size = u_size + v_size + 1
        batch_size = int(input.batch.size(0) / graph_size)
        adj = to_dense_adj(input.edge_index, input.batch)[
            :, u_size + 1 :, : u_size + 1
        ].squeeze(-1)
        print(input.edge_index)
        u_features = input.x.reshape(batch_size, -1)[:, : u_size * num_genres].reshape(
            batch_size, u_size, -1
        )

        # add features of future node
        u_features = torch.cat(
            (torch.zeros(batch_size, 1, num_genres, device=opts.device), u_features),
            dim=1,
        )
        offset = u_size * num_genres

        v_features = input.x.reshape(batch_size, -1)[:, offset:].reshape(
            batch_size, v_size, -1
        )
        # permute the nodes for data
        if opts.model != "supervised":
            idx = torch.randperm(adj.shape[1])
            adj = adj[:, idx, :].view(adj.size())
            v_features = v_features[:, idx, :].view(v_features.size())
        return StateOSBM(
            graphs=input,
            adj=adj,
            u_size=u_size,
            v_size=v_size,
            batch_size=batch_size,
            matched_nodes=(
                torch.zeros(batch_size, 1, u_size + 1, device=input.batch.device,)
            ),
            hist_sum=(
                torch.zeros(batch_size, 1, u_size + 1, device=input.batch.device,)
            ),
            hist_deg=(
                torch.zeros(batch_size, 1, u_size + 1, device=input.batch.device,)
            ),
            hist_sum_sq=(
                torch.zeros(batch_size, 1, u_size + 1, device=input.batch.device,)
            ),
            size=torch.zeros(batch_size, 1, device=input.batch.device),
            users=torch.zeros(
                batch_size, num_users, num_genres, device=input.batch.device
            ),
            i=u_size + 1,
            u_features=u_features,
            v_features=v_features,
            num_genres=num_genres,
            num_users=num_users,
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.size

    def update(self, selected):
        # Update the state
        v = self.i - (self.u_size + 1)
        users_features = self.v_features[:, v, :]
        idx = (
            selected
            + torch.arange(
                0,
                self.batch_size * (self.u_size + 1),
                (self.u_size + 1),
                device=self.adj.device,
            ).unsqueeze(1)
        ).flatten()
        selected_movie_genre = self.u_features.reshape(
            self.batch_size * (self.u_size + 1), -1
        ).index_select(0, idx)
        users_idx = users_features[:, -1].int() + torch.arange(
            0, self.batch_size * self.num_users, self.num_users, device=self.adj.device
        )
        users_covered_genre = self.users.reshape(
            self.batch_size * self.num_users, -1
        ).index_select(0, users_idx)
        s = ((selected_movie_genre + users_covered_genre) > 0).float()
        curr_weights = self.get_current_weights(users_covered_genre)
        selected_weights = curr_weights.gather(1, selected)
        total_weights = self.size + selected_weights
        # indices = torch.arange(0, self.num_genres, device=selected.device).unsqueeze(0).expand(self.batch_size, self.num_genres)
        updated_users_selected_genre = self.users.reshape(
            self.batch_size * self.num_users, -1
        ).index_copy(0, users_idx, s)
        nodes = self.matched_nodes.squeeze(1).scatter_(-1, selected, 1)
        hist_sum = self.hist_sum + curr_weights.unsqueeze(1)
        hist_sum_sq = self.hist_sum_sq + curr_weights.unsqueeze(1) ** 2
        hist_deg = self.hist_deg + (curr_weights.unsqueeze(1) != 0).float()
        return self._replace(
            matched_nodes=nodes,
            size=total_weights,
            i=self.i + 1,
            adj=self.adj[:, 1:, :],
            hist_sum=hist_sum,
            hist_sum_sq=hist_sum_sq,
            hist_deg=hist_deg,
            users=updated_users_selected_genre.reshape(
                self.batch_size, self.num_users, -1
            ),
        )

    def get_current_weights(self, users_covered_genre=None):
        v = self.i - (self.u_size + 1)
        users_features = self.v_features[:, v, :]
        if users_covered_genre is None:
            users_idx = users_features[:, -1].int() + torch.arange(
                0,
                self.batch_size * self.num_users,
                self.num_users,
                device=self.adj.device,
            )
            users_covered_genre = self.users.reshape(
                self.batch_size * self.num_users, -1
            ).index_select(0, users_idx)
        covered_genres = (
            (self.u_features + users_covered_genre.unsqueeze(1)).reshape(
                self.batch_size, self.u_size + 1, -1
            )
            > 0
        ).float()
        prev = (
            (users_features[:, : self.num_genres] * users_covered_genre)
            .sum(1)
            .unsqueeze(1)
        )
        size = (
            (covered_genres * users_features[:, : self.num_genres].unsqueeze(1))
            .reshape(self.batch_size, self.u_size + 1, -1)
            .sum(-1)
        )
        curr_weights = size - prev
        return curr_weights

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
            self.matched_nodes.squeeze(1) + mask > 0
        ).long()  # Hacky way to return bool or uint8 depending on pytorch version
