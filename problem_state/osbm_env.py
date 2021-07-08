import torch
from typing import NamedTuple
from torch_geometric.utils import to_dense_adj, sort_edge_index

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
    opts: dict
    idx: torch.Tensor
    min_sol: torch.Tensor
    max_sol: torch.Tensor
    sum_sol_sq: torch.Tensor
    num_skip: torch.Tensor

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
        idx = torch.arange(adj.shape[1], device=opts.device)
        v_features = v_features[:, idx, :].view(v_features.size())

        # permute the nodes for data
        #        if "supervised" not in opts.model and not opts.eval_only:
        #            idx = torch.randperm(adj.shape[1], device=opts.device)

        # adj = adj[:, idx, :].view(adj.size())
        v_features = v_features[:, idx, :].view(v_features.size())
        adj[adj == 0.0] = -1.0
        weights = torch.tensor([], device=opts.device)
        edge_index, _ = sort_edge_index(input.edge_index)
        input.edge_index = edge_index
        input.weight = weights
        return StateOSBM(
            graphs=input,
            adj=adj,
            u_size=u_size,
            v_size=v_size,
            batch_size=batch_size,
            matched_nodes=(
                torch.zeros(batch_size, u_size + 1, device=input.batch.device,)
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
            min_sol=torch.zeros(batch_size, 1, device=input.batch.device),
            max_sol=torch.zeros(batch_size, 1, device=input.batch.device),
            sum_sol_sq=torch.zeros(batch_size, 1, device=input.batch.device),
            num_skip=torch.zeros(batch_size, 1, device=input.batch.device),
            size=torch.zeros(batch_size, 1, device=input.batch.device),
            users=torch.zeros(
                batch_size, num_users, num_genres, device=input.batch.device
            ),
            i=u_size + 1,
            u_features=u_features,
            v_features=v_features,
            num_genres=num_genres,
            num_users=num_users,
            opts=opts,
            idx=idx,
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.size

    def update(self, selected):
        # Update the state
        v = self.idx[self.i - (self.u_size + 1)]
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
        curr_weights = self.adj[:, v, :].float()
        selected_weights = curr_weights.gather(1, selected)
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
        total_weights = self.size + selected_weights
        # indices = torch.arange(0, self.num_genres, device=selected.device).unsqueeze(0).expand(self.batch_size, self.num_genres)
        updated_users_selected_genre = self.users.reshape(
            self.batch_size * self.num_users, -1
        ).index_copy(0, users_idx, s)
        nodes = self.matched_nodes.squeeze(1).scatter_(-1, selected, 1)
        hist_sum = self.hist_sum + curr_weights.unsqueeze(1)
        hist_sum_sq = self.hist_sum_sq + curr_weights.unsqueeze(1) ** 2
        hist_deg = self.hist_deg + (curr_weights.unsqueeze(1) != 0).float()
        print(self.size)
        return self._replace(
            matched_nodes=nodes,
            size=total_weights,
            i=self.i + 1,
            hist_sum=hist_sum,
            hist_sum_sq=hist_sum_sq,
            hist_deg=hist_deg,
            users=updated_users_selected_genre.reshape(
                self.batch_size, self.num_users, -1
            ),
            num_skip=num_skip,
            max_sol=max_sol,
            min_sol=min_sol,
            sum_sol_sq=sum_sol_sq,
        )

    def get_current_weights(self, mask, users_covered_genre=None):
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
        self.add_weights(curr_weights, mask)
        return curr_weights

    def add_weights(self, curr_weights, mask):
        v = self.idx[self.i - (self.u_size + 1)]
        w = curr_weights.clone()
        w[self.adj[:, v, :] == -1] = -1.0

        self.adj[:, v, :] = w

        return

    def get_graph_weights(self):
        graph_weights = torch.cat(
            (
                self.adj[:, :, :].transpose(1, 2).reshape(self.batch_size, -1),
                self.adj[:, :, :].reshape(self.batch_size, -1),
            ),
            dim=1,
        ).flatten()
        graph_weights = graph_weights[graph_weights != -1]

        return graph_weights

    def all_finished(self):
        # Exactly v_size steps
        return (self.i - (self.u_size + 1)) >= self.v_size

    def get_current_node(self):
        v = self.i - (self.u_size + 1)
        return self.idx[v]

    def get_curr_state(self, model):
        mask = self.get_mask().float()
        opts = self.opts
        i = self.i - self.u_size
        w = self.adj[:, self.idx[i - 1], :].float()
        s = None
        if model == "ff":
            s = torch.cat((w, mask.float()), dim=1)
        elif model == "inv-ff":
            w1 = w.clone()
            w1[w1 == -1] = 0.0
            mean_w = w1.mean(1)[:, None, None].repeat(1, self.u_size + 1, 1)
            s = w.reshape(self.batch_size, self.u_size + 1, 1)
            s[:, 0, :], mean_w[:, 0, :] = -1.0, -1.0

            s = torch.cat((s, mean_w,), dim=2,)
        elif model == "ff-hist" or model == "ff-supervised":
            w = w.clone()
            h_mean = self.hist_sum.squeeze(1) / i
            h_var = ((self.hist_sum_sq - ((self.hist_sum ** 2) / i)) / i).squeeze(1)
            h_mean_degree = self.hist_deg.squeeze(1) / i
            ind = torch.ones(self.batch_size, 1, device=opts.device) * i / self.v_size
            curr_sol_size = i - self.num_skip
            var_sol = (
                self.sum_sol_sq - ((self.size ** 2) / curr_sol_size)
            ) / curr_sol_size
            mean_sol = self.size / curr_sol_size
            matched_ratio = self.matched_nodes.sum(1) / self.u_size
            s = torch.cat(
                (
                    w,
                    mask,
                    h_mean,
                    h_var,
                    h_mean_degree,
                    self.size / self.u_size,
                    ind.float(),
                    mean_sol,
                    var_sol,
                    self.num_skip / i,
                    self.max_sol,
                    self.min_sol,
                    matched_ratio.unsqueeze(1),
                ),
                dim=1,
            ).float()
        elif model == "inv-ff-hist":
            w1 = w.clone()
            w1[w1 == -1] = 0.0
            mean_w = w1.mean(1)[:, None, None].repeat(1, self.u_size + 1, 1)
            s = w.reshape(self.batch_size, self.u_size + 1, 1)
            h_mean = self.hist_sum / i
            h_var = (self.hist_sum_sq - ((self.hist_sum ** 2) / i)) / i
            h_mean_degree = self.hist_deg / i
            idx = (
                torch.ones(self.batch_size, 1, 1, device=opts.device) * i / self.v_size
            )
            curr_sol_size = i - self.num_skip
            var_sol = (
                self.sum_sol_sq - ((self.size ** 2) / curr_sol_size)
            ) / curr_sol_size
            mean_sol = self.size / curr_sol_size
            matched_ratio = self.matched_nodes.sum(1).unsqueeze(1) / self.u_size
            s = torch.cat(
                (
                    s,
                    mask.reshape(-1, self.u_size + 1, 1),
                    mean_w,
                    h_mean.transpose(1, 2),
                    h_var.transpose(1, 2),
                    h_mean_degree.transpose(1, 2),
                    idx.repeat(1, self.u_size + 1, 1),
                    self.size.unsqueeze(2).repeat(1, self.u_size + 1, 1) / self.u_size,
                    mean_sol.unsqueeze(2).repeat(1, self.u_size + 1, 1),
                    var_sol.unsqueeze(2).repeat(1, self.u_size + 1, 1),
                    self.num_skip.unsqueeze(2).repeat(1, self.u_size + 1, 1) / i,
                    self.max_sol.unsqueeze(2).repeat(1, self.u_size + 1, 1),
                    self.min_sol.unsqueeze(2).repeat(1, self.u_size + 1, 1),
                    matched_ratio.unsqueeze(2).repeat(1, self.u_size + 1, 1),
                ),
                dim=2,
            ).float()

        return s, mask

    def get_node_features(self):

        num_v = self.i - self.u_size
        batch_size = self.batch_size
        incoming_node_features = self.v_features[:, :num_v, :-1].reshape(
            batch_size * num_v, -1
        )  # Collecting node features up until the ith incoming node
        fixed_node_feature = torch.cat(
            (self.u_features, self.matched_nodes.unsqueeze(2)), dim=2
        ).reshape(batch_size * (self.u_size + 1), -1)

        node_features = torch.cat(
            (
                fixed_node_feature.reshape(self.batch_size, -1),
                incoming_node_features.reshape(self.batch_size, -1),
            ),
            dim=1,
        )
        return node_features.float()

    def get_mask(self):
        """
        Returns a mask vector which includes only nodes in U that can matched.
        That is, neighbors of the incoming node that have not been matched already.
        """
        v = self.idx[self.i - (self.u_size + 1)]
        mask = (self.adj[:, v, :] == -1).float()
        mask[:, 0] = 0.0
        self.matched_nodes[
            :, 0
        ] = 0  # node that represents not being matched to anything can be matched to more than once
        return (
            self.matched_nodes + mask > 0
        ).long()  # Hacky way to return bool or uint8 depending on pytorch version
