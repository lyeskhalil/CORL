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
    opts: dict
    idx: torch.Tensor

    @staticmethod
    def initialize(
        input, u_size, v_size, opts,
    ):
        graph_size = u_size + v_size + 1
        batch_size = int(input.batch.size(0) / graph_size)
        # print(batch_size, input.batch.size(0), graph_size)
        adj = to_dense_adj(
            input.edge_index, input.batch, input.weight.unsqueeze(1)
        ).squeeze(-1)
        adj = adj[:, u_size + 1 :, : u_size + 1]

        # permute the nodes for data
        idx = torch.arange(adj.shape[1], device=opts.device)
        # if "supervised" not in opts.model and not opts.eval_only:
        #     idx = torch.randperm(adj.shape[1], device=opts.device)
        #     adj = adj[:, idx, :].view(adj.size())

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
            opts=opts,
            idx=idx,
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.size

    def get_current_weights(self, mask):
        return self.adj[:, 0, :].float()

    def get_graph_weights(self):
        return self.graphs.weight

    def update(self, selected):
        # Update the state
        nodes = self.matched_nodes.scatter_(-1, selected, 1)
        nodes[
            :, 0
        ] = 0  # node that represents not being matched to anything can be matched to more than once
        w = self.adj[:, 0, :].clone()
        selected_weights = w.gather(1, selected).to(self.adj.device)
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

        hist_sum = self.hist_sum + w.unsqueeze(1)
        hist_sum_sq = self.hist_sum_sq + w.unsqueeze(1) ** 2
        hist_deg = self.hist_deg + (w.unsqueeze(1) != 0).float()
        hist_deg[:, :, 0] = float(self.i - self.u_size)
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
        return 0

    def get_curr_state(self, model):
        mask = self.get_mask()
        opts = self.opts
        i = self.i - self.u_size
        w = self.adj[:, 0, :].float()
        s = None
        if model == "ff":
            s = torch.cat((w, mask.float()), dim=1)

        elif model == "inv-ff":
            w = w.clone().float()
            deg = (w != 0).float().sum(1)
            mean_w = w.sum(1) / deg
            mean_w = mean_w[:, None, None].repeat(1, self.u_size + 1, 1)
            fixed_node_identity = torch.zeros(
                self.batch_size, self.u_size + 1, 1, device=opts.device
            ).float()
            fixed_node_identity[:, 0, :] = 1.0
            s = w.reshape(self.batch_size, self.u_size + 1, 1)
            s = torch.cat((fixed_node_identity, s, mean_w,), dim=2,)

        elif model == "ff-hist" or model == "ff-supervised":
            w = w.clone()
            (
                h_mean,
                h_var,
                h_mean_degree,
                ind,
                matched_ratio,
                var_sol,
                mean_sol,
                n_skip,
            ) = self.get_hist_features()
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
                    n_skip,
                    self.max_sol,
                    self.min_sol,
                    matched_ratio.unsqueeze(1),
                ),
                dim=1,
            ).float()

        elif model == "inv-ff-hist" or model == "gnn-simp-hist":
            deg = (w != 0).float().sum(1)
            mean_w = w.sum(1) / deg
            mean_w = mean_w[:, None, None].repeat(1, self.u_size + 1, 1)
            s = w.reshape(self.batch_size, self.u_size + 1, 1)
            (
                h_mean,
                h_var,
                h_mean_degree,
                ind,
                matched_ratio,
                var_sol,
                mean_sol,
                n_skip,
            ) = self.get_hist_features()
            available_ratio = (deg.unsqueeze(1)) / (self.u_size)
            fixed_node_identity = torch.zeros(
                self.batch_size, self.u_size + 1, 1, device=opts.device
            ).float()
            fixed_node_identity[:, 0, :] = 1.0
            s = torch.cat(
                (
                    s,
                    mask.reshape(-1, self.u_size + 1, 1),
                    mean_w,
                    h_mean.transpose(1, 2),
                    h_var.transpose(1, 2),
                    h_mean_degree.transpose(1, 2),
                    ind.unsqueeze(2).repeat(1, self.u_size + 1, 1),
                    self.size.unsqueeze(2).repeat(1, self.u_size + 1, 1) / self.u_size,
                    mean_sol.unsqueeze(2).repeat(1, self.u_size + 1, 1),
                    var_sol.unsqueeze(2).repeat(1, self.u_size + 1, 1),
                    n_skip.unsqueeze(2).repeat(1, self.u_size + 1, 1) / i,
                    self.max_sol.unsqueeze(2).repeat(1, self.u_size + 1, 1),
                    self.min_sol.unsqueeze(2).repeat(1, self.u_size + 1, 1),
                    matched_ratio.unsqueeze(2).repeat(1, self.u_size + 1, 1),
                    available_ratio.unsqueeze(2).repeat(1, self.u_size + 1, 1),
                    fixed_node_identity,
                ),
                dim=2,
            ).float()

        return s, mask

    def get_node_features(self):
        step_size = self.i + 1
        batch_size = self.batch_size
        incoming_node_features = (
            torch.cat(
                (torch.ones(step_size - self.u_size - 1, device=self.adj.device) * 2,)
            )
            .unsqueeze(0)
            .expand(batch_size, step_size - self.u_size - 1)
        ).float()  # Collecting node features up until the ith incoming node

        future_node_feature = torch.ones(batch_size, 1, device=self.adj.device) * -1.0
        fixed_node_feature = self.matched_nodes[:, 1:]
        node_features = torch.cat(
            (future_node_feature, fixed_node_feature, incoming_node_features), dim=1
        ).reshape(batch_size, step_size)

        return node_features

    def get_hist_features(self):
        i = self.i - (self.u_size + 1)
        if i != 0:
            h_mean = self.hist_sum / i
            h_var = (self.hist_sum_sq - ((self.hist_sum ** 2) / i)) / i
            h_mean_degree = self.hist_deg / i
            ind = (
                torch.ones(self.batch_size, 1, device=self.opts.device)
                * i
                / self.v_size
            )
            curr_sol_size = i - self.num_skip
            var_sol = (
                self.sum_sol_sq - ((self.size ** 2) / curr_sol_size)
            ) / curr_sol_size
            mean_sol = self.size / curr_sol_size
            var_sol[curr_sol_size == 0.0] = 0.0
            mean_sol[curr_sol_size == 0.0] = 0.0
            matched_ratio = self.matched_nodes.sum(1).unsqueeze(1) / self.u_size
            n_skip = self.num_skip / i
        else:
            (
                h_mean,
                h_var,
                h_mean_degree,
                ind,
                matched_ratio,
                var_sol,
                mean_sol,
                n_skip,
            ) = (
                self.hist_sum * 0.0,
                self.hist_sum * 0.0,
                self.hist_sum * 0.0,
                self.size * 0.0,
                self.num_skip * 0.0,
                self.size * 0.0,
                self.size * 0.0,
                self.size * 0.0,
            )

        return (
            h_mean,
            h_var,
            h_mean_degree,
            ind,
            matched_ratio,
            var_sol,
            mean_sol,
            n_skip,
        )

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
