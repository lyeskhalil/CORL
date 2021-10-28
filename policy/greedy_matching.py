import torch
from torch import nn
from torch_geometric.utils import subgraph, to_networkx
from networkx.algorithms.matching import max_weight_matching
from torch_geometric.data import Data


class GreedyMatching(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        problem,
        opts,
        tanh_clipping=None,
        mask_inner=None,
        mask_logits=None,
        n_encode_layers=None,
        normalization=None,
        checkpoint_encoder=False,
        shrink_size=None,
        num_actions=None,
        n_heads=None,
        encoder=None,
    ):
        super(GreedyMatching, self).__init__()
        self.decode_type = None
        self.problem = problem
        self.model_name = "greedy-m"

    def forward(self, x, opts, optimizer, baseline, return_pi=False):
        state = self.problem.make_state(x, opts.u_size, opts.v_size, opts)
        t = opts.threshold
        sequences = []
        batch_size = opts.batch_size
        graph_size = opts.u_size + opts.v_size + 1
        i = 1
        while not (state.all_finished()):
            step_size = state.i + 1
            mask = state.get_mask()
            state.get_current_weights(mask).clone()
            mask = state.get_mask()
            if i <= int(
                t * opts.v_size
            ):  # Skip matching if less than t fraction of V nodes arrived
                selected = torch.zeros(
                    batch_size, dtype=torch.int64, device=opts.device
                )
                state = state.update(selected[:, None])
                sequences.append(selected)
                i += 1
                continue
            nodes = torch.cat(
                (
                    torch.arange(1, opts.u_size + 1, device=opts.device),
                    state.idx[:i] + opts.u_size + 1,
                )
            )
            subgraphs = (
                (nodes.unsqueeze(0).expand(batch_size, step_size - 1))
                + torch.arange(
                    0, batch_size * graph_size, graph_size, device=opts.device
                ).unsqueeze(1)
            ).flatten()  # The nodes of the current subgraphs
            graph_weights = state.get_graph_weights()
            edge_i, weights = subgraph(
                subgraphs,
                state.graphs.edge_index,
                graph_weights.unsqueeze(1),
                relabel_nodes=True,
            )
            match_sol = torch.tensor(
                list(
                    max_weight_matching(
                        to_networkx(
                            Data(subgraphs, edge_index=edge_i, edge_attr=weights),
                            to_undirected=True,
                        )
                    )
                ),
                device=opts.device,
            )

            edges_sol = torch.cat(
                (
                    torch.arange(0, opts.u_size, device=opts.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                    .unsqueeze(2),
                    torch.ones(batch_size, opts.u_size, 1) * (opts.u_size + i - 1),
                ),
                dim=2,
            )
            match_sol = match_sol.sort(dim=-1)[0]
            offset = torch.arange(0, batch_size * (opts.u_size + i), opts.u_size + i)[
                :, None, None
            ]
            edges_sol = edges_sol + offset
            in_sol = edges_sol.unsqueeze(1) == match_sol[None, :, :, None].expand(
                batch_size, -1, -1, opts.u_size
            ).transpose(-1, -2)
            in_sol = in_sol * (1 - mask[:, None, 1:, None])
            in_sol = (
                in_sol.prod(-1).sum(-1).float().unsqueeze(1)
                * match_sol[None, :, :].transpose(1, 2)
            ).sum(-1)
            skip_sol = in_sol.sum(-1)
            in_sol = in_sol[:, 0] - offset.reshape(-1)
            in_sol[skip_sol == 0] = -1
            selected = (in_sol + 1).type(torch.int64)
            # print(selected, match_sol)
            state = state.update(selected[:, None])

            sequences.append(selected)
            i += 1
        if return_pi:
            return -state.size, None, torch.stack(sequences, 1), None
        return -state.size, torch.stack(sequences, 1), None

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
