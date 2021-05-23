from torch_geometric.data import Dataset
import torch
import os
import pickle
from problem_state.edge_obm_env import StateEdgeBipartite
from data.generate_data import generate_edge_obm_data_geometric


class EdgeBipartite(object):

    NAME = "e-obm"

    @staticmethod
    def make_dataset(*args, **kwargs):
        return EdgeBipartiteDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateEdgeBipartite.initialize(*args, **kwargs)


class EdgeBipartiteDataset(Dataset):
    def __init__(
        self, dataset, size, problem, seed, opts, transform=None, pre_transform=None
    ):
        super(EdgeBipartiteDataset, self).__init__(None, transform, pre_transform)
        # self.data_set = dataset
        # self.optimal_size = torch.load("{}/optimal_match.pt".format(self.data_set))
        self.problem = problem
        if dataset is not None:
            # self.optimal_size = torch.load("{}/optimal_match.pt".format(dataset))
            self.data_set = dataset
        else:
            # If no filename is specified generated data for edge obm probelm
            D, optimal_size = generate_edge_obm_data_geometric(
                opts.u_size,
                opts.v_size,
                opts.weight_distribution,
                opts.weight_distribution_param,
                opts.graph_family_parameter,
                seed,
                opts.graph_family,
                None,
                size,
                False,
            )
            self.optimal_size = optimal_size
            self.data_set = D

        self.size = size

    def len(self):
        return self.size

    def get(self, idx):
        if type(self.data_set) == str:
            data = torch.load(self.data_set + "/data_{}.pt".format(idx))
        else:
            data = self.data_set[idx]
        return data
