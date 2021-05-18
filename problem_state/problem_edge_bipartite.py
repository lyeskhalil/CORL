from torch.utils.data import Dataset
from torch_geometric.data import Dataset as geometricDataset
import torch
import os
import pickle
from problem_state.edge_obm_state import StateEdgeBipartite
from problem_state.edge_obm_old import StateEdgeBipartite as StateEdgeBipartiteGeo
from data.generate_data import generate_edge_obm_data, generate_edge_obm_data_geometric


class EdgeBipartite(object):

    NAME = "e-obm"

    @staticmethod
    def make_dataset(*args, **kwargs):
        return EdgeBipartiteDataset1(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateEdgeBipartiteGeo.initialize(*args, **kwargs)


class EdgeBipartiteDataset1(geometricDataset):
    def __init__(
        self, dataset, size, problem, seed, opts, transform=None, pre_transform=None
    ):
        super(EdgeBipartiteDataset1, self).__init__(None, transform, pre_transform)
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


class EdgeBipartiteDataset(Dataset):
    def __init__(self, dataset, size, problem, seed, opts):
        super(EdgeBipartiteDataset, self).__init__()

        # self.data_set = dataset
        # self.optimal_size = torch.load("{}/optimal_match.pt".format(self.data_set))
        self.problem = problem
        if dataset is not None:
            self.optimal_size = torch.load("{}/optimal_match.pt".format(dataset))
            self.data_set = dataset
        else:
            # If no filename is specified generated data for edge obm probelm
            D, optimal_size = generate_edge_obm_data(
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

    def __getitem__(self, index):

        # Load data and get label
        if type(self.data_set) == str:
            X = torch.load("{}/graphs/{}.pt".format(self.data_set, index))
        else:
            X = self.data_set[index]
        Y = self.optimal_size[index]
        return X, Y

    def __len__(self):
        return self.size
