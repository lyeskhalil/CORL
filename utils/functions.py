import torch
from problem_state.obm_dataset import Bipartite
from problem_state.edge_obm_dataset import EdgeBipartite
from problem_state.osbm_dataset import OSBM
from problem_state.adwords_dataset import AdwordsBipartite
import csv


def load_problem(name):

    problem = {
        "obm": Bipartite,
        "e-obm": EdgeBipartite,
        "osbm": OSBM,
        "adwords": AdwordsBipartite,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def torch_load_cpu(load_path):
    return torch.load(
        load_path, map_location=lambda storage, loc: storage
    )  # Load on CPU


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    elif isinstance(var, list):
        return list(move_to(v, device) for v in var)
    return var.to(device)


def random_max(input):
    """
    Return max element with random tie breaking
    """
    max_w, _ = torch.max(input, dim=1)
    max_filter = (input == max_w[:, None]).float()
    selected = (max_filter / (max_filter.sum(dim=1)[:, None])).multinomial(1)

    return selected


def random_min(input):
    """
    Return min element with random tie breaking
    """
    min_w, _ = torch.min(input, dim=1)
    max_filter = (input == min_w[:, None]).float()
    selected = (max_filter / (max_filter.sum(dim=1)[:, None])).multinomial(1)

    return selected


def get_best_t(model, opts):
    best_params = None
    best_r = 0
    graph_family = (
        opts.graph_family if opts.graph_family != "gmission-perm" else "gmission"
    )
    with open(
        f"val_rewards_{model}_{opts.u_size}_{opts.v_size}_{graph_family}_{opts.graph_family_parameter}.csv"
    ) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for line in csv_reader:
            if abs(float(line[-1])) > best_r:
                best_params = float(line[0])
                best_r = abs(float(line[-1]))
    return best_params
