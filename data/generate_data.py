import argparse
import os
import numpy as np
from data.data_utils import check_extension, save_dataset
import networkx as nx
from scipy.optimize import linear_sum_assignment
import torch
import pickle as pk
from tqdm import tqdm
from scipy.stats import powerlaw


def _add_nodes_with_bipartite_label(G, lena, lenb):
    """
    Helper for generate_ba_graph that initializes the initial empty graph with nodes
    """
    G.add_nodes_from(range(0, lena + lenb))
    b = dict(zip(range(0, lena), [0] * lena))
    b.update(dict(zip(range(lena, lena + lenb), [1] * lenb)))
    nx.set_node_attributes(G, b, "bipartite")
    return G


def generate_ba_graph(u, v, p, seed):
    """
    Genrates a graph using the preferential attachment scheme
    """
    np.random.seed(seed)

    G = nx.Graph()
    G = _add_nodes_with_bipartite_label(G, u, v)

    G.name = f"ba_random_graph({u},{v},{p})"

    deg = np.zeros(u)

    v1 = 0.0
    w = 0
    while v1 < v:
        d = np.random.binomial(u, float(p) / v)

        while w < d:
            p1 = deg + 1
            p1 = p1 / np.sum(p1)
            f = np.random.choice(np.arange(0, u), p=list(p1))
            if (f, v) not in G.edges:
                G.add_edge(f, v)
                deg[f] += 1
                w += 1

        v1 += 1
    return G


def generate_obm_data(
    u_size,
    v_size,
    graph_family_parameter,
    seed,
    graph_family,
    dataset_folder,
    dataset_size,
    save_data,
):
    """
    Generates graphs using the ER/BA scheme

    """
    G, M = [], []
    if graph_family == "er":
        g = nx.bipartite.random_graph
    if graph_family == "ba":
        g = generate_ba_graph
    for i in tqdm(range(dataset_size)):
        g1 = g(u_size, v_size, p=graph_family_parameter, seed=seed + i)

        cost = nx.bipartite.biadjacency_matrix(
            g1, range(0, u_size), range(u_size, u_size + v_size)
        ).toarray()
        # d_old = np.array(sorted(g1.degree))[u_size:, 1]

        # c = nx.convert_matrix.to_numpy_array(g1, s)

        g1.add_node(
            -1, bipartite=0
        )  # add extra node in U that represents not matching the current node to anything
        g1.add_edges_from(list(zip([-1] * v_size, range(u_size, u_size + v_size))))
        s = sorted(list(g1.nodes))
        m = 1 - nx.convert_matrix.to_numpy_array(g1, s)

        # ordered_m = np.take(np.take(m, order, axis=1), order, axis=0)
        if save_data:
            torch.save(torch.tensor(m), "{}/graphs/{}.pt".format(dataset_folder, i))
        else:
            G.append(m.tolist())
        i1, i2 = linear_sum_assignment(cost, maximize=True)
        M.append(cost[i1, i2].sum())
    if save_data:
        torch.save(torch.tensor(M), "{}/optimal_match.pt".format(dataset_folder))
    return (
        torch.tensor(G),
        torch.tensor(M),
    )


def generate_weights(distribution, u_size, v_size, parameters, g1):
    if distribution == "uniform":
        weights = nx.bipartite.biadjacency_matrix(
            g1, range(0, u_size), range(u_size, u_size + v_size)
        ).toarray() * np.random.randint(
            int(parameters[0]), int(parameters[1]), (u_size, v_size)
        )
        w = torch.cat(
            (torch.zeros(v_size, 1).long(), torch.tensor(weights).T.long()), 1
        )
    elif distribution == "normal":
        weights = nx.bipartite.biadjacency_matrix(
            g1, range(0, u_size), range(u_size, u_size + v_size)
        ).toarray() * (
            np.abs(
                np.random.normal(
                    int(parameters[0]), int(parameters[1]), (u_size, v_size)
                )
            )
            + 5
        )  # to make sure no edge has weight zero
        w = torch.cat(
            (torch.zeros(v_size, 1).long(), torch.tensor(weights).T.long()), 1
        )
    elif distribution == "power":
        weights = nx.bipartite.biadjacency_matrix(
            g1, range(0, u_size), range(u_size, u_size + v_size)
        ).toarray() * (
            powerlaw.rvs(
                int(parameters[0]),
                int(parameters[1]),
                int(parameters[2]),
                (u_size, v_size),
            )
            + 5
        )  # to make sure no edge has weight zero
        w = torch.cat(
            (torch.zeros(v_size, 1).long(), torch.tensor(weights).T.long()), 1
        )

    return weights, w


def generate_edge_obm_data(
    u_size,
    v_size,
    weight_distribution,
    weight_param,
    graph_family_parameter,
    seed,
    graph_family,
    dataset_folder,
    dataset_size,
    save_data,
):
    """
    Generates edge weighted bipartite graphs using the ER/BA schemes

    Supports unifrom, normal, and power distributions.
    """
    D, M = [], []
    if graph_family == "er":
        g = nx.bipartite.random_graph
    if graph_family == "ba":
        g = generate_ba_graph
    for i in tqdm(range(dataset_size)):
        g1 = g(u_size, v_size, p=graph_family_parameter, seed=seed + i)
        # d_old = np.array(sorted(g1.degree))[u_size:, 1]
        weights, w = generate_weights(
            weight_distribution, u_size, v_size, weight_param, g1
        )
        s = sorted(list(g1.nodes))
        # c = nx.convert_matrix.to_numpy_array(g1, s)

        g1.add_node(
            -1, bipartite=0
        )  # add extra node in U that represents not matching the current node to anything
        g1.add_edges_from(list(zip([-1] * v_size, range(u_size, u_size + v_size))))

        s = sorted(list(g1.nodes))
        m = 1 - nx.convert_matrix.to_numpy_array(g1, s)
        if save_data:
            torch.save(
                [w], "{}/graphs/{}.pt".format(dataset_folder, i),
            )
        else:
            D.append([torch.tensor(m).clone(), torch.tensor(w).clone()])
        # ordered_m = np.take(np.take(m, order, axis=1), order, axis=0)
        i1, i2 = linear_sum_assignment(weights, maximize=True)
        M.append(weights[i1, i2].sum())
    if save_data:
        torch.save(torch.tensor(M), "{}/optimal_match.pt".format(dataset_folder))

    return (
        D,
        torch.tensor(M),
    )


def generate_high_entropy_obm_data(opts):
    """
    Generates data from a range of graph family parameters instead of just one.
    Stores each unique dataset seperately.
    Used for model evaluation.
    """
    seed = opts.seed
    min_p, max_p = float(opts.parameter_range[0]), float(opts.parameter_range[1])
    for i, j in enumerate(
        np.arange(min_p, max_p, (min_p + max_p) / opts.num_eval_datasets)
    ):
        print(i, j)
        dataset_folder = opts.dataset_folder + "/eval{}".format(i)
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
            os.makedirs("{}/graphs".format(dataset_folder))
        generate_obm_data(
            opts.u_size,
            opts.v_size,
            j,
            seed,
            opts.graph_family,
            dataset_folder,
            opts.dataset_size,
            True,
        )
        seed += (
            opts.dataset_size + 1
        )  # Use different starting seed to make sure datasets do not overlap
    return


def generate_bipartite_data(
    dataset_size, u_size, v_size, num_edges, future_edge_weight, weights_range
):
    """
    Generate random graphs using gnmk_random_graph

    This is the old implementation. DO NOT USE.
    """
    G, D, E, W, M = [], [], [], [], []
    for i in range(dataset_size):
        g1 = nx.bipartite.gnmk_random_graph(u_size, v_size, num_edges)
        # d_old = np.array(sorted(g1.degree))[u_size:, 1]
        c = nx.bipartite.biadjacency_matrix(
            g1, range(0, u_size), range(u_size, u_size + v_size)
        ).toarray() * np.random.randint(
            weights_range[0], weights_range[1], (u_size, v_size)
        )
        f = torch.cat((torch.ones(v_size, 1).long(), torch.tensor(c).T), 1).flatten()
        w = torch.cat((torch.zeros(v_size, 1).long(), torch.tensor(c).T), 1).flatten()
        t = f.nonzero().T
        g1.add_node(-1, bipartite=0)
        g1.add_edges_from(list(zip([-1] * v_size, range(u_size, u_size + v_size))))
        d = np.array(sorted(g1.degree))[u_size + 1 :, 1]
        l1 = nx.line_graph(g1)
        # w = np.random.randint(weights_range[0], weights_range[1], num_edges)
        # w = c[c.nonzero()]
        # w[
        #     np.int_(np.sum(np.triu(np.ones((v_size, v_size))).T * d, axis=1) - 1)
        # ] = future_edge_weight
        # w = np.insert(w, np.cumsum(d_old) - d_old, future_edge_weight)
        # w = np.insert(w,np.sum(np.triu(np.ones((v_size,v_size))).T * d, axis=1), future_edge_weight)
        # add negative adjacency matrix and edge weights
        # order = np.argsort(np.array(l1.nodes)[:, 1], axis=None)
        s = sorted(list(l1.nodes), key=lambda a: a[0])
        s = sorted(s, key=lambda a: a[1])
        m = nx.convert_matrix.to_numpy_array(l1, s)
        adj = torch.ones((u_size + 1) * v_size, (u_size + 1) * v_size)
        temp = adj[t[0]]
        temp[:, t[0]] = torch.tensor(1 - m).float()
        adj[t[0]] = temp
        # ordered_m = np.take(np.take(m, order, axis=1), order, axis=0)
        G.append(adj.tolist())
        W.append(w.tolist())
        D.append(list(d))
        E.append(s)
        i1, i2 = linear_sum_assignment(c, maximize=True)
        M.append(c[i1, i2].sum())
    return (
        torch.tensor(G),
        torch.tensor(W),
        torch.tensor(D),
        torch.tensor(E),
        torch.tensor(M),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--problem",
        type=str,
        default="obm",
        help="Problem: 'obm', 'e-obm', 'adwords' or 'displayads'",
    )
    parser.add_argument(
        "--weight_distribution",
        type=str,
        default="uniform",
        help="Distributions to generate for problem, default 'uniform' ",
    )
    parser.add_argument(
        "--weight_distribution_param",
        nargs="+",
        default="5 4000",
        help="parameters of weight distribtion ",
    )
    parser.add_argument(
        "--max_weight", type=int, default=4000, help="max weight in graph",
    )

    parser.add_argument(
        "--dataset_size", type=int, default=100, help="Size of the dataset"
    )
    # parser.add_argument(
    #     "--save_format", type=str, default='train', help="Save a dataset as one pickle file or one file for each example (for training)"
    # )
    parser.add_argument(
        "--dataset_folder", type=str, default="dataset/train", help="dataset folder"
    )
    parser.add_argument(
        "--u_size", type=int, default=10, help="Sizes of U set (default 10 by 10)",
    )
    parser.add_argument(
        "--v_size", type=int, default=10, help="Sizes of V set (default 10 by 10)",
    )
    parser.add_argument(
        "--graph_family",
        type=str,
        default="er",
        help="family of graphs to generate (er, ba, etc)",
    )
    parser.add_argument(
        "--graph_family_parameter",
        type=float,
        help="parameter of the graph family distribution",
    )

    parser.add_argument(
        "--parameter_range",
        nargs="+",
        help="range of graph family parameters to generate datasets for",
    )

    parser.add_argument(
        "--num_eval_datasets",
        type=int,
        default=5,
        help="number of eval datasets to generate for a given range of family parameters",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Set true to generate datasets for evaluation of model",
    )
    parser.add_argument("--seed", type=int, default=2020, help="Intitial Random seed")

    opts = parser.parse_args()

    # assert opts.filename is None or (
    #     len(opts.problems) == 1 and len(opts.graph_sizes) == 1
    # ), "Can only specify filename when generating a single dataset"

    # assert opts.f or not os.path.isfile(
    #     check_extension(filename)
    # ), "File already exists! Try running with -f option to overwrite."

    if not os.path.exists(opts.dataset_folder):
        os.makedirs(opts.dataset_folder)
        if not opts.eval:
            os.makedirs("{}/graphs".format(opts.dataset_folder))
    np.random.seed(opts.seed)

    if opts.eval:
        generate_high_entropy_obm_data(opts)
    elif opts.problem == "obm":
        dataset = generate_obm_data(
            opts.u_size,
            opts.v_size,
            opts.graph_family_parameter,
            opts.seed,
            opts.graph_family,
            opts.dataset_folder,
            opts.dataset_size,
            True,
        )
    elif opts.problem == "e-obm":
        dataset = generate_edge_obm_data(
            opts.u_size,
            opts.v_size,
            opts.weight_distribution,
            opts.weight_distribution_param,
            opts.graph_family_parameter,
            opts.seed,
            opts.graph_family,
            opts.dataset_folder,
            opts.dataset_size,
            True,
        )
    elif opts.problem == "adwords":
        pass
    elif opts.problem == "displayads":
        pass
    else:
        assert False, "Unknown problem: {}".format(opts.problem)
    # if opts.save_format != 'train':
    #     save_dataset(
    #         dataset, "{}by{}-{}.pkl".format(opts.u_size, opts.v_size, opts.graph_family)
    #     )
