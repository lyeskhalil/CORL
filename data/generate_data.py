import argparse
import os
from networkx.classes.function import number_of_edges
import numpy as np
from data.data_utils import check_extension, save_dataset
import networkx as nx
from scipy.optimize import linear_sum_assignment
import torch
import pickle as pk
from tqdm import tqdm
from scipy.stats import powerlaw
import torch_geometric
import math

# from torch_geometric.utils import from_networkx

gMission_edges = "data/edges.txt"

gMission_tasks = "data/tasks.txt"


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

    edges = nx.bipartite.biadjacency_matrix(
        g1, range(0, u_size), range(u_size, u_size + v_size)
    ).toarray()  # U by V array of adjacacy matrix

    if distribution == "uniform":
        weights = edges * np.random.randint(
            int(parameters[0]), int(parameters[1]), (u_size, v_size)
        )

    elif distribution == "normal":
        weights = edges * (
            np.abs(
                np.random.normal(
                    int(parameters[0]), int(parameters[1]), (u_size, v_size)
                )
            )
            + 5
        )  # to make sure no edge has weight zero

    elif distribution == "power":
        weights = edges * (
            powerlaw.rvs(
                int(parameters[0]),
                int(parameters[1]),
                int(parameters[2]),
                (u_size, v_size),
            )
            + 5
        )  # to make sure no edge has weight zero
    elif distribution == "degree":
        graph = 10 * edges * edges.sum(axis=1).reshape(-1, 1)
        noise = np.random.randint(
            int(parameters[0]), int(parameters[1]), (u_size, v_size)
        )
        weights = np.where(graph, graph + noise, graph)
    w = torch.cat((torch.zeros(v_size, 1).long(), torch.tensor(weights).T.long()), 1)

    return weights, w


def from_networkx(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data["edge_index"] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data


def parse_gmission_dataset():
    f_edges = open(gMission_edges, "r")
    f_tasks = open(gMission_tasks, "r")
    edgeWeights = dict()
    edgeNumber = dict()
    count = 0
    for line in f_edges:
        vals = line.split(",")
        edgeWeights[vals[0]] = vals[1].split("\n")[0]
        edgeNumber[vals[0]] = count
        count += 1

    tasks = list()
    tasks_x = dict()
    tasks_y = dict()

    for line in f_tasks:
        vals = line.split(",")
        tasks.append(vals[0])
        tasks_x[vals[0]] = float(vals[2])
        tasks_y[vals[0]] = float(vals[3])

    return edgeWeights, tasks


def generate_gmission_graph(
    u, v, tasks, edges, workers, p, seed, weight_dist, weight_param, vary_fixed=False
):
    np.random.seed(seed)

    G = nx.Graph()
    G = _add_nodes_with_bipartite_label(G, u, v)

    G.name = f"gmission_random_graph({u},{v})"
    if vary_fixed:
        workers = list(np.random.randint(1, 533, size=u))
    availableWorkers = workers.copy()
    weights = []
    for i in range(v):
        sampledTask = np.random.choice(tasks)

        for w in range(len(availableWorkers)):
            worker = availableWorkers[w]
            edge = str(float(worker)) + ";" + str(float(sampledTask))

            if edge in edges and (w, i + u) not in G.edges:
                G.add_edge(w, i + u, weight=float(edges[edge]))
                weights.append(float(edges[edge]))
            else:
                weights.append(float(0))
    weights = np.array(weights).reshape(v, u).T
    w = np.delete(weights.flatten(), weights.flatten() == 0)
    return G, weights, w


def generate_weights_geometric(distribution, u_size, v_size, parameters, g1, seed):
    weights, w = 0, 0
    np.random.seed(seed)
    if distribution == "uniform":
        weights = nx.bipartite.biadjacency_matrix(
            g1, range(0, u_size), range(u_size, u_size + v_size)
        ).toarray() * np.random.uniform(
            int(parameters[0]), int(parameters[1]), (u_size, v_size)
        )
        w = torch.cat(
            (torch.zeros(v_size, 1).float(), torch.tensor(weights).T.float()), 1
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
            (torch.zeros(v_size, 1).float(), torch.tensor(weights).T.float()), 1
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
            (torch.zeros(v_size, 1).float(), torch.tensor(weights).T.float()), 1
        )
    elif distribution == "degree":
        weights = nx.bipartite.biadjacency_matrix(
            g1, range(0, u_size), range(u_size, u_size + v_size)
        ).toarray()
        graph = 10 * weights * weights.sum(axis=1).reshape(-1, 1)
        noise = np.random.randint(
            int(parameters[0]), int(parameters[1]), (u_size, v_size)
        )
        weights = np.where(graph, graph + noise, graph)
        w = torch.cat(
            (torch.zeros(v_size, 1).float(), torch.tensor(weights).T.float()), 1
        )
    elif distribution == "node-normal":
        adj = nx.bipartite.biadjacency_matrix(
            g1, range(0, u_size), range(u_size, u_size + v_size)
        ).toarray()
        mean = np.random.randint(int(parameters[0]), int(parameters[1]), (u_size, 1))
        variance = np.sqrt(
            np.random.randint(int(parameters[0]), int(parameters[1]), (u_size, 1))
        )
        weights = (
            np.abs(np.random.normal(0.0, 1.0, (u_size, v_size)) * variance + mean) + 5
        ) * adj
    elif distribution == "fixed-normal":
        adj = nx.bipartite.biadjacency_matrix(
            g1, range(0, u_size), range(u_size, u_size + v_size)
        ).toarray()
        mean = np.random.choice(np.arange(0, 100, 15), size=(u_size, 1))
        variance = np.sqrt(np.random.choice(np.arange(0, 100, 20), (u_size, 1)))
        weights = (
            np.abs(np.random.normal(0.0, 1.0, (u_size, v_size)) * variance + mean) + 5
        ) * adj

    w = np.delete(weights.flatten(), weights.flatten() == 0)
    return weights, w


def generate_er_graph(
    u, v, tasks, edges, workers, p, seed, weight_distribution, weight_param, vary_fixed=False
):

    g1 = nx.bipartite.random_graph(u, v, p, seed=seed)
    weights, w = generate_weights_geometric(
        weight_distribution, u, v, weight_param, g1, seed
    )
    # s = sorted(list(g1.nodes))
    # c = nx.convert_matrix.to_numpy_array(g1, s)
    d = [dict(weight=float(i)) for i in list(w)]
    nx.set_edge_attributes(g1, dict(zip(list(g1.edges), d)))

    return g1, weights, w


def generate_edge_obm_data_geometric(
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
    Generates edge weighted bipartite graphs using the ER/BA schemes in pytorch geometric format

    Supports uniformm, normal, and power distributions.
    """
    D, M , S = [], [], []
    vary_fixed = False
    edges, tasks, workers = None, None, None
    if graph_family == "er":
        g = generate_er_graph
    elif graph_family == "ba":
        g = generate_ba_graph
    elif graph_family == "gmission" or graph_family == "gmission-var":
        edges, tasks = parse_gmission_dataset()
        max_w = max(np.array(list(edges.values()), dtype="float"))
        edges = {k: (float(v) / float(max_w)) for k, v in edges.items()}
        np.random.seed(100)
        workers = list(np.random.randint(1, 533, size=u_size))
        g = generate_gmission_graph
        vary_fixed = graph_family == "gmission-var"
    for i in tqdm(range(dataset_size)):
        g1, weights, w = g(
            u_size,
            v_size,
            tasks,
            edges,
            workers,
            graph_family_parameter,
            seed + i,
            weight_distribution,
            weight_param,
            vary_fixed,
        )
        # d_old = np.array(sorted(g1.degree))[u_size:, 1]

        g1.add_node(
            -1, bipartite=0
        )  # add extra node in U that represents not matching the current node to anything
        g1.add_edges_from(
            list(zip([-1] * v_size, range(u_size, u_size + v_size))), weight=0
        )
        i1, i2 = linear_sum_assignment(weights, maximize=True)
        optimal = weights[i1, i2].sum()
        # s = sorted(list(g1.nodes))
        # m = 1 - nx.convert_matrix.to_numpy_array(g1, s)
        data = from_networkx(g1)
        data.x = i2.tolist() # this is a list, must convert to tensor when a batch is called
        data.y = torch.tensor(optimal).float()  #tuple of optimla and size of matching
        if save_data:
            torch.save(
                data, "{}/data_{}.pt".format(dataset_folder, i),
            )
        else:
            D.append(data)
            M.append(optimal)
        # ordered_m = np.take(np.take(m, order, axis=1), order, axis=0)
    return (list(D), torch.tensor(M), torch.tensor(S))


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

        # a = nx.bipartite.biadjacency_matrix(g1, range(0, u_size), range(u_size, u_size + v_size)).toarray()
        # print('a', a)
        # d_old = np.array(sorted(g1.degree))[u_size:, 1]
        weights, w = generate_weights(
            weight_distribution, u_size, v_size, weight_param, g1
        )

        # print('weights: ', weights)
        # print('w: ', w)
        # s = sorted(list(g1.nodes))
        # c = nx.convert_matrix.to_numpy_array(g1, s)

        g1.add_node(
            -1, bipartite=0
        )  # add extra node in U that represents not matching the current node to anything
        g1.add_edges_from(list(zip([-1] * v_size, range(u_size, u_size + v_size))))

        s = sorted(list(g1.nodes))
        # print('s: ', s)

        m = 1 - nx.convert_matrix.to_numpy_array(g1, s)
        # print('m: ', m)
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
        help="family of graphs to generate (er, ba, gmission, etc)",
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
        dataset = generate_edge_obm_data_geometric(
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
