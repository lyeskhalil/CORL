import argparse
import os
import numpy as np
from numpy.lib.function_base import _i0_2
from data.data_utils import (
    add_nodes_with_bipartite_label,
    get_solution,
    parse_gmission_dataset,
    parse_movie_lense_dataset,
    from_networkx,
    generate_weights_geometric,
)
import networkx as nx
from scipy.optimize import linear_sum_assignment
import torch
from tqdm import tqdm

# from .IPsolvers.IPsolver import solve_submodular_matching


def generate_ba_graph(u, v, p, seed):
    """
    Genrates a graph using the preferential attachment scheme
    """
    np.random.seed(seed)

    G = nx.Graph()
    G = add_nodes_with_bipartite_label(G, u, v)

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


def generate_movie_lense_graph(
    u, v, users, edges, movies, sampled_movies, weight_features, seed, vary_fixed=False
):
    np.random.seed(seed)
    G = nx.Graph()
    G = add_nodes_with_bipartite_label(G, u, v)

    G.name = f"movielense_random_graph({u},{v})"

    movies_id = np.array(list(movies.keys())).flatten()
    users_id = np.array(list(users.keys())).flatten()

    if vary_fixed:
        sampled_movies = list(np.random.choice(movies_id, size=u, replace=False))

    movies_features = list(map(lambda m: movies[m], sampled_movies))
    users_features = []
    user_freq_dic = {}  # {v_id: freq}, used for the IPsolver
    sampled_users_dic = {}  # {user_id: v_id}
    edge_vector_dic = {u: movies_features[u] for u in range(len(sampled_movies))}
    preference_matrix = np.zeros((15, v))  # 15 is the number of genres

    for i in range(v):
        j = 0
        while j == 0:
            sampled_user = np.random.choice(users_id)
            user_info = list(weight_features[sampled_user]) + users[sampled_user]
            for w in range(len(sampled_movies)):
                movie = sampled_movies[w]
                edge = (movie, sampled_user)
                if edge in edges and (w, i + u) not in G.edges:
                    G.add_edge(w, i + u)
                    j += 1
        if sampled_user in sampled_users_dic:
            i = sampled_users_dic[sampled_user]
            user_freq_dic[i] += 1
        else:
            sampled_users_dic[sampled_user] = i
            user_freq_dic[i] = 1
        preference_matrix[:, i] = weight_features[sampled_user]
        users_features.append(user_info)

    # user_freq = list(map(lambda id: user_freq_dic[id], user_freq_dic)) + [0] * (v - (len(user_freq_dic)))

    return (
        G,
        np.array(movies_features),
        np.array(users_features),
        nx.adjacency_matrix(G).todense(),
        user_freq_dic,
        edge_vector_dic,
    )


def generate_gmission_graph(
    u, v, tasks, edges, workers, p, seed, weight_dist, weight_param, vary_fixed=False
):
    np.random.seed(seed)

    G = nx.Graph()
    G = add_nodes_with_bipartite_label(G, u, v)

    G.name = f"gmission_random_graph({u},{v})"
    if vary_fixed:
        workers = list(np.random.choice(np.arange(1, 533), size=u, replace=False))
    availableWorkers = workers.copy()
    weights = []
    for i in range(v):
        j = 0

        while j == 0:
            curr_w = []
            sampledTask = np.random.choice(tasks)
            for w in range(len(availableWorkers)):
                worker = availableWorkers[w]
                edge = str(float(worker)) + ";" + str(float(sampledTask))

                if edge in edges and (w, i + u) not in G.edges:
                    G.add_edge(w, i + u, weight=float(edges[edge]))
                    curr_w.append(float(edges[edge]))
                    j += 1
                elif edge not in edges:
                    curr_w.append(float(0))
        weights += curr_w

    weights = np.array(weights).reshape(v, u).T
    w = np.delete(weights.flatten(), weights.flatten() == 0)
    return G, weights, w


def generate_er_graph(
    u,
    v,
    tasks,
    edges,
    workers,
    p,
    seed,
    weight_distribution,
    weight_param,
    vary_fixed=False,
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


def generate_osbm_data_geometric(
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
    D, M, S = [], [], []
    vary_fixed = False
    edges, users, movies = None, None, None
    if "movielense" in graph_family:
        users, movies, edges, feature_weights = parse_movie_lense_dataset()
        np.random.seed(100)
        movies_id = np.array(list(movies.keys())).flatten()
        sampled_movies = list(np.random.choice(movies_id, size=u_size, replace=False))
        g = generate_movie_lense_graph
        vary_fixed = "var" in graph_family
    for i in tqdm(range(dataset_size)):
        (
            g1,
            movie_features,
            user_features,
            adjacency_matrix,
            user_freq,
            edge_vector_dic,
        ) = g(
            u_size,
            v_size,
            users,
            edges,
            movies,
            sampled_movies,
            feature_weights,
            seed + i,
            vary_fixed,
        )

        g1.add_node(
            -1, bipartite=0
        )  # add extra node in U that represents not matching the current node to anything
        g1.add_edges_from(list(zip([-1] * v_size, range(u_size, u_size + v_size))))
        data = from_networkx(g1)
        data.x = torch.tensor(
            np.concatenate((movie_features.flatten(), user_features.flatten()))
        )
        data.y = 10  # solve_submodular_matching(u_size, v_size, adjacency_matrix, user_freq, edge_vector_dic)
        if save_data:
            torch.save(
                data, "{}/data_{}.pt".format(dataset_folder, i),
            )
        else:
            D.append(data)
        # ordered_m = np.take(np.take(m, order, axis=1), order, axis=0)
    return (list(D), torch.tensor(M), torch.tensor(S))


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
    D, M, S = [], [], []
    vary_fixed = False
    edges, tasks, workers = None, None, None
    if graph_family == "er":
        g = generate_er_graph
    elif graph_family == "ba":
        g = generate_ba_graph
    elif "gmission" in graph_family:
        edges, tasks, reduced_tasks, reduced_workers = parse_gmission_dataset()
        max_w = max(np.array(list(edges.values()), dtype="float"))
        edges = {k: (float(v) / float(max_w)) for k, v in edges.items()}
        np.random.seed(100)
        rep = graph_family == "gmission"
        workers = list(np.random.choice(np.arange(1, 533), size=u_size, replace=rep))
        if graph_family == "gmission-max":
            tasks = reduced_tasks
            workers = np.random.choice(reduced_workers, size=u_size, replace=False)
        g = generate_gmission_graph

        vary_fixed = "var" in graph_family
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
        i1, i2 = linear_sum_assignment(weights.T, maximize=True)

        optimal = (weights.T)[i1, i2].sum()

        solution = get_solution(i1, i2, weights.T, v_size)

        # s = sorted(list(g1.nodes))
        # m = 1 - nx.convert_matrix.to_numpy_array(g1, s)
        data = from_networkx(g1)
        data.x = torch.tensor(
            solution
        )  # this is a list, must convert to tensor when a batch is called
        data.y = torch.tensor(optimal).float()  # tuple of optimla and size of matching
        if save_data:
            torch.save(
                data, "{}/data_{}.pt".format(dataset_folder, i),
            )
        else:
            D.append(data)
            M.append(optimal)
        # ordered_m = np.take(np.take(m, order, axis=1), order, axis=0)
    return (list(D), torch.tensor(M), torch.tensor(S))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--problem", type=str, default="obm", help="Problem: 'obm', 'e-obm', 'osbm'",
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

    if not os.path.exists(opts.dataset_folder):
        os.makedirs(opts.dataset_folder)
        if not opts.eval:
            os.makedirs("{}/graphs".format(opts.dataset_folder))
    np.random.seed(opts.seed)

    if opts.problem == "e-obm":
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
    elif opts.problem == "osbm":
        dataset = generate_osbm_data_geometric(
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
