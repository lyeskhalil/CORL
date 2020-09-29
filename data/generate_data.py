import argparse
import os
import numpy as np
from data_utils import check_extension, save_dataset
import networkx as nx
from scipy.optimize import linear_sum_assignment
import torch
import pickle as pk


def generate_bipartite_data(
    dataset_size, u_size, v_size, num_edges, future_edge_weight, weights_range
):
    """
    Generate random graphs using gnmk_random_graph
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


def generate_obm_data(opts):
    """
    Generates graphs using the ER scheme

    """
    G, D, E, M = [], [], [], []
    if opts.graph_family == "er":
        g = nx.bipartite.random_graph
    if opts.graph_family == "ba":
        g = generate_ba_graph
    for i in range(opts.dataset_size):
        g1 = g(
            opts.u_size, opts.v_size, p=opts.graph_family_parameter, seed=opts.seed + i
        )
        # d_old = np.array(sorted(g1.degree))[u_size:, 1]
        s = sorted(list(g1.nodes))
        c = nx.convert_matrix.to_numpy_array(g1, s)

        g1.add_node(
            -1, bipartite=0
        )  # add extra node in U that represents not matching the current node to anything
        g1.add_edges_from(
            list(zip([-1] * opts.v_size, range(opts.u_size, opts.u_size + opts.v_size)))
        )
        d = np.array(sorted(g1.degree))[opts.u_size + 1 :, 1]

        s = sorted(list(g1.nodes))
        m = 1 - nx.convert_matrix.to_numpy_array(g1, s)

        # ordered_m = np.take(np.take(m, order, axis=1), order, axis=0)
        G.append(m.tolist())
        D.append(list(d))
        E.append(s)
        i1, i2 = linear_sum_assignment(c, maximize=True)
        M.append(c[i1, i2].sum())

    return (
        G,
        D,
        E,
        M,
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
        help="Distributions to generate for problem, default 'all'.",
    )

    parser.add_argument(
        "--dataset_size", type=int, default=10000, help="Size of the dataset"
    )

    parser.add_argument(
        "--u_size", type=int, default=100, help="Sizes of U set (default 100 by 100)",
    )
    parser.add_argument(
        "--v_size",
        type=int,
        nargs="+",
        default=100,
        help="Sizes of V set (default 100 by 100)",
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
        default=0.6,
        help="parameter of the graph family distribution",
    )

    parser.add_argument("-f", action="store_true", help="Set true to overwrite")
    parser.add_argument("--seed", type=int, default=1234, help="Intitial Random seed")

    opts = parser.parse_args()

    # assert opts.filename is None or (
    #     len(opts.problems) == 1 and len(opts.graph_sizes) == 1
    # ), "Can only specify filename when generating a single dataset"

    # assert opts.f or not os.path.isfile(
    #     check_extension(filename)
    # ), "File already exists! Try running with -f option to overwrite."

    np.random.seed(opts.seed)
    if opts.problem == "obm":
        dataset = generate_obm_data(opts)
    # elif opt.problem == "e-obm":

    # elif opts.problem == "adwords":

    # elif opts.problem == "displayads":

    else:
        assert False, "Unknown problem: {}".format(opts.problem)

    save_dataset(
        dataset, "{}by{}-{}.pkl".format(opts.u_size, opts.v_size, opts.graph_family)
    )
