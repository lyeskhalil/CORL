import os
import pickle
import numpy as np
import networkx as nx
import torch
from scipy.stats import powerlaw
import torch_geometric


# gMission files
gMission_edges = "data/gMission/edges.txt"
gMission_tasks = "data/gMission/tasks.txt"
gMission_reduced_tasks = "data/gMission/reduced_tasks.txt"
gMission_reduced_workers = "data/gMission/reduced_workers.txt"

# MovieLense files
movie_lense_movies = "data/MovieLense/movies.txt"
movie_lense_users = "data/MovieLense/users.txt"
movie_lense_edges = "data/MovieLense/edges.txt"
movie_lense_ratings = "data/MovieLense/ratings.txt"
movie_lense_feature_weights = "data/MovieLense/feature_weights.txt"


def add_nodes_with_bipartite_label(G, lena, lenb):
    """
    Helper for generate_ba_graph that initializes the initial empty graph with nodes
    """
    G.add_nodes_from(range(0, lena + lenb))
    b = dict(zip(range(0, lena), [0] * lena))
    b.update(dict(zip(range(lena, lena + lenb), [1] * lenb)))
    nx.set_node_attributes(G, b, "bipartite")
    return G


def get_solution(row_ind, col_in, weights, v_size):
    """
    returns a np vector where the index at i is the the node in u that v_i connect to. If index is zero, then v[i]
    is connected to no node in U.
    """
    new_col_in = []
    # row_ind.sort()
    col_in = col_in + 1
    new_col_in += [0] * (row_ind[0])

    for i in range(0, len(row_ind) - 1):
        if weights[row_ind[i], col_in[i] - 1] != 0.0:
            new_col_in.append(col_in[i])
        else:
            new_col_in.append(0.0)
        new_col_in += [0.0] * (row_ind[i + 1] - row_ind[i] - 1)

    if weights[row_ind[-1], col_in[-1] - 1] != 0.0:
        new_col_in.append(col_in[-1])
    else:
        new_col_in.append(0.0)
    new_col_in += [0] * (v_size - row_ind[-1] - 1)
    return new_col_in


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    with open(check_extension(filename), "wb") as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), "rb") as f:
        return pickle.load(f)


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
    f_reduced_tasks = open(gMission_reduced_tasks, "r")
    f_reduced_workers = open(gMission_reduced_workers, "r")
    edgeWeights = dict()
    edgeNumber = dict()
    count = 0
    for line in f_edges:
        vals = line.split(",")
        edgeWeights[vals[0]] = vals[1].split("\n")[0]
        edgeNumber[vals[0]] = count
        count += 1

    tasks = list()
    reduced_tasks = []
    reduced_workers = []
    tasks_x = dict()
    tasks_y = dict()

    for line in f_tasks:
        vals = line.split(",")
        tasks.append(vals[0])
        tasks_x[vals[0]] = float(vals[2])
        tasks_y[vals[0]] = float(vals[3])

    for t in f_reduced_tasks:
        reduced_tasks.append(t)

    for w in f_reduced_workers:
        reduced_workers.append(w)

    return edgeWeights, tasks, reduced_tasks, reduced_workers


def parse_movie_lense_dataset():
    f_edges = open(movie_lense_edges, "r")
    f_movies = open(movie_lense_movies, "r")
    f_users = open(movie_lense_users, "r")
    f_feature_weights = open(movie_lense_feature_weights, "r")
    num_genres = 15
    gender_map = {"M": 0, "F": 1}
    age_map = {"1": 0, "18": 1, "25": 2, "35": 3, "45": 4, "50": 5, "56": 6}
    genre_map = {
        "Action": 0,
        "Adventure": 1,
        "Animation": 2,
        "Children's": 3,
        "Comedy": 4,
        "Crime": 5,
        "Documentary": 6,
        "Drama": 7,
        "Film-Noir": 8,
        "Horror": 9,
        "Musical": 10,
        "Romance": 11,
        "Sci-Fi": 12,
        "Thriller": 13,
        "War": 14,
    }
    users = {}
    movies = {}
    edges = {}
    feature_weights = {}
    user_ids = []
    popularity = {}
    for u in f_users:
        info = u.split(",")[:4]
        info[1] = float(gender_map[info[1]])
        info[2] = float(age_map[info[2]]) / 6.0
        info[3] = float(info[3]) / 21.0
        users[info[0]] = info[1:4]
        user_ids.append(int(info[0]))
    user_ids.sort()
    for i, u in enumerate(user_ids):
        users[str(u)].append(i)

    for m in f_movies:
        info = m.split("::")
        genres = info[2].split("|")
        genres[-1] = genres[-1].split("\n")[0]  # remove "\n" character
        genres_id = np.array(list(map(lambda g: genre_map[g], genres)))
        one_hot_encoding = np.zeros(num_genres)
        one_hot_encoding[genres_id] = 1.0
        movies[info[0]] = list(one_hot_encoding)
        popularity[info[0]] = 0

    for e in f_edges:
        info = e.split(",")
        genres = info[3].split("|")
        genres[-1] = genres[-1].split("\n")[0]  # remove "\n" character
        edges[(info[2], info[1])] = list(map(lambda g: genre_map[g], genres))
        popularity[info[2]] += 1

    for w in f_feature_weights:
        feature = w.split(",")
        feature[-1] = feature[-1].split("\n")[0]  # remove "\n" character
        if feature[1] not in feature_weights:
            feature_weights[feature[1]] = [0.0] * num_genres
        feature_weights[feature[1]][genre_map[feature[0]]] = float(feature[2]) / 5.0
    return users, movies, edges, feature_weights, popularity


def find_best_tasks(tasks, edges):
    task_total = {}
    f_open = open("data/gMission/reduced_tasks.txt", "a")
    for e in edges.items():
        task = e[0].split(";")[1]
        if task not in task_total:
            task_total[task] = 1
        else:
            task_total[task] += 1
    top_tasks = sorted(task_total.items(), key=lambda k: k[1], reverse=True)[:300]
    for t in top_tasks:
        f_open.write(t[0] + "\n")
    return top_tasks


def find_best_workers(tasks, edges):
    task_total = {}
    f_open = open("data/gMission/reduced_workers.txt", "a")
    for e in edges.items():
        task = e[0].split(";")[0]
        if task not in task_total:
            task_total[task] = 1
        else:
            task_total[task] += 1
    top_tasks = sorted(task_total.items(), key=lambda k: k[1], reverse=True)[:200]
    for t in top_tasks:
        f_open.write(t[0] + "\n")
    return top_tasks


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
        graph = weights * weights.sum(axis=1).reshape(-1, 1)
        noise = np.abs(
            np.random.normal(
                float(parameters[0]), float(parameters[1]), (u_size, v_size)
            )
        )
        weights = np.where(graph, (graph + noise) / v_size, graph)
        w = torch.cat(
            (torch.zeros(v_size, 1).float(), torch.tensor(weights).T.float()), 1
        )
    elif distribution == "node-normal":
        adj = nx.bipartite.biadjacency_matrix(
            g1, range(0, u_size), range(u_size, u_size + v_size)
        ).toarray()
        mean = np.random.randint(
            float(parameters[0]), float(parameters[1]), (u_size, 1)
        )
        variance = np.sqrt(
            np.random.randint(float(parameters[0]), float(parameters[1]), (u_size, 1))
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
