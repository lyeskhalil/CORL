import gurobipy as gp
from gurobipy import GRB
import itertools
import numpy as np


def get_data_adwords(u_size, v_size, adjacency_matrix):
    """
    pre-process the data for groubi for the adwords problem
    Reads data from the specfied file and writes the graph tensor into multu dict of the following form:
        combinations, ms= gp.multidict({
            ('u1','v1'):10,
            ('u1','v2'):13,
            ('u2','v1'):9,
            ('u2','v2'):3
        })
    """

    adj_dic = {}

    for v, u in itertools.product(range(v_size), range(u_size)):
        adj_dic[(v, u)] = adjacency_matrix[u, v]

    return gp.multidict(adj_dic)


def get_data_osbm(u_size, v_size, adjacency_matrix, prefrences):
    """
    pre-process the data for groubi for the osbm problem
    """

    adj_dic = {}
    w = {}

    for v, u in itertools.product(range(v_size), range(u_size)):
        adj_dic[(v, u)] = adjacency_matrix[v][u]
    _, dic = gp.multidict(adj_dic)

    for i, j in itertools.product(
        range(prefrences.shape[0]), range(prefrences.shape[1])
    ):
        w[(j, i)] = prefrences[i][j]
    return dic, w


def solve_adwords(u_size, v_size, adjacency_matrix, budgets):
    try:
        m = gp.Model("adwords")
        # m.Params.LogToConsole = 0
        m.Params.timeLimit = 30

        _, dic = get_data_adwords(u_size, v_size, adjacency_matrix)

        # add variable
        x = m.addVars(v_size, u_size, vtype="B", name="(u,v) pairs")

        # set constraints
        m.addConstrs((x.sum(v, "*") <= 1 for v in range(v_size)), "V")
        m.addConstrs((x.prod(dic, "*", u) <= budgets[u] for u in range(u_size)), "U")

        # set the objective
        m.setObjective(x.prod(dic), GRB.MAXIMIZE)
        m.optimize()

        solution = np.zeros(v_size).tolist()
        for v in range(v_size):
            u = 0
            for nghbr_v in x.select(v, "*"):
                if nghbr_v.getAttr("x") == 1:
                    solution[v] = u + 1
                    break
                u += 1
        return m.objVal, solution

    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Encountered an attribute error")


def solve_submodular_matching(
    u_size, v_size, adjacency_matrix, r_v, movie_features, preferences, num_incoming
):
    try:
        m = gp.Model("submatching")
        m.Params.LogToConsole = 0
        # 15 is the fixed number of genres from the movielens dataset
        genres = 15
        dic, weight_dic = get_data_osbm(u_size, v_size, adjacency_matrix, preferences)

        # add variable for each edge (u,v), where v is the user and u is the movie
        x = m.addVars(v_size, u_size, vtype="B", name="(u,v) pairs")

        # set the variable to zero for edges that do not exist in the graph
        for key in x:
            x[key] = 0 if dic[key] == 0 else x[key]

        # create variable for each (genre, user) pair
        gamma = m.addVars(genres, v_size, vtype="B", name="gamma")

        # A is |genres| by |V| matrix containg the total number of edges going from u to genere g at index (g,u)
        A = m.addVars(genres, v_size)

        # first set all variables in A to zero
        for key in A:
            A[key] = 0

        for z, v in itertools.product(range(genres), range(v_size)):
            for u in range(u_size):
                if movie_features[u][z] == 1.0:  # if u belongs to genre z
                    A[(z, v)] += x[(v, u)]

        # set constraints
        r_v1 = {}
        for i, n in enumerate(r_v):
            r_v1[i] = r_v[n]
        m.addConstrs((x.sum(v, "*") <= len(r_v1[v]) for v in r_v1), "const1")
        m.addConstrs((x.sum("*", u) <= 1 for u in range(u_size)), "const2")
        m.addConstrs(
            (
                gamma[(z, v)] <= A[(z, v)]
                for z, v in itertools.product(range(genres), range(v_size))
            ),
            "const3",
        )
        m.addConstrs(
            (
                gamma[(z, v)] <= 1
                for z, v in itertools.product(range(genres), range(v_size))
            ),
            "const4",
        )

        # give each gamma variable a weight based on the user preferences and optimiza the sum
        m.setObjective(gamma.prod(weight_dic), GRB.MAXIMIZE)
        m.optimize()
        solution = np.zeros(num_incoming).tolist()
        sol_dict = dict(x)
        s = sorted(sol_dict.keys(), key=(lambda k: k[0]))
        matched = 0
        for i, p in enumerate(r_v1):
            matched = 0
            inds = r_v1[p]
            idx = i * u_size
            nodes = s[idx : idx + u_size]
            for j, u in enumerate(nodes):
                if (type(x[u]) is not int) and x[u].x == 1:
                    solution[inds[matched]] = u[1] + 1
                    matched += 1

            for j in range(len(inds) - matched):
                solution[inds[j + matched]] = 0

        return m.objVal, solution

    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Encountered an attribute error")


if __name__ == "__main__":

    # osbm exmaple:

    # {v_id : freq}
    # r_v = {0: [1, 2], 1: [0]}

    # # 3 genres (each column), 3 movies (each row)
    # movie_features = [
    #    [0.0, 0.0, 1.0],
    #    [0.0, 0.0, 1.0],
    #    [0.0, 1.0, 1.0]
    # ]

    # # user preferences  V by |genres|
    # preferences = np.array([
    #    [0.999, 0.4, 0.222],
    #    [1, 1, 1]
    # ])

    # adjacency_matrix = np.array([
    #   [1, 2, 0],
    #   [0, 1, 0],
    #   [4, 0, 0]
    # ])

    # print(solve_submodular_matching(3, 2, adjacency_matrix, r_v, movie_features, preferences, 3))

    # adwords exmaple:

    # V by U matrix
    adjacency_matrix = np.array([[1, 2, 0], [0, 1, 0], [4, 0, 0]])

    budgets = [3, 1, 4]
    print(solve_adwords(3, 3, adjacency_matrix, budgets))
