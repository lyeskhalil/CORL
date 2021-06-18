#!/usr/bin/env python3.7

import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
import itertools
import time
import torch
import os

#os.chdir('~/CORL/data/dataset')

"""
pre-process the data for groubi
Reads data from the specfied file and writes the graph tensor into multu dict of the following form:
    combinations, ms= gp.multidict({ 
        ('u1','v1'):10,
        ('u1','v2'):13,
        ('u2','v1'):9,
        ('u2','v2'):3
     })
"""
def get_data(u_size, v_size, adjacency_matrix):

    dic = {}
    # start = time.time()
    for u, v in itertools.product(range(u_size), range(u_size, v_size + u_size)):
        dic[(u, v - u_size)] = adjacency_matrix[u][v]

    #for u, v in itertools.product(range(u_size), range(v_size)):
    #    dic[(u, v)] = E[u][v]

    return gp.multidict(dic)

def solve_eobm(u_size, v_size, adjacency_matrix):
    try:
        m = gp.Model("wobm")

        combinations, dic = get_data(u_size, v_size, adjacency_matrix)
        
        print('slving eobm')
        print('combinations: ', combinations)
        print('dic: ', dic)

        # add variable
        x = m.addVars(combinations, vtype="B", name="(u,v) pairs") 

        # set constraints
        c1 = m.addConstrs((x.sum("*", v) <= 1 for v in range(v_size)), "V")
        c2 = m.addConstrs((x.sum(u, "*") <= 1 for u in range(u_size)), "U")

        # set the objective
        m.setObjective(x.prod(dic), GRB.MAXIMIZE)
        m.optimize()

        matched = 0
        for v in m.getVars():
            if abs(v.x) > 1e-6:
                matched += v.x
        
        print("total nodes matched: ", matched)
        print("total matching score: ", m.objVal)

    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Encountered an attribute error")


def solve_submodular_matching(u_size, v_size, adjacency_matrix, r_v, edge_vector_dic, preferences):
    try:
        m = gp.Model("submatching")

        #15 is the fixed number of genres
        genres= 4 #15 

        combinations, dic = get_data(u_size, v_size, adjacency_matrix)

        print('solving submodular matching')
        print('combinations: ', combinations)
        print('dic: ', dic)

        # add variable
        x = m.addVars(combinations, vtype="B", name="(u,v) pairs")
        gamma = m.addVars(genres, v_size, name="gamma")

        # A is |genres| by |V| matrix containg the total number of edges going from u to genere g at index (g,u)
        A = m.addVars(genres, v_size)
        for z, v in itertools.product(range(genres), range(v_size)):
            for u in range(u_size):
                if edge_vector_dic[u][z] == 1: #if u belongs to genre z
                    A[(z, v)] += x[(u, v)]

        # set constraints
        m.addConstrs((x.sum("*", v) <= r_v[v] for v in r_v), "const1")
        m.addConstrs((x.sum(u, "*") <= 1 for u in range(u_size)), "const2")
        m.addConstrs((gamma[(z, v)] <= A[(z, v)] for z, v in itertools.product(range(genres), range(v_size))), "const3")
        m.addConstrs((gamma[(z, v)] <= 1 for z, v in itertools.product(range(genres), range(v_size))), "const4")

        # set the objective
        m.setObjective((preferences* gamma).prod(dic), GRB.MAXIMIZE)
        m.optimize()

        #matched = 0
        #for v in m.getVars():
        #    if abs(v.x) > 1e-6:
        #        matched += v.x
        #print("total nodes matched: ", matched)

        print("total matching score: ", m.objVal)
        # print("time to generate the dictionary: ", end-start)

    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Encountered an attribute error")


if __name__ == "__main__":

    # {v_id : freq}
    r_v =  {0: 2, 1: 1}


    # 4 genres
    d = {0: [1.0, 0.0, 0.0, 0.0], 
         1: [0.0, 0.0, 1.0, 0.0], 
         2: [1.0, 1.0, 0.0, 0.0]}

   # user preferences 
    p = np.array([
       [1.2, 3.4, 5, 0],
       [3, 3, 5, 0]
    ])

    p1 = sp.csr_matrix(p)
    print('p1: ',p1)

    E = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0]
            ])

    # E = np.array([
    #     [1,0,0],
    #     [0,1,0],
    #     [0,0,1]
    # ])

    solve_submodular_matching(3, 5, E, r_v, d, p)
    #solve_eobm(3, 3, E)
