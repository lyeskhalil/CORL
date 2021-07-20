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
def get_data(u_size, v_size, adjacency_matrix, prefrences):

    adj_dic = {}
    w = {}
    
    for v, u in itertools.product(range(v_size), range(u_size)):
        adj_dic[(v, u)] = adjacency_matrix[v][u]
    _ , dic = gp.multidict(adj_dic)

    for i, j in itertools.product(range(prefrences.shape[0]), range(prefrences.shape[1])):
        w[(j, i)] = prefrences[i][j]

    return dic, w

def solve_eobm(u_size, v_size, adjacency_matrix):
    try:
        m = gp.Model("wobm")

        combinations, dic = get_data(u_size, v_size, adjacency_matrix)
    
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


def solve_submodular_matching(u_size, v_size, adjacency_matrix, r_v, movie_features, preferences):
    try:
        m = gp.Model("submatching")
        m.Params.LogToConsole = 0

        #15 is the fixed number of genres from the movielens dataset
        genres= 15

        dic, weight_dic  = get_data(u_size, v_size, adjacency_matrix, preferences)

        # add variable for each edge (u,v), where v is the user and u is the movie
        x = m.addVars(v_size, u_size, vtype="B", name="(u,v) pairs")

        # set the variable to zero for edges that do not exist in the graph
        for key in x:
           x[key] = 0 if dic[key] == 0 else x[key]

        #create variable for each (genre, user) pair
        gamma = m.addVars(genres, v_size, vtype="B", name="gamma")

        # A is |genres| by |V| matrix containg the total number of edges going from u to genere g at index (g,u)
        A = m.addVars(genres, v_size)

        # first set all variables in A to zero
        for key in A:
            A[key] = 0

        for z, v in itertools.product(range(genres), range(v_size)):
            for u in range(u_size):
                if movie_features[u][z] == 1.0: #if u belongs to genre z
                    A[(z, v)] += x[(v, u)] 
        
        # set constraints
        m.addConstrs((x.sum(v, "*") <= r_v[v] for v in r_v), "const1")
        m.addConstrs((x.sum("*", u) <= 1 for u in range(u_size)), "const2")
        m.addConstrs((gamma[(z, v)] <= A[(z, v)] for z, v in itertools.product(range(genres), range(v_size))), "const3")
        m.addConstrs((gamma[(z, v)] <= 1 for z, v in itertools.product(range(genres), range(v_size))), "const4")

        # give each gamma variable a weight based on the user preferences and optimiza the sum
        m.setObjective(gamma.prod(weight_dic), GRB.MAXIMIZE)
        m.optimize()

        #print("total matching score: ", m.objVal)
        return m.objVal

    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Encountered an attribute error")


if __name__ == "__main__":
    # {v_id : freq}
    #r_v =  {0: 2, 1: 1}  ##

    # 3 genres (each column), 3 movies (each row)
    #movie_features = [ ##
    #    [0.0, 0.0, 1.0], 
    #    [0.0, 0.0, 1.0], 
    #    [1.0, 1.0, 0.0]
    #    ]

   # user preferences  V by |genres|
    #preferences = np.array([
    #   [1, 3.4, 5],
    #   [3, 3, 5]
    #])
    
    #adjacency_matrix = np.array([ ##
    #     [1,0,1],
    #     [0,1,0]
    #])  
    
    solve_submodular_matching(3, 5, adjacency_matrix, r_v, movie_features, preferences)
    #solve_eobm(3, 3, E)
