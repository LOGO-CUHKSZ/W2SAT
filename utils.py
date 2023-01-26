import math
import time

import itertools as it
import collections
from random import shuffle

import networkx as nx
import numpy as np
import community


def graph_post_process(lig_adjacency_matrix):
    def get_literal_idx(x):
        return 2 * x - 2 if x > 0 else 2 * abs(x) - 1

    var_nums = int(len(lig_adjacency_matrix) / 2)
    pos_idx = [get_literal_idx(i) for i in range(1, var_nums + 1)]
    neg_idx = [get_literal_idx(-i) for i in range(1, var_nums + 1)]
    lig_adjacency_matrix[[pos_idx, neg_idx]] = 0
    lig_adjacency_matrix[[neg_idx, pos_idx]] = 0
    return lig_adjacency_matrix


def read_sat(sat_path):
    with open(sat_path) as f:
        sat_lines = f.readlines()
        header = sat_lines[0]
        header_info = header.replace("\n", "").split(" ")
        vars_num = int(header_info[-2])
        clauses_num = int(header_info[-1])

        sat = [
            [int(x) for x in line.replace(" 0\n", "").split(" ")]
            for line in sat_lines[1:]
        ]

        # return the probability that the polarity of literal is positive
        literal_counting = [0] * vars_num
        pos_literal_counting = [0] * vars_num

        for clause in sat:
            for var in clause:
                idx = abs(var) - 1
                literal_counting[idx] += 1
                if var > 0:
                    pos_literal_counting[idx] += 1

        return vars_num, clauses_num, sat


def sat_to_lig_adjacency_matrix(sat, num_vars):
    def get_literal_idx(x):
        return 2 * x - 2 if x > 0 else 2 * abs(x) - 1

    lig_adjacency_matrix = np.zeros([2 * num_vars, 2 * num_vars])
    lig_weighted_adjacency_matrix = np.zeros([2 * num_vars, 2 * num_vars])

    for clause in sat:
        pairs = it.combinations(clause, 2)
        #         print(f'clause: {clause}')
        for pair in pairs:
            x_idx = get_literal_idx(pair[0])
            y_idx = get_literal_idx(pair[1])
            #             print(f'pair: {(x_idx, y_idx)}')
            lig_adjacency_matrix[x_idx, y_idx] = 1
            lig_adjacency_matrix[y_idx, x_idx] = 1
            lig_weighted_adjacency_matrix[x_idx, y_idx] += 1
            lig_weighted_adjacency_matrix[y_idx, x_idx] += 1
    return lig_adjacency_matrix, lig_weighted_adjacency_matrix


def cliques_to_weighted_adjacency_matrix(cliques, num_vars):
    weighted_adjacency_matrix = np.zeros([2 * num_vars, 2 * num_vars])
    for clique in cliques:
        pairs = it.combinations(clique, 2)
        for pair in pairs:
            x_idx = pair[0]
            y_idx = pair[1]

            weighted_adjacency_matrix[x_idx, y_idx] += 1
            weighted_adjacency_matrix[y_idx, x_idx] += 1

    return weighted_adjacency_matrix


def objective(lig_weighted_adjacency_matrix, lig_weighted_adjacency_matrix_p):
    return (
        (lig_weighted_adjacency_matrix - lig_weighted_adjacency_matrix_p) ** 2
    ).mean()


def cliques_to_sat(cliques):
    sat = []
    for clique in cliques:
        clause = [int((x + 2) / 2) if x % 2 == 0 else int(-(x + 1) / 2) for x in clique]
        sat.append(clause)
    return sat


def lazy_clique_edge_cover(
    lig_weighted_adjacency_matrix, clique_candidates, cliques_quota
):
    def get_edges(clique):
        return it.combinations(clique, 2)

    node_occurrence = [[] for i in range(len(lig_weighted_adjacency_matrix))]
    for idx, clique in enumerate(clique_candidates):
        for node in clique:
            node_occurrence[node].append(idx)
    node_occurrence = [set(occ) for occ in node_occurrence]

    # build edge_occurrence
    triu_adjacency_matrix = np.triu(lig_weighted_adjacency_matrix)
    x, y = triu_adjacency_matrix.nonzero()
    edge_list = [frozenset([i, j]) for i, j in zip(x, y)]
    edge_occurrence = {}
    for edge in edge_list:
        i, j = edge
        edge_occurrence[edge] = list(
            node_occurrence[i].intersection(node_occurrence[j])
        )

    # default clique gain
    def get_default_gain(x):
        return len(x) * (len(x) - 1)

    clique_gain = np.array([get_default_gain(clique) for clique in clique_candidates])

    # the greedy process (import probability in here?)
    current_clique_idxs = []
    num_clique_candidates = len(clique_candidates)
    for i in range(cliques_quota):
        clique_gain[current_clique_idxs] = -10000
        best_clique_idx = np.argmax(clique_gain)
        current_clique_idxs.append(best_clique_idx)

        best_clique = clique_candidates[best_clique_idx]
        best_clique_edges = list(get_edges(best_clique))

        # update the weighted_adjacency_matrix
        to_update_edges = []
        for edge in best_clique_edges:
            x, y = edge
            lig_weighted_adjacency_matrix[x, y] -= 1
            lig_weighted_adjacency_matrix[y, x] -= 1
            if lig_weighted_adjacency_matrix[x, y] == 0:
                to_update_edges.append(edge)

        # find the realted clique and update the gain table
        for edge in to_update_edges:
            edge = frozenset(edge)
            cliques_idx = edge_occurrence[edge]
            for idx in cliques_idx:
                clique_gain[idx] = clique_gain[idx] - 2

    return [clique_candidates[idx] for idx in current_clique_idxs]


def get_clique_candidates(lig_adjacency_matrix, k):
    graph = nx.from_numpy_matrix(lig_adjacency_matrix)
    cliques = nx.enumerate_all_cliques(graph)
    clique_candidates = []
    for clique in cliques:
        if len(clique) <= k:
            if len(clique) > 1:
                clique_candidates.append(clique)
        else:
            break
    return clique_candidates


def cliques_to_weighted_adjacency_matrix(cliques, num_vars):
    weighted_adjacency_matrix = np.zeros([2 * num_vars, 2 * num_vars])
    for clique in cliques:
        pairs = it.combinations(clique, 2)
        for pair in pairs:
            x_idx = pair[0]
            y_idx = pair[1]

            weighted_adjacency_matrix[x_idx, y_idx] += 1
            weighted_adjacency_matrix[y_idx, x_idx] += 1

    return weighted_adjacency_matrix


def get_clique_gain(lig_weighted_adjacency_matrix, clique):
    gain = 0
    pairs = it.combinations(clique, 2)
    for pair in pairs:
        x_idx = pair[0]
        y_idx = pair[1]
        if lig_weighted_adjacency_matrix[x_idx][y_idx] > 0:
            gain += 1
        else:
            gain -= 1
        if lig_weighted_adjacency_matrix[y_idx][x_idx] > 0:
            gain += 1
        else:
            gain -= 1
    return gain


def preprocess_VIG(formula, VIG):
    """
    Builds VIG.
    """
    for cn in range(len(formula)):
        for i in range(len(formula[cn]) - 1):
            for j in range(len(formula[cn]))[i + 1 :]:
                VIG.add_edge(abs(formula[cn][i]), abs(formula[cn][j]))


def preprocess_LIG(formula, LIG, num_vars):
    for cn in range(len(formula)):
        for i in range(len(formula[cn]) - 1):
            for j in range(len(formula[cn]))[i + 1 :]:
                if formula[cn][i] > 0:
                    fst = formula[cn][i]
                else:
                    fst = abs(formula[cn][i]) + num_vars
                if formula[cn][j] > 0:
                    snd = formula[cn][j]
                else:
                    snd = abs(formula[cn][j]) + num_vars
                LIG.add_edge(fst, snd)


def preprocess_VCG(formula, VCG, num_vars):
    """
    Builds VCG
    """
    for cn in range(len(formula)):
        for var in formula[cn]:
            VCG.add_edge(abs(var), cn + num_vars + 1)


def preprocess_LCG(formula, LCG, num_vars):
    """
    Builds LCG
    """
    for cn in range(len(formula)):
        for var in formula[cn]:
            if var > 0:
                LCG.add_edge(abs(var), cn + num_vars + 1)
            else:
                LCG.add_edge(abs(var) + num_vars, cn + num_vars + 1)


def eval_solution(sat, num_vars):

    num_clauses = len(sat)

    VIG = nx.Graph()
    VIG.add_nodes_from(range(num_vars + 1)[1:])

    LIG = nx.Graph()
    LIG.add_nodes_from(range(num_vars * 2 + 1)[1:])

    VCG = nx.Graph()
    VCG.add_nodes_from(range(num_vars + num_clauses + 1)[1:])

    LCG = nx.Graph()
    VCG.add_nodes_from(range(2 * num_vars + num_clauses + 1)[1:])

    preprocess_VIG(sat, VIG)  # Build a VIG
    preprocess_LIG(sat, LIG, num_vars)  # Build a LIG
    preprocess_VCG(sat, VCG, num_vars)  # Build a VCG
    preprocess_LCG(sat, LCG, num_vars)  # Build a LCG

    clust_VIG = nx.average_clustering(VIG)
    clust_LIG = nx.average_clustering(LIG)

    part_VIG = community.best_partition(VIG)
    mod_VIG = community.modularity(part_VIG, VIG)

    part_LIG = community.best_partition(LIG)
    mod_LIG = community.modularity(part_LIG, LIG)  # Modularity of VCG

    part_VCG = community.best_partition(VCG)
    mod_VCG = community.modularity(part_VCG, VCG)  # Modularity of VCG

    part_LCG = community.best_partition(LCG)
    mod_LCG = community.modularity(part_LCG, LCG)  # Modularity of LCG

    return [clust_VIG, clust_LIG, mod_VIG, mod_LIG, mod_VCG, mod_LCG]
