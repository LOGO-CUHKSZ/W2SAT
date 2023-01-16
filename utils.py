import math
import time

import itertools as it
import collections
from random import shuffle

import networkx as nx
import numpy as np
import community


def graph_post_process(lig_adjacency_matrix):
    def get_literal_idx(x): return 2 * x - 2 if x > 0 else 2 * abs(x) - 1
    var_nums = int(len(lig_adjacency_matrix)/2)
    pos_idx = [get_literal_idx(i) for i in range(1, var_nums+1)]
    neg_idx = [get_literal_idx(-i) for i in range(1, var_nums+1)]
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

        sat = [[int(x) for x in line.replace(' 0\n', '').split(' ')]
               for line in sat_lines[1:]]

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
    def get_literal_idx(x): return 2 * x - 2 if x > 0 else 2 * abs(x) - 1
    lig_adjacency_matrix = np.zeros([2*num_vars, 2*num_vars])
    lig_weighted_adjacency_matrix = np.zeros([2*num_vars, 2*num_vars])

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
    weighted_adjacency_matrix = np.zeros([2*num_vars, 2*num_vars])
    for clique in cliques:
        pairs = it.combinations(clique, 2)
        for pair in pairs:
            x_idx = pair[0]
            y_idx = pair[1]

            weighted_adjacency_matrix[x_idx, y_idx] += 1
            weighted_adjacency_matrix[y_idx, x_idx] += 1

    return weighted_adjacency_matrix


def objective(lig_weighted_adjacency_matrix, lig_weighted_adjacency_matrix_p):
    return ((lig_weighted_adjacency_matrix - lig_weighted_adjacency_matrix_p)**2).mean()


def get_exchange_pairs(clique_bits):
    one_idxs = []
    zero_idxs = []
    for idx, bit in enumerate(clique_bits):
        if bit > 0:
            one_idxs.append(idx)
        else:
            zero_idxs.append(idx)

    idx_pairs = list(it.product(one_idxs, zero_idxs))
    shuffle(idx_pairs)
    return idx_pairs


def cliques_to_sat(cliques):
    sat = []
    for clique in cliques:
        clause = [int((x + 2)/2) if x % 2 == 0 else int(-(x + 1)/2)
                  for x in clique]
        sat.append(clause)

    return sat


def clique_edge_cover(lig_weighted_adjacency_matrix, clique_candidates, cliques_quota, num_vars, iterations):
    clique_bits = [1 if i < cliques_quota else 0 for i in range(
        len(clique_candidates))]
    shuffle(clique_bits)
    current_cliques = [clique for (clique, bit) in zip(
        clique_candidates, clique_bits) if bit == 1]
    current_weighted_adjacency_matrix = cliques_to_weighted_adjacency_matrix(
        current_cliques, num_vars)
    current_objective = objective(
        lig_weighted_adjacency_matrix, current_weighted_adjacency_matrix)

    for iteration in range(iterations):
        best_objective = current_objective
        idx_pairs = get_exchange_pairs(clique_bits)
        for pair in idx_pairs:
            temp_clique_bits = [x for x in clique_bits]
            temp_clique_bits[pair[0]] = 0
            temp_clique_bits[pair[1]] = 1
            temp_cliques = [clique for (clique, bit) in zip(
                clique_candidates, temp_clique_bits) if bit == 1]
            temp_weighted_adjacency_matrix = cliques_to_weighted_adjacency_matrix(
                temp_cliques, num_vars)
            temp_objective = objective(
                lig_weighted_adjacency_matrix, temp_weighted_adjacency_matrix)

            if temp_objective < current_objective:
                clique_bits = temp_clique_bits
                current_objective = temp_objective
                print(
                    f'iteration: {iteration}, current_objective: {current_objective}')
                break

        # print(current_objective)
        # if current_objective < 0.025:
        #     print('close enough')
        #     break
        if best_objective == current_objective:
            print('converge')
            break

    current_cliques = [clique for (clique, bit) in zip(
        clique_candidates, clique_bits) if bit == 1]
    current_weighted_adjacency_matrix = cliques_to_weighted_adjacency_matrix(
        current_cliques, num_vars)
    current_objective = objective(
        lig_weighted_adjacency_matrix, current_weighted_adjacency_matrix)

    return current_cliques, current_weighted_adjacency_matrix, current_objective


def greedy_clique_edge_cover(lig_weighted_adjacency_matrix, clique_candidates, cliques_quota, num_vars):
    current_clique_idxs = []
    current_cliques = [clique_candidates[idx] for idx in current_clique_idxs]
    current_weighted_adjacency_matrix = cliques_to_weighted_adjacency_matrix(
        current_cliques, num_vars)
    current_objective = objective(
        lig_weighted_adjacency_matrix, current_weighted_adjacency_matrix)

    clique_candidates_idx = set(range(len(clique_candidates)))
    min_clique_idx = []
    for i in range(cliques_quota):
        min_objective = 10
        for idx in clique_candidates_idx - set(current_clique_idxs):
            temp_clique_idx = current_clique_idxs + [idx]
            temp_cliques = [clique_candidates[idx] for idx in temp_clique_idx]
            temp_weighted_adjacency_matrix = cliques_to_weighted_adjacency_matrix(
                temp_cliques, num_vars)
            temp_objective = objective(
                lig_weighted_adjacency_matrix, temp_weighted_adjacency_matrix)
            if temp_objective < min_objective:
                min_objective = temp_objective
                min_clique_idx = temp_clique_idx

        current_clique_idxs = min_clique_idx
        current_objective = min_objective

        print(f'iteration: {i}, current_objective: {current_objective}')

    return current_clique_idxs, current_objective


def preprocess_VIG(formula, VIG):
    """
    Builds VIG.
    """
    for cn in range(len(formula)):
        for i in range(len(formula[cn]) - 1):
            for j in range(len(formula[cn]))[i + 1:]:
                VIG.add_edge(abs(formula[cn][i]), abs(formula[cn][j]))


def preprocess_LIG(formula, LIG, num_vars):
    for cn in range(len(formula)):
        for i in range(len(formula[cn]) - 1):
            for j in range(len(formula[cn]))[i + 1:]:
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
    weighted_adjacency_matrix = np.zeros([2*num_vars, 2*num_vars])
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


def lazy_clique_edge_cover(lig_weighted_adjacency_matrix, clique_candidates, cliques_quota, num_vars):
    def get_edges(clique): return it.combinations(clique, 2)
    current_clique_idxs = []
    triu_adjacency_matrix = np.triu(lig_weighted_adjacency_matrix)
    x, y = triu_adjacency_matrix.nonzero()

    # build edge_occurrence
    # print('dependency analysis start')
    start_time = time.time()
    num_clique_candidates = len(clique_candidates)
    edge_list = [frozenset([i, j]) for i, j in zip(x, y)]
    edge_occurrence = {x: [] for x in edge_list}
    for i in range(num_clique_candidates):
        clique = clique_candidates[i]
        pairs = get_edges(clique)
        for pair in pairs:
            edge_occurrence[frozenset(pair)].append(i)

    # print(f"edge_occurrence analysis time: {time.time() - start_time:.4f}")

    # default clique gain
    def get_default_gain(x): return len(x) * (len(x)-1)
    clique_gain = np.array([get_default_gain(clique)
                           for clique in clique_candidates])

    # the greedy process (import probability in here?)
    for i in range(cliques_quota):
        clique_gain[current_clique_idxs] = -10000
        best_clique_idx = np.argmax(clique_gain)
        current_clique_idxs.append(best_clique_idx)

        best_clique = clique_candidates[best_clique_idx]
        pairs = list(get_edges(best_clique))

        # update the weighted_adjacency_matrix
        for pair in pairs:
            x_idx = pair[0]
            y_idx = pair[1]
            lig_weighted_adjacency_matrix[x_idx, y_idx] -= 1
            lig_weighted_adjacency_matrix[y_idx, x_idx] -= 1
        # print(
        #     f'best_clique_idx: {best_clique_idx}, len of pairs: {len(pairs)}')

        # update realted clique
        # start_time = time.time()
        clique_candidates_idx = np.arange(num_clique_candidates)
        related_clique_idx = np.zeros(num_clique_candidates, dtype=bool)
        for pair in pairs:
            edge = frozenset(pair)
            cliques_idx_occurrence = edge_occurrence[edge]
            related_clique_idx[cliques_idx_occurrence] = True
        related_clique = clique_candidates_idx[related_clique_idx]
        # print(f"build related_clique time: {time.time() - start_time:.4f}")
        # related_clique_num = len(related_clique)
        # if i % 10 == 0:
        #     print(f"i: {i}, related cliques num: {related_clique_num}")

        # update gain table
        for clique_idx in related_clique:
            clique_prime = clique_candidates[clique_idx]
            clique_gain[clique_idx] = get_clique_gain(
                lig_weighted_adjacency_matrix, clique_prime)

    return current_clique_idxs


def cliques_to_sat(cliques):
    sat = []

    for clique in cliques:
        clause = [int((x + 2)/2) if x % 2 == 0 else int(-(x + 1)/2)
                  for x in clique]
        sat.append(clause)

    return sat
