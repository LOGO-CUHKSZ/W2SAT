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
    print('dependency analysis start')
    start_time = time.time()
    num_clique_candidates = len(clique_candidates)
    edge_list = [frozenset([i, j]) for i, j in zip(x, y)]
    edge_occurrence = {x: [] for x in edge_list}
    for i in range(num_clique_candidates):
        clique = clique_candidates[i]
        pairs = get_edges(clique)
        for pair in pairs:
            edge_occurrence[frozenset(pair)].append(i)

    print(f"edge_occurrence analysis time: {time.time() - start_time:.4f}")

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
        print(
            f'best_clique_idx: {best_clique_idx}, len of pairs: {len(pairs)}')

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
        related_clique_num = len(related_clique)

        if i % 10 == 0:
            print(f"i: {i}, related cliques num: {related_clique_num}")

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
