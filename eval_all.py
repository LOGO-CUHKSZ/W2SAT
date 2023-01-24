import itertools as it
import time
import os
from utils import *
from pysat.solvers import Glucose3

import numpy as np
from scipy import sparse

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# CELL
import warnings

warnings.filterwarnings("ignore")

import pickle
import scipy.sparse as sp
import torch

from cell.utils import link_prediction_performance
from cell.cell import Cell, EdgeOverlapCriterion, LinkPredictionCriterion
from cell.graph_statistics import compute_graph_statistics


class GCN(torch.nn.Module):
    def __init__(self, node_features):
        super().__init__()
        # GCN initialization
        self.conv1 = GCNConv(node_features, 128)
        self.conv2 = GCNConv(128, 128)
        # self.conv1 = GATConv(node_features, 64, 5)
        # self.conv2 = GATConv(64 * 5, 128)
        # self.conv3 = GCNConv(128, 128)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.elu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # x = F.tanh(x)
        # x = self.conv3(x, edge_index)

        return x


metrics_table = []
owc_table = []


formulas_path = "./dataset/formulas/"
sat_names = os.listdir(formulas_path)

for sat_name in sat_names:
    sat_name = "mrpp_4x4#4_5.processed.cnf"
    # sat_name = 'countbitsrotate016.processed.cnf'

    sat_path = f"./dataset/formulas/{sat_name}"
    num_vars, num_clauses, sat_instance = read_sat(sat_path)
    max_len = max([len(clause) for clause in sat_instance])

    # metric for original instance
    metrics = eval_solution(sat_instance, num_vars)
    metrics_table.append([sat_name.split(".")[0]].extend(metrics))

    lig_adjacency_matrix, lig_weighted_adjacency_matrix = sat_to_lig_adjacency_matrix(
        sat_instance, num_vars
    )

    start_time = time.time()

    clique_candidates = get_clique_candidates(lig_adjacency_matrix, 1, max_len)
    current_cliques = lazy_clique_edge_cover(
        np.copy(lig_weighted_adjacency_matrix),
        clique_candidates,
        int(num_clauses / 1.5),
    )

    # metric of owc incstance
    current_sat = cliques_to_sat(current_cliques)
    metrics = eval_solution(current_sat, num_vars)
    metrics_table.append(["OWC for origin"].extend(metrics))

    owc_time = time.time() - start_time
    owc_table.append([sat_name.split(".")[0], num_vars, num_clauses, owc_time])

    edge_index = torch.tensor(
        np.array(lig_adjacency_matrix.nonzero()), dtype=torch.long
    )
    print(edge_index.shape)
    edge_value = lig_weighted_adjacency_matrix[lig_adjacency_matrix.nonzero()]

    embeddings = torch.load(f"./model/embeddings/{sat_name}.pt")
    embeddings.requires_grad = False
    # print(embeddings)
    x = embeddings
    data = Data(x=x, edge_index=edge_index)

    # training
    model = GCN(50)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        out = model(data)
        src, dst = edge_index
        score = (out[src] * out[dst]).sum(dim=-1)
        # score = torch.sigmoid(score)
        loss = F.mse_loss(score, torch.tensor(edge_value, dtype=torch.float))
        loss.backward()
        optimizer.step()
        # print(f'epoch: {epoch}, loss: {loss.item()}')

    out = model(data)
    src, dst = edge_index
    score = (out[src] * out[dst]).sum(dim=-1)

    sparse_matrix = sparse.csr_matrix(lig_adjacency_matrix)
    cell_model = Cell(
        A=sparse_matrix,
        H=12,
        callbacks=[EdgeOverlapCriterion(invoke_every=10, edge_overlap_limit=0.80)],
    )

    cell_model.train(
        steps=400,
        optimizer_fn=torch.optim.Adam,
        optimizer_args={"lr": 0.1, "weight_decay": 1e-7},
    )

    generated_graph = cell_model.sample_graph()
    graph_prime = generated_graph.A
    graph_prime = graph_post_process(graph_prime)
    edge_index_prime = torch.tensor(graph_prime.nonzero(), dtype=torch.long)
    x = embeddings
    data_prime = Data(x=x, edge_index=edge_index_prime)
    out = model(data_prime)
    src, dst = edge_index_prime
    score = (out[src] * out[dst]).sum(dim=-1)
    weight = score.detach().numpy()
    weight[weight <= 1] = 1
    weight = np.rint(weight).astype(int)

    weighted_graph_prime = np.copy(graph_prime)
    weighted_graph_prime[weighted_graph_prime.nonzero()] = weight

    clique_candidates = get_clique_candidates(graph_prime, 1, max_len)
    current_cliques = lazy_clique_edge_cover(
        np.copy(weighted_graph_prime), clique_candidates, int(num_clauses / 1.5)
    )
    current_sat = cliques_to_sat(current_cliques)
    metrics = eval_solution(current_sat, num_vars)
    metrics_table.append(["OWC for origin"].extend(metrics))
