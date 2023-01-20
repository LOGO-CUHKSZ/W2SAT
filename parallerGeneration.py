from cell.graph_statistics import compute_graph_statistics
from cell.cell import Cell, EdgeOverlapCriterion, LinkPredictionCriterion
from cell.utils import link_prediction_performance
from scipy.sparse import load_npz
import scipy.sparse as sp
import pickle
import os
import itertools as it
from utils import *

import numpy as np
from scipy import sparse

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# from torch_geometric.nn import GATConv
from torch_geometric.data import Data

import warnings
warnings.filterwarnings('ignore')


class GCN(torch.nn.Module):
    def __init__(self, node_features):
        super().__init__()
        # GCN initialization
        self.conv1 = GCNConv(node_features, 64)
        self.conv2 = GCNConv(64, 128)
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


def train_eval(name, formulas_path='./dataset/formulas/'):
    # load data
    sat_path = f'./dataset/formulas/{name}'
    num_vars, num_clauses, sat_instance = read_sat(sat_path)
    max_len = max([len(clause) for clause in sat_instance])

    lig_adjacency_matrix, lig_weighted_adjacency_matrix = sat_to_lig_adjacency_matrix(
        sat_instance, num_vars)

    edge_index = torch.tensor(
        np.array(lig_adjacency_matrix.nonzero()), dtype=torch.long)
    edge_value = lig_weighted_adjacency_matrix[lig_adjacency_matrix.nonzero()]
    embeddings = torch.load(f'./model/embeddings/{name}.pt')
    embeddings.requires_grad = False
    x = embeddings
    data = Data(x=x, edge_index=edge_index)

    # GNN training
    model = GCN(50)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(400):
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

    # CELL training
    sparse_matrix = sparse.csr_matrix(lig_adjacency_matrix)
    cell_model = Cell(A=sparse_matrix,
                      H=10,
                      callbacks=[EdgeOverlapCriterion(invoke_every=10, edge_overlap_limit=.75)])

    cell_model.train(steps=200,
                     optimizer_fn=torch.optim.Adam,
                     optimizer_args={'lr': 0.1,
                                     'weight_decay': 1e-7})

    # WLIG generation
    generate_num = 50
    for idx in range(generate_num):
        generated_graph = cell_model.sample_graph()
        graph_prime = generated_graph.A
        graph_prime = graph_post_process(graph_prime)

        edge_index_prime = torch.tensor(
            graph_prime.nonzero(), dtype=torch.long)
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
        # lig_adjacency_matrix, lig_weighted_adjacency_matrix = sat_to_lig_adjacency_matrix(sat_instance, num_vars)
        clique_candidates = get_clique_candidates(
            lig_adjacency_matrix, max_len)
        current_cliques = lazy_clique_edge_cover(
            np.copy(lig_weighted_adjacency_matrix), clique_candidates, num_clauses)
        current_sat = cliques_to_sat(current_cliques)

        path = f"./result/generation/"
        with open(path, 'w') as f:
            f.write('p cnf 91 377\n')
            for clause in current_sat:
                f.write(f"{' '.join([str(v) for v in clause])} 0\n")


formulas_path = './dataset/formulas/'
names = os.listdir(formulas_path)
