from scipy.sparse import csr_matrix
from scipy.linalg import eig
from numpy import empty as empty_matrix
import numpy as  np
CONVERGENCE_THRESHOLD = 0.0001
import torch

def pagerank_weighted(graph, initial_value=None, damping=0.85):
    """Calculates PageRank for an undirected graph"""
    if initial_value == None: initial_value = 1.0 / len(graph.nodes())
    scores = dict.fromkeys(graph.nodes(), initial_value)

    iteration_quantity = 0
    for iteration_number in range(100):
        iteration_quantity += 1
        convergence_achieved = 0
        for i in graph.nodes():
            rank = 1 - damping
            for j in graph.neighbors(i):
                neighbors_sum = sum(graph.edge_weight((j, k)) for k in graph.neighbors(j))
                rank += damping * scores[j] * graph.edge_weight((j, i)) / neighbors_sum

            if abs(scores[i] - rank) <= CONVERGENCE_THRESHOLD:
                convergence_achieved += 1

            scores[i] = rank

        if convergence_achieved == len(graph.nodes()):
            break

    return scores


def pagerank_weighted_scipy(graph, similarity, device):
    adjacency_matrix = build_adjacency_matrix(graph)
    probability_matrix = build_probability_matrix(graph)
    assert adjacency_matrix.shape[0] == similarity.shape[0]
    adjacency_matrix = torch.FloatTensor(adjacency_matrix.todense().data.obj).view(-1, similarity.shape[0]).to(device)
    probability_matrix = torch.FloatTensor(probability_matrix).view(-1, similarity.shape[0]).to(device)

    import warnings
    with warnings.catch_warnings():
        from numpy import VisibleDeprecationWarning
        warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        pagerank_matrix = 0.85 * (adjacency_matrix + similarity) + (1 - 0.85) * probability_matrix

    # vals, vecs = eig(pagerank_matrix, left=True, right=False)
    # vals= torch.eig(pagerank_matrix)
    vals= torch.linalg.eigvals(pagerank_matrix)

    # return process_results(graph, vals)
    return abs(vals)

def build_adjacency_matrix(graph):
    row = []
    col = []
    data = []
    nodes = graph.nodes()
    length = len(nodes)

    for i in range(length):
        current_node = nodes[i]
        neighbors_sum = sum(graph.edge_weight((current_node, neighbor)) for neighbor in graph.neighbors(current_node))
        for j in range(length):
            edge_weight = float(graph.edge_weight((current_node, nodes[j])))
            if i != j and edge_weight != 0:
                row.append(i)  # 行
                col.append(j)  # 列
                data.append(edge_weight / neighbors_sum)

    return csr_matrix((data,(row,col)), shape=(length,length))


def build_probability_matrix(graph):
    dimension = len(graph.nodes())
    matrix = empty_matrix((dimension, dimension))  # 随机初始化矩阵

    probability = 1 / float(dimension)
    matrix.fill(probability)

    return matrix


def process_results(graph, vecs):
    scores = []
    for i, node in enumerate(graph.nodes()):
        score = {}
        score['src_index'] = i
        score['score'] = abs(vecs[i])
        scores.append(score)

    # scores.sort(key=lambda s: s["score"], reverse=True)
    return scores
