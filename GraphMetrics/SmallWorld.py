import numpy as np
from random import randint

from GraphMetrics.ClusteringCoefficent import global_clustering_coefficent
from GraphMetrics.PathLength import avergage_path_length


def small_world_index(graph, threshold):
    graph[graph < threshold] = 0.0
    graph[graph > threshold] = 1.0
    num_nodes = graph.shape[0]
    num_edges = np.count_nonzero(graph)
    random_graph = generate_random_graph(num_nodes, num_edges)
    graph_cc = global_clustering_coefficent(graph)
    graph_pl = avergage_path_length(graph)
    random_cc = global_clustering_coefficent(random_graph)
    random_pl = avergage_path_length(random_graph)
    return _calc_small_world(graph_cc, random_cc, graph_pl, random_pl)

def _calc_small_world(gcc, rcc, gpl,rpl):
    """

    :param gcc: graph clustering coefficent
    :param rcc: random graph clustering coefficent
    :param gpl: graph average path length
    :param rpl: random graph average path length
    :return: small world index
    """
    return (gcc / rcc) / (gpl / rpl)

def generate_random_graph(num_nodes, num_edges):
    graph = np.zeros((num_nodes, num_nodes))
    edge_count = 0
    while edge_count < num_edges:
        node1 = randint(0, 137)
        node2 = randint(0, 137)
        if graph[node1, node2] != 0:
            graph[node1, node2] = 1
            graph[node2, node1] = 1
            edge_count += 1
    return graph