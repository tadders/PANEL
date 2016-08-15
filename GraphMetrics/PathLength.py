import numpy as np

def avergage_path_length(graph):
    """

    :param graph:nxn numpy adjacency matrix
    :return:
    """
    length_sum = 0.0
    num_nodes = graph.shape[0]
    for node in range(num_nodes):
        path_distances = djikstra(graph, node)
        for other_node in range(num_nodes):
            if other_node != node:
                length_sum += path_distances[other_node]
    return length_sum / (num_nodes * (num_nodes - 1))

def local_path_length(graph, start_node, end_node):
    """
    :param graph: nxn numpy adjacency matrix
    :param start_node: int
    :param end_node: int
    :return: shortest path distance between start_node and end_node
    """
    path_distances = djikstra(graph, start_node)
    return path_distances[end_node]

def djikstra(graph, start_node):
    num_nodes = graph.shape[0]
    node_distance = np.zeros((num_nodes,))
    nodes_to_visit = [start_node]
    visited_nodes = []
    distance = 1
    while len(nodes_to_visit) > 0:
        new_nodes = []
        new_nodes = []
        for node in nodes_to_visit:
            visited_nodes += [node]
            neighbours = np.where(graph[node] == 1)[0]

            for neighbour in neighbours:
                if neighbour not in visited_nodes:
                    node_distance[neighbour] = distance
                    new_nodes += [neighbour]
        nodes_to_visit = new_nodes
        distance += 1
    return node_distance








