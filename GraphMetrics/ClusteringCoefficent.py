import numpy as np

def global_clustering_coefficent(graph, threshold=1):
    """

    :param graph: a nxn numpy matrix
    :param threshold: value used to determine presence of an edge between two
    two nodes, if edge weight < threshold presence of edge = 0 otherwise 1
    :return: double
    """
    local_coefficent_sum = 0.0
    for node in range(graph.shape[0]):
        local_coefficent_sum += local_clustering_coefficent(graph, node, threshold=threshold)
    return local_coefficent_sum / graph.shape[0]

def local_clustering_coefficent(graph, node, threshold=0.5):
    """
    :param graph: a nxn numpy matrix graph[i,j] = connection between
     node i and node j
    :param node: the node to calcualte the local clustering coefficent for
    :return: a double
    """
    num_nodes = graph.shape[0]
    neighbours = np.where(graph[node, :] >= threshold)[0]
    num_neighbour_edges = 0.0
    for neighbour in neighbours:
        neighbours_connections = np.where(graph[neighbour, :] >= threshold)[0]
        connected_neighbours = \
            np.intersect1d(neighbours, neighbours_connections)
        num_neighbour_edges += len(connected_neighbours)
    num_neighbours = len(neighbours)
    if len(neighbours) > 0:
        return num_neighbour_edges / (num_neighbours * (num_neighbours - 1))
    else:
        return 1
