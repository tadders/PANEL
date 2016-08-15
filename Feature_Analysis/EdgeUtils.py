import numpy as np

def get_top_n_features(feature_scores, n_features):
    sorted_importance_indices = np.argsort(feature_scores.flatten())
    sorted_importance_indices = sorted_importance_indices[::-1]  # Reverse list to make it descending
    top_n_indices = sorted_importance_indices[0:n_features]
    return top_n_indices, feature_scores.flatten()[top_n_indices]

def convert_indices_to_edges(indices, num_parcels=None, netmats=None):
    """

    :param index: index into netmats
    :param num_parcels: optional
    :param netmats: [1, n * n] array like objects where n is the number of
    parcels in the nemtat
    must provide one of netmats or num_parcels
    :return: return a edge in the form of (parcel1, parcel2)
    """
    if netmats is not None:
        nf = np.sqrt(netmats.shape[1])
        n = round(nf)
        if n == nf:      # is netmat square....
            num_parcels = n
        else:
            raise ValueError('Netmats should contain n*n edges where n is the number of parcels in the netmat ')
    return [ convert_index_to_edge_pair(index, num_parcels) for index in indices]

def convert_index_to_edge_pair(index, num_parcels=None, netmats=None):
    """

    :param index: index into netmats
    :param num_parcels: optional
    :param netmats: [1, n * n] array like objects where n is the number of
    parcels in the nemtat
    must provide one of netmats or num_parcels
    :return: return a edge in the form of (parcel1, parcel2)
    """
    if netmats is not None:
        nf = np.sqrt(netmats.shape[1])
        n = round(nf)
        if n == nf:      # is netmat square....
            num_parcels = n
        else:
            raise ValueError('Netmats should contain n*n edges where n is the number of parcels in the netmat ')
    parcel1 = int(np.floor(index / num_parcels)) + 1 #1st Parcel +1 assumes parcels are labelled 1 to num_parcels
    parcel2 = index % num_parcels + 1
    return (parcel1, parcel2)

def get_shared_edges(edge_list1, edge_list2):
    shared_edge_list = []
    for e1 in edge_list1:
        for e2 in edge_list2:
            if equal_edge(e1, e2):
                shared_edge_list = shared_edge_list + [e1]
                break
    return shared_edge_list

def get_shared_parcels(edge_list1, edge_list2):
    """
    :param edge_list1: list of tuples
    :param edge_list2: list of tuples
    :return: Returns an array of common parcels between both sets of edge
    """
    return np.intersect1d(np.unique(edge_list1), np.unique(edge_list2))



def equal_edge(edge1, edge2):
    same_edge = edge1[0] == edge2[0] and edge1[1] == edge2[1]
    inverted_edge = edge1[0] == edge2[1] and edge1[1] == edge2[0]
    return same_edge or inverted_edge