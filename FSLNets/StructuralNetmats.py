import numpy as np

SUM = 'SUM'
MEAN='MEAN'

def gen_struct_netmat(connectivty_matrix, parcellation, method=SUM):
    """

    :param connectivty_matrix: Connectivity matrix to produce a netmat for
    :param parcellation: A CIFTI (dscalar) or numpy parcellation containing
    n parcels
    :param method: String for which method to use to produce netmat
    either SUM which is the sum of the structural conenction weights
    between parcels or MEAN which is the mean
    :return: netmat in the form of a (1, n^2) numpy array
    """

    parcels = get_parcels(parcellation)
    discrete = False # flag for discrete parcellations for which
                     #  optimisations can be used
    if len(np.unique(parcels)) == 2:
        discrete = True
        parcels = parcels.astype(bool)
    num_parcels = parcels.shape[1]
    netmat = np.zeros((num_parcels ** 2, ))
    for i in range(num_parcels):
        if discrete:
            parcel_connections = connectivty_matrix[parcels[:, i], :]
        else:
            parcel_connections = connectivty_matrix * parcels[:, i]
        for j in range(num_parcels):
            if discrete:
                conns = parcel_connections[:, parcels[:, j]]
            else:
                conns =  parcel_connections * parcels[:, j].T
            if method is SUM:
                netmat[i * num_parcels + j] = sum(sum(conns))
            elif method is MEAN:
                netmat[i * num_parcels + j] = np.mean(np.mean(conns))
            else:
                raise ValueError("Incorrect method string, select from SUM or MEAN")
    return netmat








def get_parcels(parcellation):
    if len(parcellation) == 1:
        return parcellation.get_data()
    else:
        return parcellation