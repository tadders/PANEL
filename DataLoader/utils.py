import nibabel as nib
import numpy as np



def load_cifti_if_necessary(cifti):
    """
    :param cifti: Either an cifti/nifti object or a filepath to one
    :return: if passed a filename, loads the file and returns
    the cifti/nifti otherwise returns cifti unmodified.
    """
    if isinstance(cifti, basestring):
        cifti = nib.load(cifti)
    return cifti


def load_netmats(netmats_file, delimiter=","):
    """

    :param netmats_file: Path to netmats txt file
    :return: numpy matrix of netmats
    """
    return np.genfromtxt(netmats_file, delimiter=delimiter)


def get_unique_edges(netmats):
    """netmats tend to be symmetric, putting in duplicate features can
    reduce performance and increase training times. This takes the upper
    triangle and diagonal of edges
    :param netmats: numpy matrix of netmat edges either s*n **2
    where s is the number of subject netmats and n is the number of parcels
    :param k: diagonal offset, offset from leading diagonal to collects, default
    t 0 to include the main diagonal, 1 does not include the main idagonal
    :return 1 * ((n * n-1) /2) numpy matrix consisting of the upper triangle
    edges
    """
    num_subjects = netmats.shape[0]
    num_edges = netmats.shape[1]
    num_parcels = int(np.sqrt(num_edges))
    unique_edges = np.zeros((num_subjects, int((num_parcels * (num_parcels -1)) /2) + num_parcels))
    if num_parcels ** 2 != num_edges:
        raise ValueError("Error non square netmat provided")
    for i in range(num_subjects):
        netmat = netmats[i, :]
        edges = np.reshape(netmat, (num_parcels, num_parcels))
        unique_edges[i, :] = edges[np.triu_indices(num_parcels)]
    return  unique_edges




