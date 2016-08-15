import numpy as np


def DSC_parcel(parcel1, parcel2):
    """
    :param parcel1: binary array like object of shape(n x 1)
    :param parcel2: binary array like object of shape(n x 1)
    1 indicates point is part of the pacel
    :return: the Dice similarity coefficent between the two parcels
    """
    return 2 * np.dot(parcel1, parcel2) / float((sum(parcel1) ** 2 + sum(parcel2) **2))

def DSC_parcellation(parcellation1, parcellation2):
    """

    :param parcellation1:
    :param parcellation2:
    :return:
    """
    assert parcellation1.shape[1] == parcellation2.shape[1]
    num_parcels = parcellation1.shape[1]
    avg_dsc = 0.0
    for i in range(num_parcels):
        avg_dsc += DSC_parcel(parcellation1[:, i], parcellation2[:, i])
    return avg_dsc/ num_parcels