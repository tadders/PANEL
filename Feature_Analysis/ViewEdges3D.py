import numpy as np
from FSLNets.nets_3Dedge import nets_3Dedge
from DataLoader.utils import load_cifti_if_necessary

def view_edge_3D(parcellation, surface, edge):
    """

    :param parcellation: A CIFTI parcellation (dscalar) or numpy matrix
    contianing the parcels
    :param edge: Tuple edge for which you want to view (to view single parcel
    link edge to itself)
    :return: Produces a view of the surface
    """
    parcellation = load_cifti_if_necessary(parcellation)
    data = parcellation.get_data()
    if data.ndim > 1 and data.shape[1] > 1:
        max_parcellation = np.zeros((data.shape[0], ))
        for i in range(data.shape[0]):
            if not np.all(data[i, :] == 0):
                max_parcellation[i] = np.argmax(data[i, :], axis=0) + 1 # indexing first parcel from 1 a) to amthc matlab
                                                                          # b) to reserve parcel 0 for areas not included
                                                                          # in the parcellation
    else:
        max_parcellation = data
    index0 = max_parcellation == edge[0]
    index1 = max_parcellation == edge[1]
    index = (index0 + index1) > 0
    max_parcellation[~index] = 0
    max_parcellation[index0] = 25
    max_parcellation[index1] = 100 # Changin values of parcel id to make colour map clearer
    parcellation.data = data
    nets_3Dedge(parcellation, surface)

if __name__ == '__main__':
    #parcellation = '/vol/bitbucket/ta2812/Data/Salim_par/3LAYER/L150par.dscalar.nii'
    parcellation = '/home/tadlington/bitbucket/Structural_ICA/l_migp_nsub50_dim_2562.nii.gz_d.ica/d137_max_ica.dscalar.nii'
    getfrom = '/vol/vipdata/data/HCP100/100307/structural/MNINonLinear/fsaverage_LR32k/'
    gl = getfrom + '100307' + '.L.inflated.32k_fs_LR.surf.gii'
    #gl = getfrom + '100307' + '.L.inflated.32k_fs_LR.surf.gii'
    #gr = getfrom + '100307' +
    gl = '/home/tadlington/scp_files/L_inf32.surf.gii'
    gr = '/home/tadlington/scp_files/L_inf32.surf.gii'
    view_edge_3D(parcellation, [gl], (2, 10))
