import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MyAxis3D import Axes3D as axe
from matplotlib import cm
from DataLoader.GIFTIHelper import get_gifti_faces
from DataLoader.GIFTIHelper import get_gifti_vertices
from DataLoader.utils import load_cifti_if_necessary
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import mode

def nets_3Dedge(cortical_parcellation, surfaces, cmap=""):
    """

    :param cortical_parcellation: "Filename or cifti object containing a parcellation"
    :param surfaces: "list of Filenames or cifti/gifti objects"
    :return: shows the two parcels connected by edge on the surfaces
    """
    if isinstance(surfaces, basestring):
        surfaces = [surfaces]
    num_surfaces =  len(surfaces)
    cortical_parcellation = load_cifti_if_necessary(cortical_parcellation)
    cortical_data = cortical_parcellation.get_data()
    print cortical_data.shape
    plots = []
    labels = ['Left', 'Right', 'Subcortical']
    fig = plt.figure()
    gs = gridspec.GridSpec(2, num_surfaces)
    cmap = cm.get_cmap("jet")
    smap = cm.ScalarMappable(norm=None, cmap=cmap)
    vertex_count = 0
    for i, surface in enumerate(surfaces):

        surf = load_cifti_if_necessary(surface)
        pointset = get_gifti_vertices(surf)
        triangles = get_gifti_faces(surf)

        x = pointset[:, 0]
        y = pointset[:, 1]
        z = pointset[:, 2]
        num_points = pointset.shape[0]
        #TODO when adding in parcellation look atr shade_rgb
        surf_cortical_data = cortical_data[vertex_count: vertex_count + num_points]
        vertex_count += num_points
        face_parcels = _interpolate_face_parcels(surf_cortical_data, triangles)
        col = smap.to_rgba(face_parcels)
        zero_col = smap.to_rgba(0)
        col[face_parcels == 0, :] = cm.colors.ColorConverter().to_rgba('#D3D3D3')
        #col = np.ones(col.shape)

        plots.append(fig.add_subplot(gs[0, i],  projection='3d'))

        ax = plots[i * 2]
        ax.set_axis_off()
        plots[i * 2].view_init(0, (i + 1) * 180)

        tri = plots[i * 2].plot_trisurf(x, y, z, triangles=triangles, linewidth=0, shade=True,  facecolors= col, antialiased=False)
        ax.set_title(labels[i] +' Hemisphere Frontal View', color = 'c')

        plots.append(fig.add_subplot(gs[1, i],  projection='3d'))
        ax = plots[i * 2 + 1]
        ax.set_axis_off()
        ax.view_init(0, i * 180)
        tri = plots[i * 2 + 1].plot_trisurf(x, y, z, triangles=triangles, linewidth=0, shade=True,  facecolors= col, antialiased=False)
        ax.set_title(labels[i] +' Hemisphere Back View', color = 'c')

        """
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
        #plt.axis([xmin,xmax,ymin,ymax])
        fig.tight_layout()
        """
    plt.tight_layout()
    plt.show()

def _interpolate_face_parcels(parcellation, triangles):
    faceparcels = np.zeros((triangles.shape[0], ))
    for i, trinagle in enumerate(triangles):
        faceparcels[i] = mode(parcellation[triangles[i, :]])[0]
    return faceparcels


if __name__ == '__main__':
    parcellation = '/vol/bitbucket/ta2812/Data/Salim_par/3LAYER/L150par.dscalar.nii'
    getfrom = '/vol/vipdata/data/HCP100/100307/structural/MNINonLinear/fsaverage_LR32k/'
    gl = getfrom + '100307' + '.L.inflated.32k_fs_LR.surf.gii'
    gr = getfrom + '100307' + '.R.inflated.32k_fs_LR.surf.gii'
    #parcellation = '/home/tadlington/bitbucket/Salim_par/3LAYER/dim150_max_par.dscalar.nii'
    #gl = '/home/tadlington/scp_files/L_inf32.surf.gii'
    #gr = '/home/tadlington/scp_files/L_inf32.surf.gii'
    nets_3Dedge(parcellation, [gl])


