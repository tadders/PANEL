from DataLoader.utils import load_cifti_if_necessary
import numpy as np
import os
from MatrixMaths import normalise
import nibabel as nib

def dualregression(parcellation, input_dtseries,
                   output_spatial_file, normalize=True, save_stage1=True,
                   save_stage2=True):
    """
    http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/DualRegression/UserGuide
    :param parcellation: A cifti format parcellation object or filee.g. from melodic
    :param input_dtseries: A .dtseries.nii object/file to regress against
    :param output_spatial_file: template filename to save to, spatial map is aved to
    this filename, timeseries is saved with a .txt extension
    :param normalize: Flag for normalization  If you don't normalise them,
    then you will only test for RSN "shape" in your cross-subject testing.
    If you do normalise them, you are testing for RSN "shape" and "amplitude".
    :param save_stage1: Flag to save the regressed timeseries
    :param save_stage2: Flag to save the regressed spatial map
    :returns: (regressed_timeseries, regressed_map)
    """

    parcellation = load_cifti_if_necessary(parcellation)
    input_dtseries = load_cifti_if_necessary(input_dtseries)
    tcs = input_dtseries.get_data()
    if normalize:
        tcs = normalise(tcs)
    betatcs = np.linalg.pinv(tcs).dot(tcs)
    if save_stage1:
        filename, extension = os.path.splitext(output_spatial_file)
        np.savetxt(filename + ".txt", np.transpose(betatcs))
    betatcs = normalise(betatcs, 1)
    betaICA = np.transpose(np.transpose(np.linalg.pinv(betatcs)) * np.transpose(tcs))
    header = parcellation.get_header()
    img = nib.CIFTI2Image(data=betaICA, header=header)
    img.to_filename(output_spatial_file)






