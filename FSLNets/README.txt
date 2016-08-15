This code is a python adaption of the FSLNets code developed by the FMRIB Analysis Group.
Our acknowledgements and thanks goes to the developers for providing the opensource MATLAB code.
%%%  Licence is same as FSL licence
%%%  See documentation at  http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLNets

Some key differences in this implementation are:

That nets_netmats does not support the following methods of generating netmats:
Pairwise Causality
L1-norm  Regularised  partial correlation
as these relied on external MATLAB libraries

nets_edgepics has been replaced with nets_3Dedge which shows the edge viewed on a 3d surface
as opposed to 2D. Nets_3dEdge does not rely on previously running FSL's slices_summarry but will require
a valid GIFTI file to view the edge on.

