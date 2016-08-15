import numpy as np
import nibabel as nib
import subprocess
import tempfile

def nets_glm(netmats, design_matrix, contrast_matrix, view=False, nperms=5000):

    num_subjects, num_edges = netmats.shape
    tf = tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz')
    filename = tf.name
    tf.close()
    netmat_img = nib.Nifti1Image(np.reshape(netmats.T, (num_edges, 1, 1, num_subjects)), np.eye(4))
    nib.save(netmat_img, filename)
    fsl_cmd ='. /etc/fsl/5.0/fsl.sh'
    randomise_cmd = 'randomise -i {} -o {} -d {} -t {} -x --uncorrp -n {}'.format(
        filename, filename, design_matrix, contrast_matrix, nperms)
    glob_cmd = 'imglob {}_vox_corrp_tstat*.* | wc -w'.format(filename)
    all_cmds = '{}; {}; {}'.format(fsl_cmd, randomise_cmd, glob_cmd)
    randomise = subprocess.Popen(all_cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable='/bin/bash')

    ncontrast, err = randomise.communicate()
    if err is not '':
        raise ValueError(err)
    ncontrast = ncontrast.split('\n')[-2]
    ncontrast = int(ncontrast)
    p_uncorrected = np.zeros([ncontrast, num_edges])
    p_corrected = np.zeros([ncontrast, num_edges])
    for i in range(ncontrast):
        p_uncorrected[i, :] = nib.load('{}_vox_p_tstat{}.nii.gz'.format(filename, i + 1)).get_data()
        p_corrected[i, :] = nib.load('{}_vox_corrp_tstat{}.nii.gz'.format(filename, i + 1)).get_data()
        fdr_cmd = 'fdr -i {}_vox_p_tstat{} -q 0.05 --oneminusp | grep -v Probability'.format(filename, i + 1)
        try:
            FDRthresh =  subprocess.check_output('{}; {}'.format(fsl_cmd, fdr_cmd), shell=True)
        except subprocess.CalledProcessError, e:
            print "Ping stdout output:\n", e.output

    return p_uncorrected, p_corrected, FDRthresh



