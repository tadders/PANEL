import numpy as np
from TimeSeriesData import  TimeSeriesData
import warnings
from random import random

def nets_netmats(ts, do_r_to_z, method, var_arg=None):

    inmode = 0
    if not np.size(ts):
        grot = ts
        ts = TimeSeriesData()
        ts.Nsubjects = 1
        ts.ts = grot
        ts.Nnodes = grot.shape[1]
        ts.NtimepointsPerSubject = ts.ts.shape[0]
        inmode = 1
    N = ts.Nnodes
    just_diag = 0
    method_type = 0
    netmats = []
    for s in range(ts.Nsubjects):
        grot = ts.ts[s * ts.NtimepointsPerSubject: (s + 1) * ts.NtimepointsPerSubject, :]
        if method in ['cov', 'covariance', 'multiggm']:
            grot = np.cov(grot.T)
        elif method in ['amp', 'amplitude']:
            grot = np.std(grot, axis=0, ddof=1)
            just_diag = 1
        elif method in ['corr', 'correlation']:
            grot = np.corrcoef(grot.T)
            grot[np.eye(N) > 0] = 0
            method_type = 1
        elif method in ['rcorr']:
            mgrot = np.mean(grot, axis=1)
            rgrot = np.dot(np.linalg.pinv(mgrot[:, None]), grot)
            g = np.dot(mgrot[:, None], rgrot)
            grot -= g
            grot = np.corrcoef(grot.T)
            grot[np.eye(N) > 0] = 0
            method_type = 1
        elif method in ['icov','partial']:
            grot = np.cov(grot.T)
            if var_arg:
                warnings.warn('L1-norm regularized partial correlation'
                             'is not currently supported')
            else:
                grot = -np.linalg.inv(grot)
                diags = np.matrix(np.sqrt(abs(np.diag(grot))))
                grot = grot / diags / diags.transpose()
                grot[np.eye(N) > 0] = 0
                method_type = 2
        elif method in ['ridgep']:
            grot = np.cov(grot.T)
            grot /= np.sqrt((np.diag(grot) ** 2).mean())
            if var_arg:
                rho = var_arg
            else:
                rho = 0.1
            grot = - np.linalg.inv((grot + rho * np.eye(N)))
            diags = np.matrix(np.sqrt(abs(np.diag(grot))))
            grot = grot / diags / diags.transpose()
            grot[np.eye(N) > 0] = 0
            method_type = 2
        else:
            print 'unknown method {}'.format(method)
        if s == 0:
            if just_diag:
                netmats = np.zeros((ts.Nsubjects, N))
            else:
                netmats = np.zeros((ts.Nsubjects, N * N))
        if just_diag == 1:
            netmats[s, :] = grot
        else:
            netmats[s, :] = np.reshape(grot, [1, N * N])
    """
    if method is 'multiggm':
        if var_arg:
            rho = var_arg
        else:
            rho = 0.1
            TODO raise flag that this is not supported at the momment
    """


    if do_r_to_z and method_type > 0:
        arone = []

        for s in range(ts.Nsubjects):
            grot = ts.ts[s * ts.NtimepointsPerSubject: (s + 1) * ts.NtimepointsPerSubject, :]
            for i in range(N):
                g = grot[:, i]
                ar = sum(np.multiply(g[0:-1], g[1:]))
                ar /= sum(g ** 2 )
                arone += [ar]
        arone = np.median(arone)
        # create null data using the estimate AR(1) coefficent clear
        grotR = np.zeros((ts.Nsubjects, N * (N -1)))
        grotts = np.zeros((ts.Nsubjects * ts.NtimepointsPerSubject, N))
        for s in range(ts.Nsubjects):
            grot = np.zeros((ts.NtimepointsPerSubject, 1))
            for i in range(N):
                grot[0, 0] = random()
                for t in range(1, ts.NtimepointsPerSubject):
                    grot[t, 0] = grot[t - 1, 0] * arone + random()
                grotts[s * ts.NtimepointsPerSubject: (s + 1) * ts.NtimepointsPerSubject , i] = grot[:, 0]
            if method_type == 1:
                grotr = np.corrcoef(grotts[s * ts.NtimepointsPerSubject: (s + 1) * ts.NtimepointsPerSubject - 1, :].T)
            else:
                grotr = - np.linalg.inv(
                    np.cov(grotts[s * ts.NtimepointsPerSubject: (s + 1) * ts.NtimepointsPerSubject - 1, :].T))
                diags = np.matrix(np.sqrt(abs(np.diag(grotr))))
                grotr = grotr / diags / diags.transpose()
            grotR[s, :] = grotr[np.eye(N) < 1]
        grotZ = 0.5 * np.log((1 + grotR) / (1 - grotR))
        RtoZcorrection = 1 / np.std(grotZ)
        netmats = 0.5 * np.log((1 + netmats) / (1 - netmats)) * RtoZcorrection

    if inmode == 1:
        netmats = np.reshape(netmats, [1, N * N])

    return netmats

if __name__ == '__main__':
    ts = TimeSeriesData.nets_load('/home/tadlington/bitbucket/Salim_par/test_dr/', 0.4, 1)
    netmats = nets_netmats(ts, 1, "rcorr")