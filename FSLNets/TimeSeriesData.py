import os
import glob
import numpy as np


class TimeSeriesData:

    def __init__(self, ts_dir=None, tr=1, varnorm=1, nruns=1):
        if ts_dir:
            self = TimeSeriesData.nets_load(ts_dir, tr, varnorm, nruns)
        else:
            self.ts = []
            self.tr = []
            self.Nsubjects = 0
            self.Nnodes = 0
            self.NnodesOrig = self.Nnodes
            self.Ntimepoints = 0
            self.NtimepointsPerSubject = 0
            self.DD = []
            self.UNK = []

    @staticmethod
    def nets_load(ts_dir, tr, varnorm, nruns=1):
        ts = TimeSeriesData()
        ts_files = sorted(glob.glob(ts_dir + '/*.txt'))
        nsubs = len(ts_files)
        TS = []
        for i in range(nsubs):
            grotALL = np.genfromtxt(ts_files[i], delimiter=',')
            gn = grotALL.shape[0]
            if i == 0:
                num_maps = grotALL.shape[1]
                TS = np.zeros([nsubs * gn, num_maps])
                ts.NtimepointsPerSubject = gn
            elif gn != ts.NtimepointsPerSubject:
                raise ValueError(
                    'Error: not all subjects have the same number of timepoints!')
            gn /= nruns
            for j in range(1, nruns + 1):
                grot = grotALL[(j - 1) * gn: j * gn - 1, :]
                grot = grot - grot.mean()  # demean
                if varnorm == 1:
                    grot = grot / np.std(grot, ddof=1)  # normalize whole subject std
                elif varnorm == 2:
                    # normalise each seperate timeseries from each subject
                    grot = np.divide(grot, np.tile(np.std(grot, ddof=1), (grot.shape[0], 1)))
                TS[(i * nruns + j - 1) * gn: (i * nruns + j) * gn - 1, :] = grot
        ts.ts = TS
        ts.tr = tr
        ts.Nsubjects = nsubs * nruns
        ts.Nnodes = TS.shape[1]
        ts.NnodesOrig = ts.Nnodes
        ts.Ntimepoints = TS.shape[0]
        ts.NtimepointsPerSubject /= nruns
        ts.DD = range(ts.Nnodes)
        ts.UNK = []
        return ts

    @staticmethod
    def nets_joint_load(ts_dirs, tr, varnorm, nruns=1):
        """
          :param ts_dirs: list of dualregressed files to combine for each subject 
        """
        ts = TimeSeriesData()
        ts_files = [sorted(glob.glob(ts_dir + '/*.txt')) for ts_dir in ts_dirs]
        nfiles = len(ts_files)
        nsubs = len(ts_files[0])
        TS = []
        for i in range(nsubs):
            grotALL = []
            for f in range(nfiles):
                if f == 0:
                    grotALL = np.genfromtxt(ts_files[f][i], delimiter=',')
                else:
                    grotALL = np.vstack((grotALL, np.genfromtxt(ts_files[f][i], delimiter=',')))
            gn = grotALL.shape[0]
            if i == 0:
                num_maps = grotALL.shape[1]
                TS = np.zeros([nsubs * gn, num_maps])
                ts.NtimepointsPerSubject = gn
            elif gn != ts.NtimepointsPerSubject:
                raise ValueError(
                    'Error: not all subjects have the same number of timepoints!')
            gn /= nruns
            for j in range(1, nruns + 1):
                grot = grotALL[(j - 1) * gn: j * gn - 1, :]
                grot = grot - grot.mean()  # demean
                if varnorm == 1:
                    grot = grot / np.std(grot, ddof=1)  # normalize whole subject std
                elif varnorm == 2:
                    # normalise each seperate timeseries from each subject
                    grot = np.divide(grot, np.tile(np.std(grot, ddof=1), (grot.shape[0], 1)))
                TS[(i * nruns + j - 1) * gn: (i * nruns + j) * gn - 1, :] = grot
        ts.ts = TS
        ts.tr = tr
        ts.Nsubjects = nsubs * nruns
        ts.Nnodes = TS.shape[1]
        ts.NnodesOrig = ts.Nnodes
        ts.Ntimepoints = TS.shape[0]
        ts.NtimepointsPerSubject /= nruns
        ts.DD = range(ts.Nnodes)
        ts.UNK = []
        return ts
