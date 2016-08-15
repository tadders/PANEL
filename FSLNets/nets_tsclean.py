import numpy as np


def nets_tsclean(ts, aggressive):
    """

    :param ts: TimeSeriesData
    ts.DD = list of good nodes
    ts.UNK = list of unknown components (gets deleted but never regressed out of good). Can be empty.
    :param aggressive=0: "soft" - just deletes the bad and the unknown components
    % aggressive=1: regresses the bad timeseries (not in DD or UNK) out of the good, and deletes the bad & unknown components
%
    :returns: timeseries with bad nodes removed, optionally regressing those out of the good (for further cleanup)
    """

    ts.NnodesOrig = ts.Nnodes
    nongood = _setdiff(range(ts.Nnodes), ts.DD)  # bad or unknown components
    bad = _setdiff(nongood, ts.UNK)  # only bad components
    num_good = len(ts.DD)
    num_points = ts.ts.shape[0]
    newts = np.zeros([num_points, num_good])
    for s in range(ts.Nsubjects):
        grot = ts.ts[s * ts.NtimepointsPerSubject:
            (s + 1) * ts.NtimepointsPerSubject - 1, :]  # all comps
        goodTS = grot[:, ts.DD]
        badTS = grot[:, bad]  # bad components
        if aggressive:
            mat = np.dot(np.linalg.pinv(badTS), goodTS)
            regressed_subject_ts = goodTS - np.dot(badTS, mat)
            newts[s * ts.NtimepointsPerSubject:
                (s + 1) * ts.NtimepointsPerSubject - 1, :] = regressed_subject_ts
        else:
            newts[s * ts.NtimepointsPerSubject:
            (s + 1) * ts.NtimepointsPerSubject - 1, :] = goodTS
    ts.ts = newts
    ts.Nnodes = newts.shape[1]


def _setdiff(list1, list2):
    s = set(list2)
    return [x for x in list1 if x not in s]
