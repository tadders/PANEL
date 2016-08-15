import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
import warnings


class ElasticNetTuner(BaseEstimator, RegressorMixin):

    def __init__(self, n_jobs=1, cv=5, random_state=42):
        self.strongest_feature_indices_ = []
        self.fitted = False
        self.net = None
        self.cv= cv
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, Y):
        X = self._discard_weakest_half(X, Y)
        elastic_net = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                                   cv=self.cv, n_jobs=self.n_jobs,
                                   random_state=self.random_state)
        elastic_net.fit(X, Y)
        self.net = elastic_net
        self.fitted = True

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        self._check_is_fitted()
        return X[:, self.strongest_feature_indices_]

    def predict(self, X):
        self._check_is_fitted()
        X = self.transform(X)
        return self.net.predict(X)

    def score(self, X, y):
        self._check_is_fitted()
        X = self.transform(X)
        return self.net.score(X, y)

    def _discard_weakest_half(self, X, Y):
        num_features = X.shape[1]
        corrs =np.zeros((num_features, ))
        for i in range(num_features):
            corr, p = pearsonr(X[:, i], Y)
            corrs[i] = corr
        # constant value features have a corr of nan replace it with 0
        corrs = np.nan_to_num(corrs)
        abs_corrs = abs(corrs)
        indices = np.argsort(abs_corrs)[::-1]
        num_features = len(indices)
        num_keep = int(round(num_features / 2))
        self.strongest_feature_indices_ = indices[0:num_keep]
        return X[:, self.strongest_feature_indices_]

    def _check_is_fitted(self):
        """
        Raises error if performing function e.g. predict before being fitted
        :return:
        """
        if not self.fitted:
            raise RuntimeError("Error Tuner must fit, before predicting")

if __name__ == '__main__':
    from DataLoader.MetadataHelper import *
    from DataLoader.utils import load_netmats
    from sklearn.preprocessing import LabelEncoder
    from sklearn.cross_validation import cross_val_predict
    import numpy as np

    # Load netmats and meta data
    metadata = load_patient_metadata('../Data/joint_HCP_500_metadata.csv', subject_measures=["Gender"])

    # metadata = metadata.as_matrix()
    # metadata = np.ravel(metadata)

    le = LabelEncoder()
    metadata = le.fit_transform(metadata["Gender"].values)

    netmats = load_netmats('/home/tadlington/bitbucket/HCP_500/HCP500_460_partial_corr_netmat.txt')
    #tuner = ElasticNetTuner(n_jobs=-1, cv=2)
    #predictions = cross_val_predict(tuner, netmats, y=metadata, cv=2, verbose=2)
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.7)
    elastic_net.fit(netmats, metadata)
    #print predictions