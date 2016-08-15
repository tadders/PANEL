import math
from abc import ABCMeta, abstractmethod
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class Tuner(BaseEstimator, TransformerMixin):

    def __init__(self, scale=None, pca=None):
        self.preprocessors = []
        self.fitted = False
        if scale:
            self.preprocessors.append(("scale", StandardScaler))
        if pca:
            self.preprocessors.append(("pca", PCA(0.95)))
        pass

    def fit(self, X, Y, classifier=True):
        self.fitted = True

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, fit_params)
        return self.transform(X)

    def transform(self):
        if not self.fitted:
            raise ValueError("Error must fit, before predicting")
        return self.preprocess_pipeline.transform()

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Error must fit, before predicting")
        return self.transform(X)

    def predict_proba(self, X):
        return self.predict(X)

    def predit_log_proba(self, X):
        return self.predict(X)


    def _gen_fine_intervals(self, units, interval_size, num_intervals=8, upper_limit=None, lower_limit=None):
        print "TODO fix lower limit for _gen_fine-intervals"
        if isinstance(units, int):
            cast = int
        elif isinstance(units, float):
            cast = float
        else:
            raise TypeError("units must be numeric")
        if int(interval_size) == 0:
            interval_size = 1
        num_new_values = num_intervals -1
        lower_l = units - interval_size * math.ceil(num_new_values / 2.0)
        lower_limit = max(x for x in [lower_l, lower_limit] if x is not None )
        upper_l = units + interval_size * math.floor(num_new_values / 2.0)
        upper_limit = min(x for x in [upper_l, upper_limit] if x is not None)

        return list(range(int(lower_limit), int(upper_limit), int(interval_size)))

    def _is_fitted(self):
        self.fitted = True;

