import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from PTuner import _gen_fine_intervals
import warnings




class RandomForestTuner(BaseEstimator, RegressorMixin):

    def __init__(self, n_jobs=-1, classifier=False, cv=5, random_state=42):
        """

       :param n_jobs: How many processors to use in parallel for training,
       defaults to use as many as possible
       :param classifier: True for classification task, False for regression
       :param cv: Number of folds to use in the GridSearchCV
       :return:
       """
        self.fitted = False
        self.grid = None
        self.cv = cv
        self.forest = None
        self.n_jobs = n_jobs
        self.classifier = classifier
        self.random_state = random_state

    def fit(self, X, y):
        """Fit estimator.
            Parameters
            ----------
            X : array-like or sparse matrix, shape=(n_samples, n_features)
                The input samples. Use ``dtype=np.float32`` for maximum
                efficiency. Sparse matrices are also supported, use sparse
                ``csc_matrix`` for maximum efficiency.
            y : array-like or sparse matrix shape=(n_samples, 1)
                Input samples are examples of the predictive variable
            classifier: Whether to tune a regressor or classifier, defaults to
            regressor as classifier problems can sometimes be formulated as a
            regression problem
            Returns
            -------
            self : object
                Returns self.
        """
        coarse_grid = self._tune_coarse(X, y)
        fine_grid = self._tune_fine(coarse_grid.best_params_, X, y)
        if coarse_grid.best_score_ > fine_grid.best_score_:
            self.grid = coarse_grid
        else:
            self.grid = fine_grid
        self.forest = self.grid.best_estimator_
        self.fitted = True

    def fit_transform(self, X, y):
        """
            Fits and then returns an unaltered copy of X
        """
        self.fit(X, y)
        return X

    def predict(self, X):
        print "predict"
        self._check_is_fitted()
        if self.grid :
            return self.grid.predict(X)
        else:
            warnings.warn("Forest not refitted with all the training data")
            return self.forest.predict(X)


    def predict_proba(self, X):
        self._check_is_fitted()
        if not self.classifier:
            raise RuntimeError("Only a Random Forest Classifier supports predict_proba")
        if self.grid and not self.refit:
            warnings.warn("Forest not refitted with all the training data")
            return self.forest.predict_proba(X)
        else:
            return self.grid.predict_proba(X)

    def predict_log_proba(self, X):
        self._check_is_fitted()
        if not self.classifier:
            raise RuntimeError("Only a Random Forest Classifier supports predict_log_proba")
        if self.grid and not self.refit:
            warnings.warn("Forest not refitted with all the training data")
            return self.forest.predict_log_proba(X)
        else:
            return self.grid.predict_log_proba(X)

    def transform(self, X):
        """

        :param X: X : array-like or sparse matrix, shape=(n_samples, n_features)
                The input samples. Use ``dtype=np.float32`` for maximum
                efficiency. Sparse matrices are also supported, use sparse
                ``csc_matrix`` for maximum efficiency.
        :return: No transformation of X ocurrs in the tuner so X is returned
                unmodified.
        """
        return X

    def score(self, X, y):
        self._check_is_fitted()
        if self.grid:
            return self.grid.score(X, y)
        else:
            return self.forest.score(X, y)

    def set_params(self, **params):
        warnings.warn("RandomForestTuner autotunes the forest so does not"
                      "support set_params, to set your own params use the"
                      "RandomForestClassifier Class")


    #https://www.researchgate.net/publication/230766603_How_Many_Trees_in_a_Random_Forest
    # Num trees to use
    def _tune_coarse(self, X, y):
        estimator = None
        scoring = ""
        criterion = []
        if self.classifier:
            estimator = RandomForestClassifier(random_state=self.random_state)
            scoring = "accuracy"
            criterion = ["gini", "entropy"] # https://www.garysieling.com/blog/sklearn-gini-vs-entropy-criteria
        else:
            estimator = RandomForestRegressor(random_state=self.random_state)
            criterion = ["mse"]
            scoring = "r2"
        param_dict = {"n_estimators": [128],
                      "criterion": criterion,
                      "max_features": [None, "sqrt", "log2"],
                      "min_samples_leaf": [1, 4, 8],
        }
        grid = GridSearchCV(estimator, param_dict, scoring=scoring, cv=self.cv, n_jobs=self.n_jobs, error_score=0, refit=True)
        grid.fit(X, y)
        return grid

    def _tune_fine(self, param_dict, X, Y):
        estimator = None
        scoring = ""
        if self.classifier:
            scoring = "accuracy"
            estimator = RandomForestClassifier(random_state=self.random_state)
        else:
            estimator = RandomForestRegressor(random_state=self.random_state)
            scoring = "r2"
        fine_params = dict()
        fine_params["n_estimators"] = [param_dict["n_estimators"]]
        max_features = self._calc_max_features(param_dict["max_features"], X.shape[1])
        fine_params["max_features"] = _gen_fine_intervals(max_features, max_features / 20, num_intervals=6, upper_limit=X.shape[1])
        fine_params["criterion"] = [param_dict["criterion"]]
        fine_params["min_samples_leaf"] = [param_dict["min_samples_leaf"]]
        grid = GridSearchCV(estimator, fine_params, scoring=scoring, cv=self.cv, n_jobs=self.n_jobs, error_score=0, refit=True)
        grid.fit(X, Y)
        return grid


    def _calc_max_features(self, max_features_param, num_features):
        if isinstance( max_features_param, int ):
            return max_features_param
        elif max_features_param == "log2":
            return int(np.log2(num_features))
        elif max_features_param == "sqrt":
            return int(np.sqrt(num_features))
        else:
            return num_features

    def clear_grid(self):
        """
        :return: Deletes the grid to free memory if using large training set
        """
        self.grid = None

    def _check_is_fitted(self):
        """
        Raises error if performing function e.g. predict before being fitted
        :return:
        """
        if not self.fitted:
            raise RuntimeError("Error Tuner must fit, before predicting")