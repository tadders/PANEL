from itertools import product

import numpy as np
from sklearn.grid_search import GridSearchCV
from sknn.mlp import Classifier, Regressor, Layer
from PTuner import _gen_fine_intervals
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin


class NNTuner(BaseEstimator, RegressorMixin):

    def __init__(self, n_jobs=-1, classifier=False, cv=5, random_state=42):
        """

        :param n_jobs: How many processors to use in parallel for training,
        defaults to use as many as possible
        :param classifier: True for classification task, False for regression
        :param cv: Number of folds to use in the GridSearchCV
        :return:
        """
        pca = PCA(0.95)
        scaler = StandardScaler()
        self.preprocess_pipeline = Pipeline([("pca", pca), ("scaler", scaler)])
        self.grid = None
        self.fitted = False
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state
        self.classifier = classifier
        self.NN = None

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
            Returns
            -------
            self : object
                Returns self.
        """
        self.preprocess_pipeline.fit(X, y)
        coarse_grid = self._tune_coarse(X, y)
        fine_grid = self._tune_fine(coarse_grid.best_params_, X, y)
        if coarse_grid.best_score_ > fine_grid.best_score_:
            self.grid = coarse_grid
        else:
            self.grid = fine_grid
        self.NN = self.grid.best_estimator_
        self.fitted = True

    def fit_transform(self, X, y):
        """
            Fits the preprocessing pipe and Neural Network
            Returns the transformed X values
        """
        self.fit(X, y )
        return self.preprocess_pipeline.transform(X)

    def predict(self, X):
        self._check_is_fitted()
        X = self.transform(X)
        if self.grid:
            return self.grid.predict(X)
        else:
            warnings.warn("NN not refitted with all the training data")
            return self.NN.predict(X)

    def predict_proba(self, X):
        self._check_is_fitted()
        X = self.transform(X)
        if self.grid:
            return self.grid.predict_proba(X)
        else:
            warnings.warn("NN not refitted with all the training data")
            return self.NN.predict_proba(X)

    def predict_log_proba(self, X):
        self._check_is_fitted()
        X = self.transform(X)
        if self.grid:
            warnings.warn("Forest not refitted with all the training data")
            return self.forest_.predict_log_proba(X)
        else:
            return self.grid.predict_log_proba(X)

    def transform(self, X):
        """

        :param X:  X : array-like or sparse matrix, shape=(n_samples, n_features)
                The input samples. Use ``dtype=np.float32`` for maximum
                efficiency. Sparse matrices are also supported, use sparse
                ``csc_matrix`` for maximum efficiency.
        :return: Transformed X values from the transformations in the preprocessing
        pipeline, i.e. a PCA reduction to 95% variance and feature scaling
        """
        self._check_is_fitted()
        return self.preprocess_pipeline.transform(X)

    def score(self, X, y):
        self._check_is_fitted()
        X = self.transform(X)
        if self.grid:
            return self.grid.score(X, y)
        else:
            return self.NN.score(X, y)

    def set_params(self, **params):
        warnings.warn("RandomForestTuner autotunes the forest so does not"
                      "support set_params, to set your own params use the"
                      "RandomForestClassifier Class")


    def _tune_coarse(self, X, y):
        """
        X: Training Data in the form of a numpy matrix of n_samples * n_features
        Y: labels numpy array of n samples
        """

        clf = None
        output_layer = None
        scoring = ""
        types = ["Sigmoid", "Rectifier", "Tanh", "ExpLin"]
        if self.classifier:
            clf = Classifier([], random_state=30)
            types.remove("Rectifier")
            scoring = "accuracy"
            output_layer = Layer("Softmax", name="out")
        else:
            clf = Regressor([], random_state=30)
            types.remove("Sigmoid")
            scoring = "r2"
            output_layer = Layer("Linear", name="out")
        layers = []
        num_features = X.shape[1]
        num_outputs = self._calc_num_outputs(self.classifier, y)
        for type in types:
            for units in [int(num_features * 0.25), int((num_features + num_outputs) * 0.5),
             int(num_features * 0.75), num_features, int(num_features * 1.5)]:
                for hunits in [None, int(num_features * 0.25), int((num_features + num_outputs) * 0.5),
                    int(num_features * 0.75), num_features, int(num_features * 1.5)]:
                    layer = [Layer(type, units=units, name="layer1")]
                    if hunits:
                        layer.append(Layer(type, units=hunits, name="layer2"))
                    layer.append(output_layer)
                    layers.append(layer)

        param_dict = {"layers":layers,
                      "learning_rule": ["sgd", "momentum", "nesterov", "adadelta", "adagrad", "rmsprop"],
                      "learning_rate": [ 0.00001, 0.001, 0.0001],
                      "n_iter": [50, 150],
        }
        grid = GridSearchCV(clf, param_dict, scoring=scoring, cv=self.cv,
                            n_jobs=self.n_jobs, error_score=0)
        grid.fit(X, y)
        return grid

    def _tune_fine(self, param_dict, X, y):

        if self.classifier:
            clf = Classifier([], random_state=30)
            scoring = "accuracy"
        else:
            clf = Regressor([], random_state=30)
            scoring = "r2"

        fine_dict = dict()
        fine_dict["layers"] = self._create_fine_tuning_layers(param_dict["layers"])
        fine_dict["learning_rule"] = [param_dict["learning_rule"]]
        if param_dict["learning_rule"] == "momentum":
            fine_dict["learning_momentum"] = [0.9, 0.85, 0.95]
        fine_dict["regularize"] =[None, "L2"],
        fine_dict["random_state"] = [30]
        fine_dict["learning_rate"] = [param_dict["learning_rate"], 0.0001]
        fine_dict["n_iter"] = [param_dict["n_iter"]]
        grid = GridSearchCV(clf, fine_dict, scoring=scoring, cv=self.cv,
                            n_jobs=self.n_jobs, error_score=0)
        grid.fit(X, y)
        return grid


    def _calc_num_outputs(self, is_classifier, labels):
        if is_classifier:
            return len(np.unique(labels))
        else:
            return 1

    def _create_fine_tuning_layers(self, coarse_layers):

        layers = []
        for layer_combo in product(*[_gen_fine_intervals(layer.units) for layer in coarse_layers[:-1]]):
            layer = []
            for i in range(0,len(layer_combo)):
                layer.append(Layer(coarse_layers[i].type, units= layer_combo[i], name= "layer{0}".format(i)))
            layer.append(Layer(coarse_layers[-1].type, name="output"))
            layers.append(layer)
        return layers


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
