from Predictors.RandomForestTuner import RandomForestTuner
import numpy as np
from EdgeUtils import convert_indices_to_edges
from EdgeUtils import get_top_n_features


class RFFeatureExtraction:

    def __init__(self, netmats=None, measures=None, classifier=False, forest=None):
        if forest is not None:
            self.forest = forest
        elif netmats is not None and measures is not None:
            self._tune_forest(netmats, measures, classifier=classifier)

        else:
            raise ValueError("Error must provide either random_forest or "
                             "both netmats and measures arguments")

    def _tune_forest(self, netmats, measures, classifier):
        tuner = RandomForestTuner(classifier=classifier)
        tuner.fit(netmats, measures)
        self.tuner = tuner
        self.forest = tuner.forest

    def get_feature_importance(self):
        """
        :return: list of features in order of predictive importance.
        """
        return self.forest.feature_importances_

    def get_top_n_features(self, num_features):
        """
        :param num_features: Number of features to return
        :return: A sorted list of num_features tuples (feature_index, importance)
        """
        feature_importance = self.forest.feature_importances_
        return get_top_n_features(feature_importance)

    def get_feature_importance(self):
        return self.forest.feature_importances_

    def get_top_n_edges(self, num_edges):
        """

        :param num_edges: Number of edges to return
        :return: a list of edges in the form: ((parcel1, parcel2), importance)
        for the importance of the edge from parcel1 to parcel2
        """
        top_features = self.get_top_n_features(num_edges)
        fake_netmat = np.zeros([1, self.forest.n_features_])
        feature_indices = [index for (index, imp) in top_features]
        top_edges = convert_indices_to_edges(feature_indices, fake_netmat)
        return zip(top_edges, [imp for (feat, imp) in top_features])





if __name__ == '__main__':
    r = RFFeatureExtraction(1, 2)




