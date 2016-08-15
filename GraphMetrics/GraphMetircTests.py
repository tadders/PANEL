import unittest
import numpy as np
from ClusteringCoefficent import *
from PathLength import *
from SmallWorld import _calc_small_world
from DiceSimilarityCoefficient import *

class TestGraphMetrics(unittest.TestCase):

    def setUp(self):
        self.cluster_graph = \
            np.array([[0, 1, 1, 1, 0],
                      [1, 0, 1, 0, 0],
                      [1, 1, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [0, 0, 1, 1, 0]])
        self.path_graph = np.array([[0, 1, 0, 0, 0, 0],
                                    [1, 0, 1, 0, 0, 0],
                                    [0, 1, 0, 1, 1, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 1],
                                    [0, 0, 0, 0, 1, 0]])
        self.dsc_graph1 = np.array([[0, 1],
                                    [1, 1],
                                    [1, 0],
                                    [0, 0]])
        self.dsc_graph2 = np.array([[0, 1],
                                    [0, 1],
                                    [1, 0],
                                    [1, 0]])

    def test_local_clustering_coefficent_fully_connected(self):
        self.assertEqual(local_clustering_coefficent(self.cluster_graph, 1), 1.0)

    def test_local_clustering_coefficent_partially_connected(self):
        self.assertEqual(local_clustering_coefficent(self.cluster_graph, 2), 1 /3.0)

    def test_local_clustering_coefficent_no_connections(self):
        graph = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0]])
        self.assertEqual(local_clustering_coefficent(graph, 0), 1)

    def test_global_clustering_coefficent(self):
        self.assertEqual(global_clustering_coefficent(self.cluster_graph), 1/3.0)

    def test_local_path_length_short(self):
        self.assertEqual(local_path_length(self.path_graph, 0, 2), 2)

    def test_local_path_length_long(self):
        self.assertEqual(local_path_length(self.path_graph, 4, 0), 3)

    def test_average_path_length(self):
        self.assertEqual(avergage_path_length(self.path_graph), 62 / 30.0)

    def test_calc_small_world(self):
        self.assertEqual(_calc_small_world(4, 2, 2, 2), 2)

    def test_DSC_parcel(self):
        self.assertEqual(DSC_parcel(self.dsc_graph1[:, 0], self.dsc_graph2[:, 0]), 0.25)

    def test_DSC_parcellation(self):
        self.assertEqual(DSC_parcellation(self.dsc_graph1, self.dsc_graph2), 0.375)
