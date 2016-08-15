import unittest
import numpy as np
from numpy.testing import assert_array_equal
from EdgeUtils import *

class TestEdgeUtils(unittest.TestCase):

    def setUp(self):
        self.netmat = np.array([])

    def test_equal_edge_true(self):
        self.assertTrue(equal_edge((1, 2), (1, 2)))

    def test_equal_edge__invert_true(self):
        self.assertTrue(equal_edge((1, 2), (2, 1)))

    def test_equal_edge_false(self):
        self.assertFalse(equal_edge((1, 2), (1, 4)))

    def test_equal_edge__invert_false(self):
        self.assertFalse(equal_edge((1, 4), (2, 1)))

    def test_convert_index_to_edge_pair(self):
        index = 1
        num_parcels = 3
        correct_edge = (1, 2)
        edge = convert_index_to_edge_pair(index, num_parcels=3)
        self.assertEqual(correct_edge, edge)

    def test_convert_index_to_edge_pair_second_row(self):
        index = 5
        num_parcels = 3
        correct_edge = (2, 3)
        edge = convert_index_to_edge_pair(index, num_parcels=3)
        self.assertEqual(correct_edge, edge)

    def test_get_top_n_features(self):
        scores = np.array([5, 1, 4, 10, 12, 2, 4])
        top_indices, top_scores = get_top_n_features(scores, 3)
        assert_array_equal(top_indices, np.array(4, 3, 0))
        assert_array_equal(top_scores, np.array(12, 10, 5))
