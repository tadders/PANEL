import unittest
import numpy as np
from numpy.testing import assert_array_equal
from utils import *

class TestUtils(unittest.TestCase):

    def test_get_unique_edges(self):
        netmat = np.array([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]]).reshape((1, 16))
        unique_edges = get_unique_edges(netmat)
        correct_edges = np.array([[0.0, 1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 6.0, 0.0]])
        assert_array_equal(correct_edges, unique_edges)