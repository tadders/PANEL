import unittest
import numpy as np
from StructuralNetmats import *
from numpy.testing import assert_array_equal

class TestStructuralNetmats(unittest.TestCase):

    def setUp(self):
        self.conn_mat = np.array([[1, 1, 0.5, 0.5],
                                  [1, 1, 0.2, 0.2],
                                  [0.1, 0.1, 1, 1],
                                  [0.1, 0.1, 1, 1]])
        self.parcellation = np.array([[1, 0],
                                      [1, 0],
                                      [0, 1],
                                      [0, 1]])

    def test_sum_netmat(self):
        netmat = gen_struct_netmat(self.conn_mat, self.parcellation, method="SUM")
        correct_net = np.array([4, 1.4, 0.4, 4])
        assert_array_equal(netmat, correct_net)


    def test_mean_netmat(self):
        netmat = gen_struct_netmat(self.conn_mat, self.parcellation, method="MEAN")
        correct_net = np.array([1, 0.35, 0.1, 1])
        assert_array_equal(netmat, correct_net)