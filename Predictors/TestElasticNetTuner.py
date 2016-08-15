import unittest
import numpy as np
from ElasticNetTuner import ElasticNetTuner
class TestElasticNetTuner(unittest.TestCase):

    def setUp(self):
        self.X = \
            np.array([[1, 0.75, -1, 0],
                      [2, 1.5, -2, 0],
                      [3, 2.5, -3, 0],
                      [4, 3.5, -4, 0]])
        self.Y = [1,2,3,4]


    def test_discard_weakest_half(self):
        tree = [[1, 2], [0, 3], [0], [1]]
        net = ElasticNetTuner(n_jobs=-1, cv=2)
        strong = net._discard_weakest_half(self.X, self.Y)

        self.assertEqual(strong, [0 ,2])