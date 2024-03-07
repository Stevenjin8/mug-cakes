import unittest

import numpy as np

from mug_cakes.kernel import rbf, rbfv


class TestKernel(unittest.TestCase):
    """Behaviour fixing test cases"""

    X = np.array(
        [
            [0, 1],
            [1, 1],
            [3, 2],
        ]
    )
    Y = np.array(
        [
            [0, 1],
            [3, 1],
            [3, -9],
        ]
    )

    def test_rbfv(self):
        scale = np.array([1, 2])
        s2f = np.array([0.5, 4])
        result = rbfv(self.X, self.Y, scale, s2f)
        expected = np.array(
            [
                [
                    [5.00000000e-01, 5.55449827e-03, 1.07132377e-24],
                    [3.03265330e-01, 6.76676416e-02, 1.30513953e-23],
                    [3.36897350e-03, 3.03265330e-01, 2.65554612e-27],
                ],
                [
                    [4.00000000e00, 4.21596898e-01, 5.85511406e-12],
                    [3.11520313e00, 1.47151776e00, 2.04363561e-11],
                    [3.28339994e-01, 3.11520313e00, 2.91508964e-13],
                ],
            ]
        )
        self.assertEqual(expected.shape, result.shape)
        self.assertAlmostEqual(np.abs(expected - result).max(), 0)

    def test_rbf(self):
        scale = 1.0
        s2f = 0.5
        result = rbf(self.X, self.Y, scale, s2f)
        expected = np.array(
            [
                [5.00000000e-01, 5.55449827e-03, 1.07132377e-24],
                [3.03265330e-01, 6.76676416e-02, 1.30513953e-23],
                [3.36897350e-03, 3.03265330e-01, 2.65554612e-27],
            ],
        )
        self.assertEqual(expected.shape, result.shape)
        self.assertAlmostEqual(np.abs(expected - result).max(), 0)

