import unittest

import numpy as np
from numpy.typing import NDArray

from mug_cakes.kernel import _rbfv, drbf, rbf

from . import utils


class TestKernel(utils.NpTestCase):
    """Behaviour fixing test cases"""

    X: NDArray[np.float64] = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [3.0, 2.0],
        ]
    )
    Y: NDArray[np.float64] = np.array(
        [
            [0.0, 1.0],
            [3.0, 1.0],
            [3.0, -9.0],
        ]
    )

    def test_rbfv(self):
        scale = np.array([1.0, 2.0])
        s2f = np.array([0.5, 4.0])
        result = _rbfv(self.X, self.Y, scale, s2f)
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

    def test_drbfv(self):
        eps = 0.000001
        scale = 0.9
        s2f = 0.5
        base = rbf(self.X[1][None], self.Y, scale, s2f)[0]
        grad = drbf(self.X[1], self.Y, scale, s2f)
        self.assertEqual(grad.shape, (3, 2))

        x1 = self.X[1].copy()
        x1[1] += eps
        self.assert_np_array_equals(
            (rbf(x1[None], self.Y, scale, s2f)[0] - base) / eps, grad[:, 1], places=5
        )
        x0 = self.X[1].copy()
        x0[0] += eps
        self.assert_np_array_equals(
            (rbf(x0[None], self.Y, scale, s2f)[0] - base) / eps, grad[:, 0], places=5
        )
