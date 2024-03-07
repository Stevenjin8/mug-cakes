import numpy as np
from numpy.typing import NDArray

from mug_cakes.gp import conditional_covar, conditional_mean, conditional_var

from . import utils


class TestGP(utils.NpTestCase):
    joint_cov: NDArray
    mu: NDArray
    x2: NDArray

    def __init__(self, *args, **kwargs):
        M = np.array(
            [
                [1, 3, 4, 1, 2],
                [4, 1, 2, 4, 0],
                [1, 3, 4, 1, 2],
                [1, 0, -4, -1, 2],
                [1, 3, 4, 10, 2],
            ]
        )
        self.joint_cov = M.T @ M
        self.mu = np.array([0, 1, 0, 1, 2])
        self.x2 = np.array([1, 2, 3])
        super().__init__(*args, **kwargs)

    def test_conditional_mean(self):
        """Behaviour fixing test cases"""
        precision2 = np.linalg.inv(self.joint_cov[2:, 2:])
        cov12 = self.joint_cov[3:, :3]
        mu1 = self.mu[:2]
        mu2 = self.mu[2:]
        result = conditional_mean(self.x2, precision2, cov12, mu1, mu2)
        expected = np.array([3.86305582, 1.98864711])
        self.assert_np_array_equals(expected, result)

    def test_conditional_covar(self):
        """Behaviour fixing test cases"""
        precision2 = np.linalg.inv(self.joint_cov[2:, 2:])
        var1 = np.linalg.inv(self.joint_cov[:2, :2])
        cov12 = self.joint_cov[3:, :3]
        result1 = conditional_covar(var1, precision2, cov12)
        expected1 = np.array(
            [[-251.41333041, -64.55358867], [-64.55358867, -17.23550946]]
        )
        self.assert_np_array_equals(expected1, result1)

        expected2 = np.diag(expected1)
        result2 = conditional_var(var1, precision2, cov12)
        self.assert_np_array_equals(expected2, result2)
