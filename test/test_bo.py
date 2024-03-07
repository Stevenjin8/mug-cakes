import numpy as np

from mug_cakes import bo

from . import utils


class TestGP(utils.NpTestCase):
    X = (
        np.array(
            [
                [1, 1],
                [1, 2],
                [3, 2],
                [3, 9],
            ]
        )
        / 10
    )  # so its in (0, 1)
    var_b = 1
    y = np.array([1.1, 1.0, 0.9, 1])
    B = np.array([0, 1, 0, 1])
    N_b = 2

    def test_target(self):
        """Test gradients"""

        eps = 0.000001  # can't make this too small.
        xa = np.zeros(3)
        grad = bo._dhp_target(xa, self.X, self.y, self.N_b, self.B, self.var_b)
        for i in range(3):
            xb = xa.copy()
            xb[i] += eps
            est = (
                bo._hp_target(xb, self.X, self.y, self.N_b, self.B, self.var_b)
                - bo._hp_target(xa, self.X, self.y, self.N_b, self.B, self.var_b)
            ) / eps
            self.assertAlmostEqual(est, grad[i], places=3)

    def test_optimize(self):
        res = bo.optimize_rbf_params(self.X, self.y, self.N_b, self.B, self.var_b)
        res = np.log(res)
        for i in range(3):
            other = res.copy()
            other[i] += 0.0001
            self.assertLess(
                bo._hp_target(res, self.X, self.y, self.N_b, self.B, self.var_b),
                bo._hp_target(other, self.X, self.y, self.N_b, self.B, self.var_b),
            )
