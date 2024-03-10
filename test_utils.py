# import unittest
#
# import numpy as np
#
# from mug_cakes import utils, kernel, bo, gp
# from scipy import stats
#
#
# class TestUtils(unittest.TestCase):
#    def test_rowpeat(self):
#        A = np.array(
#            [
#                [1, 2, 3],
#                [4, 5, 6],
#            ],
#        )
#        B = utils.rowpeat(A, 3)
#        expected = np.array(
#            [
#                [1, 2, 3],
#                [1, 2, 3],
#                [1, 2, 3],
#                [4, 5, 6],
#                [4, 5, 6],
#                [4, 5, 6],
#            ],
#        )
#        self.assertTrue((B == expected).all())
#
#    def test_cartesian_product(self):
#        A = np.array([
#            [1, 2, 3],
#            [4, 5, 6],
#            [7, 8, 9],
#        ])
#        breakpoint()
#        B = np.array(utils.cartesian_product(*A))
#        # order is kinda whack
#        # fmt: off
#        expected = np.array([
#            [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3],
#            [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6],
#            [7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9],
#        ])
#        # fmt: on
#        self.assertTrue((B == expected).all())

# def test_move(self):
#    scale = 0.1
#    s2f = 25.
#    s2e = 0.3 ** 2
#    X = np.linspace(0, 1, 100).reshape(-1, 1)
#    K = kernel.rbf(X, X, scale, s2f)
#    var = K + s2e * np.eye(K.shape[0])
#    Y = stats.multivariate_normal.rvs(cov = var)
#    stf_star, scale_star, s2e_star = bo.optimize_rbf_params(1, 100, 0.1, 1, 0.001, 1, 100, X, Y, 1, 1)
#    breakpoint()
