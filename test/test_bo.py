import numpy as np
from numpy.typing import NDArray

from mug_cakes import bo
from mug_cakes.kernel import rbf

from . import utils


class TestGP(utils.NpTestCase):
    """Tests for Gaussian Processes module"""

    X: NDArray[np.float64] = (
        np.array(
            [
                [1.0, 1.0],
                [1.0, 2.0],
                [3.0, 2.0],
                [3.0, 9.0],
            ]
        )
        / 10
    )  # so its in (0, 1)
    var_b = 1.0
    y: NDArray[np.float64] = np.array([1.1, 1.0, 0.9, 1])
    B: NDArray[np.uint64] = np.array([0, 1, 0, 1])
    N_b = 2

    def test_target(self):
        """Test gradients by comparing grad function with limit approximation."""

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
        """Test that we can optimize kernel hyperparameters"""
        res = bo.optimize_rbf_params(self.X, self.y, self.N_b, self.B, self.var_b)
        res = np.log(res)
        for i in range(3):
            other = res.copy()
            other[i] += 0.0001
            self.assertLess(
                bo._hp_target(res, self.X, self.y, self.N_b, self.B, self.var_b),
                bo._hp_target(other, self.X, self.y, self.N_b, self.B, self.var_b),
            )

    def test_expected_diff(self):
        """No MC tests, bc thye take too long to be any good. Just make sure things make sense."""
        s2f = 0.1
        scale = 1
        s2e = s2f
        x_M = np.array([1.1, 1.1])
        x_s1 = np.array([1.1, 1.1])
        x_s2 = np.array([1.1, 1.100001])
        x_s3 = np.array([1.1, 1.2])
        x_s4 = np.array([1.1, 2])
        x_s5 = np.array([1.1, 200])
        x_s6 = np.array([1.1, 201])
        K = rbf(self.X, self.X, scale, s2f)
        var = bo.full_cov(K, self.N_b, self.B, s2e, self.var_b)[
            : -self.N_b, : -self.N_b
        ]

        diff1 = bo.minus_expected_diff(
            x_s1, x_M, self.X, self.y, np.linalg.inv(var), scale, s2f
        )
        diff2 = bo.minus_expected_diff(
            x_s2, x_M, self.X, self.y, np.linalg.inv(var), scale, s2f
        )
        diff3 = bo.minus_expected_diff(
            x_s3, x_M, self.X, self.y, np.linalg.inv(var), scale, s2f
        )
        diff4 = bo.minus_expected_diff(
            x_s4, x_M, self.X, self.y, np.linalg.inv(var), scale, s2f
        )
        diff5 = bo.minus_expected_diff(
            x_s5, x_M, self.X, self.y, np.linalg.inv(var), scale, s2f
        )
        diff6 = bo.minus_expected_diff(
            x_s6, x_M, self.X, self.y, np.linalg.inv(var), scale, s2f
        )

        self.assertEqual(diff1, 0)
        self.assertAlmostEqual(diff2, 0, places=4)
        self.assertTrue(diff1 > diff2 > diff3 > diff4 > diff5)
        self.assertAlmostEqual(diff5, diff6)

        lambda1 = 1
        lambda2 = 0.5
        lambda3 = 0
        x_s = np.array([1.1, 1.2])

        diff7 = bo.minus_expected_diff(
            x_s, x_M, self.X, self.y, np.linalg.inv(var), scale, s2f, lambda_=lambda1
        )
        diff8 = bo.minus_expected_diff(
            x_s, x_M, self.X, self.y, np.linalg.inv(var), scale, s2f, lambda_=lambda2
        )
        diff9 = bo.minus_expected_diff(
            x_s, x_M, self.X, self.y, np.linalg.inv(var), scale, s2f, lambda_=lambda3
        )
        self.assertTrue(0 > diff7 > diff8 > diff9)

    def test_dexpected_diff(self):
        """Use limit approx for test gradients"""
        s2f = 0.1
        scale = 0.9
        s2e = s2f
        K = rbf(self.X, self.X, scale, s2f)
        var = bo.full_cov(K, self.N_b, self.B, s2e, self.var_b)[
            : -self.N_b, : -self.N_b
        ]
        x_s = np.array([0.33, 0.87])
        x_M = np.array([0.12, 0.13])
        eps = 0.000000001
        kwargs = {"gamma": 0.99, "lambda_": 0.2394}
        grad = bo.dminus_expected_diff(
            x_s, x_M, self.X, self.y, np.linalg.inv(var), scale, s2f, **kwargs
        )
        base = bo.minus_expected_diff(
            x_s, x_M, self.X, self.y, np.linalg.inv(var), scale, s2f, **kwargs
        )

        x_s0 = x_s.copy()
        x_s0[0] += eps
        result0 = (
            bo.minus_expected_diff(
                x_s0, x_M, self.X, self.y, np.linalg.inv(var), scale, s2f, **kwargs
            )
            - base
        ) / eps
        self.assertAlmostEqual(grad[0], result0, places=5)

        x_s1 = x_s.copy()
        x_s1[1] += eps
        result1 = (
            bo.minus_expected_diff(
                x_s1, x_M, self.X, self.y, np.linalg.inv(var), scale, s2f, **kwargs
            )
            - base
        ) / eps
        self.assertAlmostEqual(grad[1], result1, places=5)

    def test_bo_state(self):
        """Basic tests for dataclass"""
        X = np.array([[0, 1], [0.0, 2]])
        y = np.array([0, 1.0])
        B = np.array([0, 1], dtype=np.uint64)
        N_b = 2
        var_b = 1

        state = bo.BoState(X.copy(), y.copy(), N_b, B.copy(), var_b, simplex=False)
        self.assert_np_array_equals(state.X, X)
        self.assert_np_array_equals(state.y, y)
        self.assert_np_array_equals(state.B, B)
        self.assertEqual(state.N, 2)
        self.assertEqual(state.N_b, 2)
        self.assertEqual(state.var_b, 1)

        x_new = np.array([0, 0.1])
        y_new = -0.1
        state.add(x_new, y_new, 1)
        self.assert_np_array_equals(state.X, np.array([[0, 1], [0.0, 2], [0, 0.1]]))
        self.assert_np_array_equals(state.y, np.array([0, 1.0, -0.1]))
        self.assert_np_array_equals(state.B, np.array([0, 1, 1]))
        self.assertEqual(state.B.dtype, np.uint64)
        self.assertEqual(state.N, 3)
        self.assertEqual(state.N_b, 2)

    def test_full_cycle(self):
        """Test that we can perform a BO experiment on simple data"""

        def target(x):
            """Has clear global max at x[0] = 0.67967968 but has local max at x[0]=0.16616617"""
            return (
                2 * np.sin(10 * x[..., 0] + 1)
                + 20 * x[..., 0] ** 0.5
                + 20 * (x[..., 1]) ** 0.3
                - 27
            )

        np.random.seed(42)

        s2e = 0.4**2
        bb = np.array([-2.0, 2.0, 0])
        var_b = 1**2
        N_b = 3
        N = 3
        X = np.random.rand(N, 1)
        X = np.hstack((X, 1 - X))
        B = np.array([0, 1, 2], dtype=np.uint64)
        y = (target(X) + np.random.normal(size=(X.shape[0])) * s2e**0.5) + bb[B]
        state = bo.BoState(X, y, N_b, B, var_b, simplex=True)

        # idk 8 might be too few
        for _ in range(8):
            x_new, _, x_M = bo.bo_iter(state)
            j = np.random.choice(N_b)
            y_new = target(x_new) + bb[j] + np.random.normal() * s2e**0.5
            np.array(j, dtype=np.uint64)
            state.add(x_new, y_new, j)
        x_new, _, x_M = bo.bo_iter(state)
        self.assertAlmostEqual(0.67967968, x_M[0], places=1)
        self.assertAlmostEqual(x_M.sum(), 1)
