import numpy as np

from mug_cakes.simplex import dreparam, dsimplex_wrap, reparam, reparaminv, simplex_wrap

from . import utils


class TestSimplex(utils.NpTestCase):
    def test_reparaminv(self):
        r = np.array([1 / 3, 1 / 2, 1 / 4, 0, 0, 0])
        self.assert_np_array_equals(r, reparaminv(reparam(r)))
        self.assert_np_array_equals(
            np.array([0.5, 1, 0]), reparaminv(np.array((0.5, 0.5, 0, 0)))
        )

    def test_decorators(self):
        def identity(x):
            return x

        identity1 = simplex_wrap(identity)
        r = np.array([1 / 3, 1 / 2, 1 / 4, 0, 0, 0])
        self.assert_np_array_equals(reparam(r), identity1(r))

        def didentity(x):
            return np.eye(x.shape[0])

        didentity1 = dsimplex_wrap(didentity)
        self.assert_np_array_equals(dreparam(r), didentity1(r))

    def test_reparam(self):
        r = np.array([1 / 4, 1 / 3, 1 / 2])
        expected = np.ones(4) / 4
        result = reparam(r)
        self.assert_np_array_equals(expected, result)

        r = np.array([0.5])
        expected = np.ones(2) / 2
        result = reparam(r)
        self.assert_np_array_equals(expected, result)

        r = np.ones(4) * 0.5
        expected = np.array([1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 16])
        result = reparam(r)
        self.assert_np_array_equals(expected, result)

        r = np.array([0.0, 0.0, 1.0, 0.0])
        expected = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        result = reparam(r)
        self.assert_np_array_equals(expected, result)

    def test_dreparam(self):
        """Limit approximation"""
        r = np.array([0.22, 0.41, 0.44, 0.1, 0.34])
        delta = np.array([0.18471509, 0.00830847, 0.5466092, 0.79519768, 1.6941829])
        eps = 0.00000001
        expected = (reparam(r + eps * delta) - reparam(r)) / eps
        result = dreparam(r) @ delta
        self.assert_np_array_equals(expected, result, places=6)

        r = np.array([0.60946231, 0.03726698, 0.3193043, 0.57608681, 0.50879197])
        for i in range(r.shape[0]):
            delta = np.zeros(r.shape[0])
            delta[i] += 1
            expected = (reparam(r + eps * delta) - reparam(r)) / eps
            result = dreparam(r) @ delta
            self.assert_np_array_equals(expected, result, places=6)
