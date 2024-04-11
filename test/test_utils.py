import numpy as np
from numpy.typing import NDArray

import mug_cakes

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
        r = np.array([0.29223721, 0.11693068, 0.31209208, 0.27874003])
        res = mug_cakes.utils.to_measurements(r, 8)
        exp = np.array(
            [[2, 0, 2, 2], [0, 1, 1, 0], [1, 1, 0, 1], [0, 1, 0, 0], [1, 1, 0, 0]],
            dtype=np.uint64,
        )
        self.assert_np_array_equals(res, exp)

    def test_format(self):
        m = np.array(
            [[2, 0, 2, 2], [0, 1, 1, 0], [1, 1, 0, 1], [0, 1, 0, 0], [1, 1, 0, 0]],
            dtype=np.uint64,
        )
        ingredients = ["a", "b", "c", "d"]
        expected = """                    a                   b                   c                   d                   
ones:               2                   0                   2                   2                   
halves:             0                   1                   1                   0                   
quarters:           1                   1                   0                   1                   
eighths:            0                   1                   0                   0                   
16ths:              1                   1                   0                   0                   
"""
        res = mug_cakes.utils.format_ingredients(m, ingredients)
        self.assertEqual(expected, res)
