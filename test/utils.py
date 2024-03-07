import unittest
import numpy as np

class NpTestCase(unittest.TestCase):
    """Test case with numpy utils"""
    def assert_np_array_equals(self, expected, result):
        self.assertEqual(expected.shape, result.shape)
        self.assertAlmostEqual(np.abs(expected - result).max(), 0)


