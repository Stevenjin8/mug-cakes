import unittest

import numpy as np

from mug_cakes import BayesianOptimizer


class TestStringMethods(unittest.TestCase):
    def test_convergence(self):
        """
        A reasonable 1-D test
        """

        np.random.seed(42)
        def target(x):
            return -((x * 20 - 10) ** 2) / 50 + np.sin(x * 20)

        noise = 0.3
        scale = 1 / 20
        bo = BayesianOptimizer(1, noise, scale)
        X_star = np.linspace(0, 1, 300).reshape(-1, 1)

        for _ in range(500):
            new_x = bo.ucb(z=5, X_star=X_star)
            new_y = target(new_x) + (np.random.normal(0, noise))
            bo.add(new_x[None], new_y)

        m, _ = bo.predict(X_star)

        true_best = X_star[np.argmax(target(X_star))]
        estimated_best = X_star[np.argmax(m)]

        self.assertAlmostEqual(true_best[0], estimated_best[0], places=2)



if __name__ == "__main__":
    unittest.main()
