from typing import Optional

import numpy as np
import scipy
from numpy.typing import NDArray


class BayesianOptimizer:
    """
    Initial implementation of a Gaussian Process
    """

    _X: NDArray
    _y: NDArray
    _K: NDArray
    _noise: float
    _scale: float

    def __init__(self, d: int, noise: float, scale: float):
        self._X = np.empty((0, d))
        self._K = np.empty((0, 0))
        self._y = np.empty((0,))
        self._noise = noise
        self._scale = scale

    def d(self) -> int:
        return self._X.shape[1]

    def k(self, x: NDArray, y: NDArray) -> NDArray:
        """
        RBF kernel
        TODO: add kernel params
        """
        return np.exp(
            -0.5
            * scipy.spatial.distance.cdist(x, y, "sqeuclidean")
            / (self._scale**2)
        )

    def ucb(self, z: int, X_star: NDArray ) -> NDArray:
        # X_star = np.hstack(
        #     [
        #         v.reshape(-1, 1)
        #         for v in np.meshgrid(
        #             [np.linspace(0, 1, resolution) for _ in range(self.d())]
        #         )
        #     ]
        # )
        m, var = self.predict(X_star)
        ucb = m + z * (var**0.5)
        i_star = np.argmax(ucb)
        return X_star[i_star]

    def add(self, X: NDArray, y: NDArray) -> None:
        assert X.shape[0] == y.shape[0], "Arrays must have same first dimension"
        self._X = np.concatenate((self._X, X), axis=0)
        self._y = np.concatenate((self._y, y), axis=0)
        self._K = self.k(self._X, self._X) + self._noise * np.eye(self._X.shape[0])

    def predict(self, X_star: NDArray) -> tuple[NDArray, NDArray]:
        K_star = self.k(X_star, self._X)
        K_2star = self.k(X_star, X_star)
        y_star = K_star @ np.linalg.solve(self._K, self._y)  # FIXME could cache this?
        ## TODO: use Cholesky?
        ## todo optimize the np.diag
        var = np.diag(K_2star) - (K_star * np.linalg.solve(self._K, K_star.T).T).sum(
            axis=1
        )
        return y_star, var
