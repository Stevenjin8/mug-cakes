import numpy as np
import scipy
from numpy.typing import NDArray


def _rbfv(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    scale: NDArray[np.float64],
    s2f: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate RBF kernel for x,y with many different parameters.
    Artifact for when I was doing a grid search to optimize hyperparams.
    """
    K = scipy.spatial.distance.cdist(X, Y, "sqeuclidean")
    K = np.exp(-0.5 * np.multiply.outer(1 / scale, K))
    K = s2f.reshape(-1, 1, 1) * K
    return K


def rbf(
    X: NDArray[np.float64], Y: NDArray[np.float64], scale2: float, s2f: float
) -> NDArray[np.float64]:
    """
    RBF kernel.
    Parameters:
        X: N x D
        Y: M x D

    Returns:
        N x M
    """
    scalev2 = np.array([scale2])
    s2fv = np.array([s2f])
    return _rbfv(X, Y, scalev2, s2fv)[0]


def drbf(
    x: NDArray[np.float64], Y: NDArray[np.float64], scale2, s2f
) -> NDArray[np.float64]:
    """
    RBF gradient wrt firstparameter.

    Parameters:
        x: D
        Y: M x D

    Returns:
        M x D
    """
    assert len(x.shape) == 1
    assert len(Y.shape) == 2
    assert x.shape[0] == Y.shape[1]

    x = x[None]
    return rbf(x, Y, scale2, s2f).reshape(-1, 1) / (-scale2) * (x - Y)
