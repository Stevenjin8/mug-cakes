import numpy as np
import scipy
from numpy.typing import NDArray


def rbfv(x: NDArray, y: NDArray, scale: NDArray, s2f: NDArray) -> NDArray:
    """
    Calculate RBF kernel for x,y with many different parameters.
    """
    K = scipy.spatial.distance.cdist(x, y, "sqeuclidean")
    K = np.exp(-0.5 * np.multiply.outer( 1 / scale , K))
    K = s2f.reshape(-1, 1, 1) * K
    return K

def rbf(x: NDArray, y: NDArray, scale: float, s2f: float) -> NDArray:
    """Non vectorized rbf kernel"""
    scalev = np.array([scale])
    s2fv = np.array([s2f])
    return rbfv(x, y, scalev, s2fv)[0]

