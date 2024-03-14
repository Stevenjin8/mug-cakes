import functools

import numpy as np
from numpy._typing import NDArray


def reparam(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Diffeomorphism from an D dimensional unit hyper cube to D dimensional simplex.
    """
    x = np.zeros(r.shape[0] + 1)
    x[0] = r[0]
    s = r[0]
    for i in range(1, x.shape[0] - 1):
        x[i] = r[i] * (1 - s)
        s += x[i]
    x[-1] = 1 - s
    # return ss
    return x


def reparaminv(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r = np.empty(x.shape[0] - 1)
    s = 0
    for i in range(r.shape[0]):
        if 1 - s <= 0:
            r[i] = 0
        else:
            r[i] = x[i] / (1 - s)
        s += x[i]
    return r


def dreparam(r: NDArray[np.float64]) -> NDArray[np.float64]:
    D = r.shape[0]

    # cache?
    M = np.tril(np.ones((D, D, D)))
    for i in range(0, D):
        M[i, :, i] = 0
    M[:, 0, :] = 0

    dS = np.triu(np.exp(M @ np.log(1 - r)))
    dr = np.zeros((D + 1, D))
    dr[1:-1, :] = (dS[:, 0:-1] * -r[1:]).T
    np.fill_diagonal(dr, np.diag(dS))
    dr[-1] = -dS[:, -1]
    # dr[-1, -1] *= -1

    return dr


def simplex_wrap(func):
    @functools.wraps(func)
    def wrapped(x, *args, **kwargs):
        return func(reparam(x), *args, **kwargs)

    return wrapped


def dsimplex_wrap(func):
    @functools.wraps(func)
    def wrapped(x, *args, **kwargs):
        return func(reparam(x), *args, **kwargs) @ dreparam(x)

    return wrapped
