"""Utility functions"""
import numpy as np
from numpy.typing import NDArray

def rowpeat(A: NDArray, M: int) -> NDArray:
    """
    Repeat a rows of A a total of M times.
    """
    ret = np.empty((A.shape[0] * M, *A.shape[1:]), dtype=A.dtype)
    for i in range(A.shape[0]):
        ret[M * i : M * (i + 1)] = A[i]
    return ret

def cartesian_product(*xs: NDArray) -> list[NDArray]:
    ret = list(np.meshgrid(*xs))
    for i in range(len(ret)):
        ret[i] = ret[i].reshape(-1)
    return ret

