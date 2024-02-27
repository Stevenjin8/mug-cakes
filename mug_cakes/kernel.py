import numpy as np
from numpy.typing import NDArray


class RbfKernel:
    def __init__(self):
        # TODO
        pass

    def __call__(self, x1: NDArray, x2: NDArray) -> NDArray:
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(1, -1)
        return np.exp(-0.5 * (x1 - x2) ** 2)
