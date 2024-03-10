import numpy as np

def reparam(x):
    ret = np.zeros(x.shape[0] + 1)
    ret[0] = x[0]
    s = x[0]
    for i in range(1, ret.shape[0] - 1):
        ret[i] = x[i] * (1 - s)
        s += ret[i]
    ret[-1] = 1 - s
    return ret
