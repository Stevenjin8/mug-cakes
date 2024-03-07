from typing import Optional, Tuple

import numpy as np
import scipy
from numpy.typing import NDArray

from . import gp, kernel


def full_cov(
    K: NDArray[np.float64],
    J: int,
    B: NDArray[np.uint64],
    s2e: float,
    var_b: float,
):
    assert J == 0 or K.shape[0] % J == 0, "Wrong shape"
    assert K.shape[0] == B.shape[0], "Wrong shape"
    assert B.max() < J, "Invalid indices"
    N = K.shape[0] // J

    Sigma = np.zeros((N * J + J + N * J, N * J + J + N * J))
    Sigma[: J * N, : J * N] = s2e * np.eye(J * N)
    Sigma[J * N : J * N + J, J * N : J * N + J] = np.eye(J) * var_b
    Sigma[-N * J :, -N * J :] = K
    A = np.zeros((N * J + J, N * J + J + N * J))
    A[: N * J + J, : N * J + J] = np.eye(N * J + J)

    mix = np.zeros((N * J, J + N * J))
    mix[np.arange(0, N * J, dtype=np.uint64), B] = 1
    mix[:, J:] = np.eye(N * J)
    A[: N * J, N * J :] = mix

    return A @ Sigma @ A.T


###################################################################################
# Optimizing Hyper Parameters
# There are a bunch of issues with the but the main one is that all the parameters
# have to be positive, so we optimize their logs.
###################################################################################


def _hp_target(
    x0: NDArray[np.float64],
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    N_b: int,
    B: NDArray[np.uint64],
    var_b: float,
) -> float:
    N = X.shape[0]
    s2f, scale, s2e = np.exp(x0)
    Ka = kernel.rbf(X, X, scale, s2f)
    vara = full_cov(Ka, N_b, B, s2e, var_b)[:N, :N]
    # FIXME should pass these as prior parameters.
    return -gp.mvn_multi_log_unnormalized_pdf(y, np.array(0), vara[None])[0] + 0.5 * (
        scale**2 + N * s2e**2
    )


def _dhp_target(
    x0: NDArray[np.float64],
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    N_b: int,
    B: NDArray[np.uint64],
    var_b: float,
) -> NDArray[np.float64]:
    N = X.shape[0]
    log_s2f, log_scale, log_s2e = x0
    s2f = np.exp(log_s2f)
    scale = np.exp(log_scale)
    s2e = np.exp(log_s2e)
    K = kernel.rbf(X, X, scale, s2f)
    dKds2f = K / s2f
    dKdscale = (
        K
        * (-0.5)
        * scipy.spatial.distance.cdist(X, X, "sqeuclidean")
        * (-1 / scale**2)
    )
    dvards2f = full_cov(dKds2f, N_b, B, 0, 0)[:N, :N]
    dvardscale = full_cov(dKdscale, N_b, B, 0, 0)[:N, :N]
    dvards2e = full_cov(np.zeros((N, N)), N_b, B, 1, 0)[:N, :N]
    var = full_cov(K, N_b, B, s2e, var_b)[:N, :N]
    precision = np.linalg.inv(var)
    ds2f = gp.likelyhood_grad(y, np.array(0), precision, dvards2f)
    dscale = gp.likelyhood_grad(y, np.array(0), precision, dvardscale) - scale
    ds2e = gp.likelyhood_grad(y, np.array(0), precision, dvards2e) - N * s2e
    return -np.array([s2f * ds2f, scale * dscale, s2e * ds2e])


def optimize_rbf_params(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    N_b: int,
    B: NDArray[np.float64],
    var_b: float,
    bounds: Tuple[Tuple[float, float], ...] = ((-1.0, 4.0), (-7.0, 4.0), (-7.0, 4.0)),
    x0: Optional[NDArray[np.float64]] = None,
    disp: bool=False,
):
    if x0 is None:
        x0 = np.zeros(3)
    res = scipy.optimize.minimize(
        _hp_target,
        x0,
        method="L-BFGS-B",
        jac=_dhp_target,
        options={"disp": disp},
        args=(X, y, N_b, B, var_b),
        bounds=bounds,
    )
    assert res.success, "Did not converge"
    return np.exp(res.x)
