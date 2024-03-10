from typing import Optional, Tuple

import numpy as np
import scipy
from numpy.typing import NDArray

from . import gp, kernel


def full_cov(
    K: NDArray[np.float64],
    N_b: int,
    B: NDArray[np.uint64],
    s2e: float,
    var_b: float,
):
    """Get the full covariance matrix of (biased) observations given Gram matrix.

    Parameters:
        K: N x N Gram matrix
        N_b: number of different observers
        B: N array showing which observations belong with which observers.
        s2e: observation noise.
        var_b: variance of biases

    Returns:
        (N + N_b) x (N + N_b) covariance matrix for observations and biases
    """
    # Check shapes
    assert N_b == 0 or K.shape[0] % N_b == 0, "Wrong shape"
    assert K.shape[0] == B.shape[0], "Wrong shape"
    assert B.max() < N_b, "Invalid indices"

    N = K.shape[0]

    # create covariance matrix for (e1, ..., eN, b1, ..., bN_b, (x1), ..., f(xN))
    Sigma = np.zeros((N + N_b + N, N + N_b + N))
    Sigma[:N, :N] = s2e * np.eye(N)
    Sigma[N : N + N_b, N : N + N_b] = np.eye(N_b) * var_b
    Sigma[-N:, -N:] = K
    A = np.zeros((N + N_b, N + N_b + N))
    A[: N + N_b, : N + N_b] = np.eye(N + N_b)

    # write (y1, .., yN, b1, ..., bN_b) as a linear combination of
    # (e1, ..., eN, b1, ..., bN_b)
    mix = np.zeros((N, N_b + N))
    mix[np.arange(0, N, dtype=np.uint64), B] = 1
    mix[:, N_b:] = np.eye(N)
    A[:N, N:] = mix

    # variance of linearly transformed normal variable.
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
    """Log scaled inputs for negative log posterior"""
    N = X.shape[0]
    s2f, scale, s2e = np.exp(x0)

    K = kernel.rbf(X, X, scale, s2f)
    vara = full_cov(K, N_b, B, s2e, var_b)[:N, :N]
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
    """Gradient or Jacobian"""
    N = X.shape[0]
    log_s2f, log_scale2, log_s2e = x0
    s2f = np.exp(log_s2f)
    scale2 = np.exp(log_scale2)
    s2e = np.exp(log_s2e)
    K = kernel.rbf(X, X, scale2, s2f)
    dK_ds2f = K / s2f
    dK_dscale = (
        K
        * (-0.5)
        * scipy.spatial.distance.cdist(X, X, "sqeuclidean")
        * (-1 / scale2**2)  # sic
    )
    dvar_ds2f = full_cov(dK_ds2f, N_b, B, 0, 0)[:N, :N]
    dvar_dscale = full_cov(dK_dscale, N_b, B, 0, 0)[:N, :N]
    dvar_ds2e = full_cov(np.zeros((N, N)), N_b, B, 1, 0)[:N, :N]
    var = full_cov(K, N_b, B, s2e, var_b)[:N, :N]
    precision = np.linalg.inv(var)

    # this is where we insert the prior loss.
    dpost_ds2f = gp.likelyhood_grad(y, np.array(0), precision, dvar_ds2f)
    dpost_dscale = gp.likelyhood_grad(y, np.array(0), precision, dvar_dscale) - scale2
    dpost_ds2e = gp.likelyhood_grad(y, np.array(0), precision, dvar_ds2e) - N * s2e
    return -np.array([s2f * dpost_ds2f, scale2 * dpost_dscale, s2e * dpost_ds2e])


def optimize_rbf_params(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    N_b: int,
    B: NDArray[np.uint64],
    var_b: float,
    bounds: Tuple[Tuple[float, float], ...] = ((-1.0, 4.0), (-7.0, 4.0), (-7.0, 4.0)),
    x0: Optional[NDArray[np.float64]] = None,
    disp: bool = False,
):
    """Optimize kernel hyperparameters.
    
    Parameters:
        X: N x D 
        y: N
        N_b: total number of observers
        B: N which observations correspond to which observers
        var_b: variance of biases
        bounds: Search bounds. I really regret adding this because L-BFGS-B supports constraints which I have to add anyways for stability.
        x0: initial guess. Defaults to zeros.
        disp: Display L-BFSG-B progress.
    """
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


def _diff_params(
    x_s: NDArray,
    x_M: NDArray,
    X: NDArray,
    y: NDArray,
    precision: NDArray,
    scale: float,
    s2f: float,
    lambda_: float = 1,
    gamma: float = 1,
) -> tuple[float, float]:
    """Mean and variance of f(x_s) - f(x_M) | D_n"""
    assert x_s.shape == x_M.shape
    assert len(x_s.shape) == len(x_M.shape) == 1
    assert len(X.shape) == len(precision.shape) == 2
    assert x_s.shape[0] == x_M.shape[0] == X.shape[1]
    assert precision.shape[0] == precision.shape[1] == X.shape[0]
    x_s = x_s[None]
    x_M = x_M[None]
    x = np.vstack((x_s, x_M))
    k_s = kernel.rbf(x_s, X, scale, s2f)
    k_M = kernel.rbf(x_M, X, scale, s2f)
    k = np.vstack((k_s, k_M))
    kk = kernel.rbf(x, x, scale, s2f)
    mu = gp.conditional_mean(y, precision, k)
    cov = gp.conditional_covar(kk, precision, k)
    v = cov[0, 0] + gamma * cov[1, 1] - lambda_ * 2 * cov[0, 1]
    return mu[0] - mu[1], v


def expected_diff(
    x_s: NDArray,
    x_M: NDArray,
    X: NDArray,
    y: NDArray,
    precision: NDArray,
    scale: float,
    s2f: float,
    lambda_: float = 1,
    gamma: float = 1,
) -> float:
    """Find E[max(f(x_s) - f(x_M), 0) | D_n]

    Parameters:
        x_s: D
        x_M: D
        X: N x D 
        y: N
        lambda_, gamma: exploration vs exploitation parameters.
    """
    return gp.expected_improvement(
        *_diff_params(x_s, x_M, X, y, precision, scale, s2f, lambda_, gamma)
    )


def dexpected_diff(
    x_s: NDArray,
    x_M: NDArray,
    X: NDArray,
    y: NDArray,
    precision: NDArray,
    scale: float,
    s2f: float,
    lambda_: float = 1,
    gamma: float = 1,
) -> NDArray[np.float64]:
    """
    Derivative with respect to x_s.
    See docstring for `expected_diff`.
    """
    mu, var = _diff_params(
        x_s, x_M, X, y, precision, scale, s2f, lambda_, gamma
    )

    k_s = kernel.rbf(x_s[None], X, scale, s2f).reshape(-1, 1)
    k_M = kernel.rbf(x_M[None], X, scale, s2f).reshape(-1, 1)
    dk_s_dx_s = kernel.drbf(x_s, X, scale, s2f)
    dmu_dx_s = dk_s_dx_s.T @ precision @ y
    dkappa_dx_s = kernel.drbf(x_s, x_M[None], scale, s2f)[0]
    dvar_dx_s = (
        -2 * k_s.T @ precision @ dk_s_dx_s
        - 2 * lambda_ * dkappa_dx_s
        + 2 * lambda_ * k_M.T @ precision @ dk_s_dx_s
    )
    dvar_dx_s = dvar_dx_s[0]
    dEI = gp.dexpected_improvement(mu, var)
    return dEI[0] * dmu_dx_s + dEI[1] * dvar_dx_s
