"""
Functions relation to Gaussian Processes and Gaussian Distribution.
Work tracker:

[x] create basic gaussian functions
[x] functions for biases
[x] kernel parameter optmizers
[ ] acquisition functions with gradient optimization.
[ ] tests lol
"""

import numpy as np
import scipy
from numpy.typing import NDArray


def conditional_mean(
    x2: NDArray,
    precision2: NDArray,
    cov12: NDArray,
    mu1: NDArray = np.array(0),
    mu2: NDArray = np.array(0),
) -> NDArray:
    """Posterior mean of x1 given x2 where x1, x2 are jointly gaussian.

    Parameters:
        x2: the observed data
        precision2: precision matrix for x2
        cov12: covariance of entries in x1 and x2
        mu1: mean of x1
        mu2: mean of x2
    """
    return mu1 + cov12 @ precision2 @ (x2 - mu2)


def conditional_covar(
    var1: NDArray,
    precision2: NDArray,
    cov12: NDArray,
) -> NDArray:
    """
    Variance of x1 given x2 where x1, x2 are jointly Gaussian.

    Parameters:
        var1: variance of x1
        precision2: precision matrix of x2
        cov12: covariance of x1 and x2
    """
    return var1 - cov12 @ precision2 @ cov12.T


def conditional_var(
    var1: NDArray,
    precision2: NDArray,
    cov12: NDArray,
) -> NDArray:
    """
    Variance of x1 given x2 where x1, x2 are jointly Gaussian.
    That is, we return a vector because we don't care about the covariance terms.

    Parameters:
        var1: variance of the entries of x1.
            Since we only care about the diagonal, `var` can be a covariance matrix or the diagonal of a covariance matrix.
        precision2: precision matrix of x2.
    """

    if len(var1.shape) == 2:
        var1 = np.diag(var1)
    if len(var1.shape) > 2:
        raise ValueError("var1 must be square of vector")
    return var1 - (cov12 * (precision2 @ cov12.T).T).sum(axis=1)


def likelyhood_grad(
    y: NDArray, mu: NDArray, precision: NDArray, dvar: NDArray
) -> NDArray:
    alpha = precision @ (y - mu)
    return 0.5 * np.trace((np.multiply.outer(alpha, alpha) - precision) @ dvar)


def mvn_multi_log_unnormalized_pdf(y: NDArray, mus: NDArray, covs: NDArray) -> NDArray:
    """
    Find the unnormalized log density of y given a many different Gaussian distribution.
    For `mus` and `cov`, the first index indexes the distributionl

    This is mostly useful to find the MLE of kernel parameters.

    Parameters:
        y: the observed data
        mus: the means of the Gaussian distributions.
        covs: the covariances of Gaussian distributions.
    """
    covs_inv = np.linalg.inv(covs)
    # Throw the minus sign in front of the first y as to negate less.
    ret = np.einsum("i,lij,j->l", (-y - mus), covs_inv, (y - mus))
    # det should always be positive
    ret -= np.linalg.slogdet(covs)[1]
    # This constant wrt covs and mus so if don't care about it if we are maximzing
    N = len(y)
    ret -= N * np.log(2 * np.pi)
    return 0.5 * ret
    #return ret


