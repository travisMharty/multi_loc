import numpy as np
import scipy as sp
import scipy.special


def correlation_matern(rho, rho0, nu=2.5):
    """
    Return correlation as evaluated by the Matern correlation function.
    https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function

    If nu = 0.5 then the Matern correlation function is equivalent to the
    exponential covariance function.

    As nu approaches infinity the Matern function approaches the squared
    exponential covariance function.

    Parameters
    ----------
    rho : array_like
        Distances of first position to all others.

    rho0 : Scalar
        Characteristic distance.

    nu : Scalar
        Smoothness parameter. Must be positive real.
    """
    rho1 = np.sqrt(2 * nu) * np.abs(rho) / rho0
    zero_indices = np.where(rho1 == 0)[0]
    corr = (rho1 ** nu
            * sp.special.kv(nu, rho1)
            / sp.special.gamma(nu)
            / 2.0 ** (nu - 1))
    corr[zero_indices] = 1.0
    return corr


def correlation_exp(rho, rho0, nu=None):
    """
    Return correlation as evaluated by the exponential covariance function.

    Parameters
    ----------
    rho : array_like
        Distances of first position to all others.

    rho0 : Scalar
        Characteristic distance.

    nu : Scalar
        Unused dummy variable
    """
    if nu is not None:
        print("nu is unused")
    corr = np.exp(-np.abs(rho/rho0))
    return corr


def correlation_sqd_exp(rho, rho0, nu=None):
    """
    Return correlation as evaluated by the squared exponential covariance
    function.

    Parameters
    ----------
    rho : array_like
        Distances of first position to all others.

    rho0 : Scalar
        Characteristic distance.

    nu : Scalar
        Unused dummy variable
    """
    if nu is not None:
        print("nu is unused")
    corr = np.exp(-(rho**2 / (2 * rho0**2)))
    return corr


def make_correlation_matrix(rho, rho0, correlation, nu=None):
    """
    Return correlation as evaluated by the squared exponential covariance
    function.

    Parameters
    ----------
    rho : array_like
        Distances of first position to all others.

    rho0 : Scalar
        Characteristic distance.

    correlation : function
        Correlation function such that correlation(rho, rho0, nu) returns the
        desired correlation for distances rho.

    nu : Scalar
        Matern smootheness parameter. Should be None is correlation is not
        correlation_matern.
    """
    rho_size = rho.size
    cor_vec = correlation(rho, rho0, nu)
    corr = np.zeros([rho_size, rho_size])
    cor_vec = np.concatenate([cor_vec[:0:-1], cor_vec])
    for i in range(rho_size):
        corr[i] = cor_vec[rho_size-1 - i:2 * rho_size - 1 - i]
    return corr
