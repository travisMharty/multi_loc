import numpy as np
import scipy as sp
import scipy.special


def correlation_matern(rho, rho0, nu=2.5):
    rho1 = np.sqrt(2 * nu) * np.abs(rho) / rho0
    zero_indices = np.where(rho1 == 0)[0]
    corr = (rho1 ** nu
            * sp.special.kv(nu, rho1)
            / sp.special.gamma(nu)
            / 2.0 ** (nu - 1))
    corr[zero_indices] = 1.0
    return corr


def correlation_exp(rho, rho0, nu=None):
    if nu is not None:
        print("nu is unused")
    corr = np.exp(-np.abs(rho/rho0))
    return corr


def correlation_sqd_exp(rho, rho0, nu=None):
    if nu is not None:
        print("nu is unused")
    corr = np.exp(-(rho**2 / (2 * rho0**2)))
    return corr


def make_correlation_matrix(rho, rho0, correlation, nu=None):
    rho_size = rho.size
    cor_vec = correlation(rho, rho0, nu)
    corr = np.zeros([rho_size, rho_size])
    cor_vec = np.concatenate([cor_vec[:0:-1], cor_vec])
    for i in range(rho_size):
        corr[i] = cor_vec[rho_size-1 - i:2 * rho_size - 1 - i]
    return corr
