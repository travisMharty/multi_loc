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

    Returns
    -------
    corr : array_like
        Correlation of different positions whose distances were given by rho.
        Same shape as rho.
    """
    rho1 = np.sqrt(2.0OB * nu) * np.abs(rho) / rho0
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

    Returns
    -------
    corr : array_like
        Correlation of different positions whose distances were given by rho.
        Same shape as rho.
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

    Returns
    -------
    corr : array_like
        Correlation of different positions whose distances were given by rho.
        Same shape as rho.
    """
    if nu is not None:
        print("nu is unused")
    print(rho0)
    corr = np.exp(-(rho**2 / (2.0 * rho0**2)))
    return corr


def make_correlation_matrix(rho, rho0, correlation_fun, nu=None):
    """
    Return correlation matrix for distances rho, using the correlaiton_function
    and parameters rho0 and nu.

    Parameters
    ----------
    rho : array_like
        Distances of first position to all others.

    rho0 : Scalar
        Characteristic distance.

    correlation_fun : function
        Correlation function such that correlation_fun(rho, rho0, nu) returns the
        desired correlation for distances rho.

    nu : Scalar
        Matern smootheness parameter. Should be None is correlation is not
        correlation_matern.

    Returns
    -------
    Corr : array_like
        Correlation matrix based on the distances defined by rho and
        correlation_fun with parameters rho0 and nu. If n = rho.size, then
        (n, n) = rho.shape.
    """
    rho_size = rho.size
    cor_vec = correlation_fun(rho, rho0, nu)
    print(rho0)
    Corr = np.zeros([rho_size, rho_size])
    cor_vec = np.concatenate([cor_vec[:0:-1], cor_vec])
    for i in range(rho_size):
        Corr[i] = cor_vec[rho_size-1 - i:2 * rho_size - 1 - i]
    return Corr


def eig_decomp(C):
    """
    Returns eigenvalues and eigenvectors of C. Assumes that C is real symmetric
    and positive semi-definite. Eigenvalues will be clipped at zero and real.

    Parameters
    ----------
    C : array_like
        A  matrix which is real symmetric and positive semi-definite.

    Returns
    -------
    eig_values : array_like
        Vector containing eigenvalues of C in descending order.

    eig_vector : array_like
        Matrix containing the eigenvectors of C corresponding to eig_values.

    """
    eig_val, eig_vec = sp.linalg.eigh(C)
    # eig_val = eig_val.clip(min=0)
    eig_val = eig_val[::-1]
    eig_vec = eig_vec[:, ::-1]
    return eig_val, eig_vec

def matrix_sqrt(C=None, eig_val=None, eig_vec=None, return_eig=False):
    """
    Returns the symmetric matrix square root of C through eigendecomposition.
    Assumes that C is real symmetric and positive semi-definite. Eigenvalues
    will be clipped at zero and real.

    Parameters
    ----------
    C : array_like
        A correlation matrix which is real symmetric and positive semi-definite.
        Does not need to be provided if eig_val and eig_vec are provided.

    eig_val : array_like
        Eigenvalues of matrix to be square rooted. Should be in descending
        order, and non-negative real. Will be calculated if not provided.

    eig_vec : array_like
        Eigenvectors of matrix to be square rooted. Should correspond to
        eig_val. Will be calculated if not provided.

    return_eig : bool
        If True will return both the matrix square root as well as the
        eigendecomposition of C.

    Returns
    -------
    C_sqrt : array_like
        The symmetric square root of C where all eigenvalue square roots are
        taken as positive.

    eig_values : array_like
        Vector containing eigenvalues of C in descending order.

    eig_vector : array_like
        Matrix containing the eigenvectors of C corresponding to eig_values.
    """

    calc_eig = (eig_val is None) or (eig_vec is None)
    if calc_eig:
        eig_val, eig_vec = eig_decomp(C)

    C_sqrt = eig_vec @ np.diag(np.sqrt(eig_val + 0j)) @ eig_vec.T
    if return_eig:
        to_return  = (C_sqrt, eig_val, eig_vec)
    else:
        to_return = C_sqrt

    return to_return


def matrix_inv(C=None, eig_val=None, eig_vec=None, return_eig=False):
    """
    Returns the inverse or pseudo inverse of C through eigendecomposition.
    Assumes that C is real symmetric and positive semi-definite. Eigenvalues
    will be clipped at zero and real.

    Parameters
    ----------
    C : array_like
        A correlation matrix which is real symmetric and positive semi-definite.
        Does not need to be provided if eig_val and eig_vec are provided.

    eig_val : array_like
        Eigenvalues of matrix to be inverted. Should be in descending order, and
        non-negative real. Will be calculated if not provided.

    eig_vec : array_like
        Eigenvectors of matrix to be inverted. Should correspond to eig_val.
        Will be calculated if not provided.

    return_eig : bool
        If True will return both the matrix inverse as well as the
        eigendecomposition of C.

    Returns
    -------
    C_inv : array_like
        The inverse of matrix C.

    eig_values : array_like
        Vector containing eigenvalues of C in descending order.

    eig_vector : array_like
        Matrix containing the eigenvectors of C corresponding to eig_values.
    """

    calc_eig = (eig_val is None) or (eig_vec is None)
    if calc_eig:
        eig_val, eig_vec = eig_decomp(C)

    eig_val_inv = eig_val.copy()
    eig_val_inv[eig_val != 0] = 1/eig_val[eig_val != 0]

    C_inv = eig_vec @ np.diag(eig_val_inv) @ eig_vec.T

    if return_eig:
        to_return  = (C_inv, eig_val, eig_vec)
    else:
        to_return = C_inv

    return to_return


def matrix_sqrt_inv(C=None, eig_val=None, eig_vec=None, return_eig=False):
    """
    Returns the inverse or pseudo inverse of the square root of C through
    eigendecomposition. Assumes that C is real symmetric and positive
    semi-definite. Eigenvalues will be clipped at zero and real.

    Parameters
    ----------
    C : array_like
        A correlation matrix which is real symmetric and positive semi-definite.
        Does not need to be provided if eig_val and eig_vec are provided.

    eig_val : array_like
        Eigenvalues of matrix to be square rooted and inverted. Should be in
        descending order, and non-negative real. Will be calculated if not
        provided.

    eig_vec : array_like
        Eigenvectors of matrix to be square rooted and inverted. Should
        correspond to eig_val. Will be calculated if not provided.

    return_eig : bool
        If True will return both the matrix square root as well as the
        eigendecomposition of C.

    Returns
    -------
    C_sqrt_inv : array_like
        The inverse of the square root of the matrix C.

    eig_values : array_like
        Vector containing eigenvalues of C in descending order.

    eig_vector : array_like
        Matrix containing the eigenvectors of C corresponding to eig_values.
    """

    calc_eig = (eig_val is None) or (eig_vec is None)
    if calc_eig:
        eig_val, eig_vec = eig_decomp(C)
    eig_val_sqrt = np.sqrt(eig_val + 0j)

    C_sqrt_inv = matrix_inv(eig_val=eig_val_sqrt, eig_vec=eig_vec)

    if return_eig:
        to_return  = (C_sqrt_inv, eig_val, eig_vec)
    else:
        to_return = C_sqrt_inv

    return to_return


def generate_ensemble(mu, P_sqrt, ens_size):
    """
    Return ensemble drawn from N(mu, P).

    Parameters
    ----------
    mu : array_like
        Mean of the returned ensemble samples. P.shape must equal
        (mu.size, mu.size).

    P_sqrt : array_like
        Square root of the desired covariance of the ensemble samples.

    ens_size : scalar
        Number of ensemble members to return.

    Returns
    ------
    ens : array_like
        Ensemble of samples drawn from N(mu, P) of size m x n where m is the
        dimension of the state space and m is ens_size.
    """
    dimension = mu.size
    mu = mu[:, None]
    ens = mu + P_sqrt @ np.random.randn(dimension, ens_size)

    return ens
