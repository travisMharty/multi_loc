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
    rho1 = np.sqrt(2.0 * nu) * np.abs(rho) / rho0
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
        Correlation function such that correlation_fun(rho, rho0, nu) returns
        the desired correlation for distances rho.

    nu : Scalar
        Matern smootheness parameter. Should be None if correlation is not
        correlation_matern.

    Returns
    -------
    Corr : array_like
        Correlation matrix based on the distances defined by rho and
        correlation_fun with parameters rho0 and nu. If n = rho.size, then
        (n, n) = rho.shape.
    """
    rho_size = rho.size
    corr_vec = correlation_fun(rho, rho0, nu)
    Corr = np.zeros([rho_size, rho_size])
    corr_vec = np.concatenate([corr_vec[:0:-1], corr_vec])
    for i in range(rho_size):
        Corr[i] = corr_vec[rho_size-1 - i:2 * rho_size - 1 - i]
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
    eig_val = eig_val[::-1]
    eig_vec = eig_vec[:, ::-1]
    return eig_val, eig_vec

def matrix_sqrt(C=None, eig_val=None, eig_vec=None, return_eig=False):
    """
    Returns the symmetric matrix square root of C through eigendecomposition.
    Assumes that C is real symmetric and positive semi-definite.

    Parameters
    ----------
    C : array_like
        A correlation matrix which is real symmetric and positive semi-definite.
        Does not need to be provided if eig_val and eig_vec are provided.

    eig_val : array_like
        Eigenvalues of matrix to be square rooted. Should be in descending
        order. Will be calculated if not provided.

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

    C_sqrt = eig_vec @ np.diag(np.sqrt(eig_val + 0j)) @ eig_vec.conj().T
    if return_eig:
        to_return  = (C_sqrt, eig_val, eig_vec)
    else:
        to_return = C_sqrt

    return to_return


def matrix_inv(C=None, eig_val=None, eig_vec=None, return_eig=False):
    """
    Returns the inverse or pseudo inverse of C through eigendecomposition.
    Assumes that C is real symmetric and positive semi-definite.

    Parameters
    ----------
    C : array_like
        A correlation matrix which is real symmetric and positive semi-definite.
        Does not need to be provided if eig_val and eig_vec are provided.

    eig_val : array_like
        Eigenvalues of matrix to be inverted. Should be in descending order.
        Will be calculated if not provided.

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

    C_inv = eig_vec @ np.diag(eig_val_inv) @ eig_vec.conj().T

    if return_eig:
        to_return  = (C_inv, eig_val, eig_vec)
    else:
        to_return = C_inv

    return to_return


def matrix_sqrt_inv(C=None, eig_val=None, eig_vec=None, return_eig=False):
    """
    Returns the inverse or pseudo inverse of the square root of C through
    eigendecomposition. Assumes that C is real symmetric and positive
    semi-definite.

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


def generate_ensemble(ens_size, mu, P_sqrt=None, eig_val=None, eig_vec=None):
    """
    Return ensemble drawn from N(mu, P).

    Parameters
    ----------
    ens_size : scalar
        Number of ensemble members to return.

    mu : array_like
        Mean of the returned ensemble samples. P.shape must equal
        (mu.size, mu.size).

    P_sqrt : array_like
        Square root of the desired covariance of the ensemble samples.

    eig_val : array_like
        Eigenvalues corresponding to the correlation matrix P.
        If eig_val and eig_vec are provided, P_sqrt is not used.

    eig_vec : array_like
        Eigenvectors corresponding to the correlation matrix P.
        If eig_val and eig_vec are provided, P_sqrt is not used.

    Returns
    ------
    ens : array_like
        Ensemble of samples drawn from N(mu, P) of size m x n where m is the
        dimension of the state space and m is ens_size.
    """
    dimension = mu.size
    mu = mu[:, None]
    eig_condition = ((eig_val is not None )
                     and (eig_vec is not None))
    if eig_condition:
        eig_dim = eig_val.size
        ens = (mu + eig_vec
               @ (np.sqrt(eig_val[:, None])
                  * np.random.randn(eig_dim, ens_size)))

        # ens = (mu + (eig_vec @ np.diag(np.sqrt(eig_val)) @ eig_vec.conj().T)
        #        @ np.random.randn(eig_dim, ens_size))
    else:
        ens = mu + P_sqrt @ np.random.randn(dimension, ens_size)

    return ens


def generate_circulant(rho, rho0, correlation_fun, nu=None, return_eig=True,
                       return_Corr=False):
    """
    Return correlation matrix for distances rho, using the correlaiton_function
    and parameters rho0 and nu. Assumes that rho is such that the produced
    correlation matrix will be a circulant.

    Parameters
    ----------
    rho : array_like
        Distances of first position to all others.

    rho0 : Scalar
        Characteristic distance.

    correlation_fun : function
        Correlation function such that correlation_fun(rho, rho0, nu) returns
        the desired correlation for distances rho.

    nu : Scalar
        Matern smootheness parameter. Should be None if correlation is not
        correlation_matern.

    return_eig : bool
        If True then will return eig_val and eig_vec.

    return_Corr : bool
        If True the will return Coor matrix.

    Returns
    -------
    eig_val : array_like
        Eigenvalues of the circulant correlation matrix corresponding to
        correlation_fun(rho, rho0, nu). Only returns if retrun_eig is True.

    eig_vec : array_like
        Eigenvectors of the circulant correlation matrix corresponding to
        correlation_fun(rho, rho0, nu). Only returns if return_eig is True.

    Corr : array_like
        Circulant correlation matrix corresponding to eig_val and eig_vec.
        Only returns if return_Corr is True.
    """
    rho_size = rho.size
    corr_vec = correlation_fun(rho, rho0, nu)
    eig_val = np.fft.fft(corr_vec)
    eig_vec = np.fft.fft(np.eye(rho_size))/np.sqrt(rho_size)
    sort_index = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sort_index]
    eig_vec = eig_vec[:, sort_index]

    if return_Corr:
        Corr = eig_vec @ np.diag(eig_val) @ eig_vec.conj().T
    to_return = None
    if return_Corr and return_eig:
        to_return = (eig_val, eig_vec, Corr)
    elif return_Corr:
        to_return = Corr
    elif return_eig:
        to_return = (eig_val, eig_vec)

    return to_return
