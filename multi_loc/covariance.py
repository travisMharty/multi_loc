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
    Returns the symmetric matrix square root of C through eigen decomposition.
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
        eigen decomposition of C.

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
    Returns the inverse or pseudo inverse of C through eigen decomposition.
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
        eigen decomposition of C.

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
    eigen decomposition. Assumes that C is real symmetric and positive
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
        eigen decomposition of C.

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


def return_waves(Nx, dx):
    """
    Returns wavenumbers corresponding to np.fft.fft of a function on a line of
    length Nx and spacing dx.

    Parameters
    ----------
    Nx : scalar
        Length of wavenumber vector.

    dx : scalar
        Step size of grid in physical space.

    Returns
    -------
    k : ndarray
        The wave numbers which correspond to vector of length Nx and grid
        size dx.
    """
    domain_length = dx*Nx
    if Nx % 2 == 0:
        k = np.arange(-Nx/2, Nx/2)
    else:
        k = np.arange(-(Nx - 1)/2, (Nx - 1)/2 + 1)
    k = np.fft.ifftshift(k)/domain_length
    return k


def fft_exp_1d(Nx, dx, rho0, nu=None):
    """
    Returns the fft of a exponential correlation function on a 1D periodic
    domain of size Nx which has grid size dx and characteristic distance
    rho0.

    Parameters
    ----------
    Nx : scalar
        The size of the grid over which the calculation will be made.

    dx : scalar
        The grid spacing in physical space.

    rho0 : scalar
        The characteristic length scale in physical space.

    Returns
    -------
    eig_val : ndarray
        The fft of the 1D exponential correlation function. These are also the
        eigenvalues corresponding to the correlation matrix.
    """
    k = return_waves(Nx, dx)
    a = rho0
    eig_val = (
        (2 * a) / (1 + (2 * np.pi * k * a)**2))
    return eig_val


def fft_sqd_exp_1d(Nx, dx, rho0, nu=None):
    """
    Returns the fft of a squared exponential correlation function on a 1D
    periodic domain of size Nx which has grid size dx and characteristic
    distance rho0.

    Parameters
    ----------
    Nx : scalar
        The size of the grid over which the calculation will be made.

    dx : scalar
        The grid spacing in physical space.

    rho0 : scalar
        The characteristic length scale in physical space.

    Returns
    -------
    eig_val : ndarray
        The fft of the 1D squared exponential correlation function. These are
        also the eigenvalues corresponding to the correlation matrix.
    """
    k = return_waves(Nx, dx)
    a = 2 * rho0**2
    eig_val = (np.sqrt(np.pi * a)
               * np.exp( - np.pi**2 * a * k**2))
    return eig_val


def generate_circulant(Nx, dx, rho0, correlation_fun, nu=None, return_eig=True,
                       return_Corr=False, eig_tol=None):
    """
    Return correlation matrix for distances rho, using the correlaiton_function
    and parameters rho0 and nu. Assumes that rho is such that the produced
    correlation matrix will be a circulant. adf

    Parameters
    ----------
    Nx : scalar
        The size of the grid covering the physical space.

    dx : scalar
        The grid spacing in physical space.

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
    eig_val = correlation_fun(Nx, dx, rho0, nu)
    sort_index = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sort_index]
    # Need to set this up in the future to only calculate the needed
    # eigenvectors based on some tolerance.
    eig_vec = np.fft.fft(np.eye(Nx))/np.sqrt(Nx)
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


def generate_circulant_old(rho, rho0, correlation_fun, nu=None, return_eig=True,
                       return_Corr=False, eig_tol=None):
    """
    Return correlation matrix for distances rho, using the correlaiton_function
    and parameters rho0 and nu. Assumes that rho is such that the produced
    correlation matrix will be a circulant. adf

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
    sort_index = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sort_index]
    # Need to set this up in the future to only calculate the needed
    # eigenvectors based on some tolerance.

    eig_vec = np.fft.fft(np.eye(rho_size))/np.sqrt(rho_size)
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


def approx(eig_val, tol):
    """
    Returns the number of eigenvalues needed to meet some tolerance.

    Parameters
    ----------
    eig_val : array_like
        Eigenvalues  to be analyzed. Should be non-negative real and in
        descending order.

    tol : scalar
        The tolerance which must be reached in terms of the squared sum of the
        eigenvalues.

    Returns
    -------
    eig_num : scalar
        The number of eigenvalues which should be used to meet the tolerance.
    """
    eig_num = 1
    eig_tol = (1 - tol) * (eig_val**2).sum()
    eig_sum = 0
    cond = True
    while cond:
        eig_sum += eig_val[eig_num-1]**2
        # print(eig_tol)
        if eig_sum > eig_tol or eig_num == eig_val.size:
            cond = False
        else:
            eig_num += 1
    return eig_num


def get_approx_eig(Corr, tol):
    """
    Return eig_val and eig_vec which approximates Corr.

    Parameters
    ----------
    Corr : array_like
        The correlation matrix to be approximated.

    tol : scalar
        The tolerance which will be met in terms of the squared sum of the
        eigenvalues of Corr.

    Returns
    -------
    eig_val : array_like
        Eigenvalues that approximate Corr.

    eig_vec : array_like
        Eigenvectors that approximate Corr.
    """
    eig_val, eig_vec = np.linalg.eigh(Corr)
    eig_val = eig_val[::-1]
    eig_vec = eig_vec[:, ::-1]
    eig_num = approx(eig_val, tol)
    eig_val = eig_val[:eig_num]
    eig_vec = eig_vec[:, :eig_num]
    return eig_val, eig_vec


def eig_2d_covariance(rho_x, rho_y, rho0_x, rho0_y, tol,
                      correlation_fun, nu=None):
    """
    Return correlation matrix for distances rho_x and rho_y, using the
    correlaiton_function and parameters rho0_x, rho0_y and nu.

    Parameters
    ----------
    rho_x : array_like
        Distances of first position to all others for x.

    rho_y : array_like
        Distances of first position to all others for y.

    rho0_x : Scalar
        Characteristic distance for x.

    rho0_y : Scalar
        Characteristic distance for y.

    tol : Scalar
        Tolerance which will be used to reduce the number of eigenvalues Corr_x,
        Corr_y, and Corr in terms of the squared sum of the eigenvalues.

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
    Corr_x = make_correlation_matrix(
        rho_x, rho0_x, correlation_fun, nu=nu)
    Corr_y = make_correlation_matrix(
        rho_y, rho0_y, correlation_fun, nu=nu)

    eig_val_x, eig_vec_x = get_approx_eig(Corr_x, tol)
    eig_val_y, eig_vec_y = get_approx_eig(Corr_y, tol)
    eig_val = np.kron(eig_val_y, eig_val_x)
    eig_vec = np.kron(eig_vec_y, eig_vec_x)

    sorted_indices = np.argsort(eig_val)
    sorted_indices = sorted_indices[::-1]
    eig_val = eig_val[sorted_indices]
    eig_vec = eig_vec[:, sorted_indices]
    eig_num = approx(eig_val, tol)
    eig_val = eig_val[:eig_num]
    eig_vec = eig_vec[:, :eig_num]
    return eig_val, eig_vec
