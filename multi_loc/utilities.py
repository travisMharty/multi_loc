import numpy as np
import scipy as sp
from scipy import integrate

def angle(V0, V1):
    """
    Calculate the maximum angle between two vector spaces spanned by the basis
    vectors contained in V0 and V1. This is based on the smallest singular value
    of the matrix of inner products between V0 and V1.
    wiki: https://en.wikipedia.org/wiki/Angles_between_flats

    Parameters
    ----------
    V0 : ndarray
        Each column of V0 is a basis vector for the first space.

    V1 : ndarray
        Each column of V1 is a basis vector for the second space.

    Returns
    -------
    angle : float
        This is the largest angle between the vector spaces spanned by V0 and V1
    """
    IPs = V0.T @ V1
    norm0 = (V0 ** 2).sum(axis=0)
    norm1 = (V1 ** 2).sum(axis=0)
    norm0 = norm0[:, None]
    norm1 = norm1[None, :]
    sigmas = sp.linalg.svd(
        IPs / np.sqrt(norm0 * norm1),
        compute_uv=False)
    angle = np.arccos(sigmas.min())
    return angle


def resample_bootstrap(ensemble, resample_size=None):
    """
    Resamples an ensemble with replacement.

    Parameters
    ----------
    ensemble : ndarray
        An ensemble of samples such that each column is a sample.

    resample_size : ndarray
        The number of samples the resampled ensemble will have. If no size is
        given then the original number of samples will be used.

    Returns
    -------
    resampled_ensemble : ndarray
        The resampled ensemble. Made up of samples taken from the original
        ensemble.
    """
    if resample_size is None:
        resample_size = ensemble.shape[1]
    max_int = ensemble.shape[1]
    random_indexes = np.random.randint(0, max_int, resample_size)

    resampled_ensemble = ensemble[:, random_indexes]
    return resampled_ensemble


def return_P_sample_array(ensemble, ens_ens_size,
                          resample_type, resample_size=None):
    dimension = ensemble.shape[0]
    P_sample_array = np.ones([dimension, dimension, ens_ens_size])
    if resample_type == 'bootstrap':
        if resample_size is None:
            resample_size = ensemble.shape[1]
        for ens_ens_num in range(ens_ens_size):
            temp_ens = resample_bootstrap(
                ensemble=ensemble,
                resample_size=resample_size)
            P_sample_array[:, :, ens_ens_num] = np.cov(temp_ens)
    elif resample_type == 'bayes_bootstrap':
        if resample_size is not None:
            print('No resample size for bayes!')
        ens_size = ensemble.shape[1]
        weights = bayes_bootstrap_weights(ens_size, ens_ens_size)
        for ens_ens_num in range(ens_ens_size):
            P_sample_array[:, :, ens_ens_num] = np.cov(
                ensemble, ddof=1,
                aweights=weights[ens_ens_num])
    return P_sample_array


def bayes_bootstrap_weights(ens_size, ens_ens_size):
    weights = np.random.uniform(
        low=0, high=1,
        size=(ens_size - 1) * ens_ens_size)
    weights = weights.reshape(ens_ens_size, ens_size - 1)
    weights.sort()
    weights = np.concatenate([np.zeros([ens_ens_size, 1]),
                              weights,
                              np.ones([ens_ens_size, 1])],
                             axis=1)
    weights = weights[:, 1:] - weights[:, :-1]
    return weights


def lorenz_96(x, F=8):
    """
    Lorenz 96 state derivative ie dx/dt = lorenz_96(x).
    """
    xp1 = np.roll(x, -1, axis=0)
    xm2 = np.roll(x, 2, axis=0)
    xm1 = np.roll(x, 1, axis=0)
    return (xp1 - xm2) * xm1 - x + F


def return_lorenz_96_data(x0, F, t):
    def this_L96(x, t):
        return lorenz_96(x, F=F)
    x = integrate.odeint(this_L96, x0, t)
    return x


def L96_deriv(x):
    n = x.size
    dJdt = -1 * np.eye(n)
    for i in range(n):
        dJdt[i, i - 2] = -x[i - 1]
        dJdt[i, i - 1] = x[np.mod(i + 1, n)] - x[i - 2]
        dJdt[i, np.mod(i + 1, n)] = x[i - 1]
    return dJdt


def L96_and_deriv(x_J, n, F=8):
    x = x_J[:n]
    dJdt = L96_deriv(x)
    dxdt  = lorenz_96(x, F=F)
    dxdt_dJdt = np.concatenate([dxdt, dJdt.ravel()])
    return dxdt_dJdt


def return_lorenz_96_TL(x0, dt, F=8):
    N_state = x0.size
    J0 = np.zeros(N_state*N_state)
    x0_J0 = np.concatenate([x0, J0])

    def this_L96_and_deriv(x_J, t):
        return L96_and_deriv(x_J, N_state, F=F)
    x_dM = integrate.odeint(this_L96_and_deriv, x0_J0, [0, dt])

    dM = x_dM[1, N_state:].reshape(N_state, N_state)
    return dM


def L96_multi(X, Y, F=8, c=10, b=10, h=1):
    """
    X is the coarse variable, Y is the fine variable.
    The size of Y should be a multiple of the size of X.
    """
    N_X = X.size
    N_Y = Y.size
    N_YpX = N_Y//N_X

    if N_X * N_YpX != N_Y:
        raise Exception('Size of X should evenly divide size of Y.')

    Y_sum = Y.reshape(N_X, N_YpX).sum(axis=1)

    Xp1 = np.roll(X, -1, axis=0)
    Xm2 = np.roll(X, 2, axis=0)
    Xm1 = np.roll(X, 1, axis=0)
    dXdt = (Xp1 - Xm2) * Xm1 - X + F - (h * c) / b * Y_sum

    X_forcing =  np.repeat(X, N_YpX)
    Ym1 = np.roll(Y, 1, axis=0)
    Yp2 = np.roll(Y, -2, axis=0)
    Yp1 = np.roll(Y, -1, axis=0)
    dYdt = (c * b)*(Ym1 - Yp2) * Yp1 - c*Y + (h*c)/b*X_forcing
    return dXdt, dYdt


def return_L96_multi_data(X0, Y0, t, F=8, b=10, c=10, h=1):
    """
    Returns time series starting at X0, Y0, with times t.
    The size of X0 should evenly divide the size of Y0.
    """
    N_X = X0.size
    N_Y = Y0.size
    N_YpX = N_Y//N_X
    if N_X*N_YpX != N_Y:
        raise Exception('Size of X0 should evenly divide size of Y0.')
    def this_L96_multi(XY_1D, t):
        X = XY_1D[:N_X]
        Y = XY_1D[N_X:]
        dXdt, dYdt = L96_multi(X, Y, F=F, b=b, c=c, h=h)
        dXYdt = np.concatenate([dXdt, dYdt])
        return dXYdt
    XY0 = np.concatenate([X0, Y0])
    XY = integrate.odeint(this_L96_multi, XY0, t)
    XY = XY.T
    X = XY[:N_X]
    Y = XY[N_X:]
    return X, Y


def generate_XY_X_ens(X_ts, Y_ts, N_eXY, N_eX):
    """
    Subsamples X and Y to generate an ensemble of both X and Y of size N_eXY
    and an ensemble of X of size N_eX.
    """
    N_t = X_ts.shape[1]
    fine_indices = np.random.choice(N_t, size=N_eXY, replace=False)
    Y_ens = Y_ts[:, fine_indices]
    XfY_ens = X_ts[:, fine_indices]
    XY_ens = np.concatenate([XfY_ens, Y_ens], axis=0)

    coarse_indices = np.random.choice(N_t, size=N_eX, replace=False)
    X_ens = X_ts[:, coarse_indices]
    return XY_ens, X_ens


def L96_multi_ensemble(XY_ens, X_ens, F=8, c=10, b=10, h=1):
    """
    Function to return the derivative of XY and X as described by Lorenz 96
    model for an ensemble of XY and of X alone. Y forcing for X alone ensemble
    comes from the XY ensemble.
    """
    N_XY, N_eXY = XY_ens.shape
    N_X, N_eX = X_ens.shape
    N_Y = N_XY - N_X
    N_YpX = N_Y // N_X
    if N_X * N_YpX != N_Y:
        raise Exception('Size of X should evenly divide size of Y.')
    XfY_ens = XY_ens[:N_X]
    Y_ens = XY_ens[N_X:]

    Y_sum = Y_ens.reshape(N_X, N_YpX, N_eXY).sum(axis=1)

    Xp1 = np.roll(XfY_ens, -1, axis=0)
    Xm2 = np.roll(XfY_ens, 2, axis=0)
    Xm1 = np.roll(XfY_ens, 1, axis=0)
    dXfYdt = (Xp1 - Xm2) * Xm1 - XfY_ens + F - (h * c) / b * Y_sum

    X_forcing =  np.repeat(XfY_ens, N_YpX, axis=0)
    Ym1 = np.roll(Y_ens, 1, axis=0)
    Yp2 = np.roll(Y_ens, -2, axis=0)
    Yp1 = np.roll(Y_ens, -1, axis=0)
    dYdt = (c * b) * (Ym1 - Yp2) * Yp1 - c*Y_ens + (h*c)/b*X_forcing

    dXYdt = np.concatenate([dXfYdt, dYdt],
                           axis=0)

    repeats = N_eX / N_eXY
    if repeats != int(repeats):
        raise Exception('Ensemble size of X should evenly divide ensemble size of Y.')
    repeats = int(repeats)
    Y_sum = np.repeat(Y_sum, repeats, axis=1)

    Xp1 = np.roll(X_ens, -1, axis=0)
    Xm2 = np.roll(X_ens, 2, axis=0)
    Xm1 = np.roll(X_ens, 1, axis=0)
    dXdt = (Xp1 - Xm2) * Xm1 - X_ens + F - (h * c) / b * Y_sum

    return dXYdt, dXdt


def return_L96_multi_ens_data(XY0_ens, X0_ens, t, F=8, b=10, c=10, h=1):
    """
    Returns XY and X ensemble using L96_multi_ensemble integrated over times t.
    """
    N_XY, N_eXY = XY0_ens.shape
    N_X, N_eX = X0_ens.shape
    N_Y = N_XY - N_X
    N_YpX = N_Y // N_X
    if N_X * N_YpX != N_Y:
        raise Exception('Size of X should evenly divide size of Y.')
    XY_ens_size = N_XY * N_eXY
    N_t = t.size
    def this_L96_multi_ens(XYX_raveled, t):
        XY_ens = XYX_raveled[:XY_ens_size].reshape(N_XY, N_eXY)
        X_ens = XYX_raveled[XY_ens_size:].reshape(N_X, N_eX)
        dXYdt, dXdt = L96_multi_ensemble(
            XY_ens, X_ens, F=F, b=b, c=c, h=h)
        to_return = np.concatenate(
            [dXYdt.ravel(), dXdt.ravel()])
        return to_return
    XY0_raveled = np.concatenate(
        [XY0_ens.ravel(), X0_ens.ravel()])
    XYX_raveled = integrate.odeint(
        this_L96_multi_ens, XY0_raveled, t)
    XYX_raveled = XYX_raveled.T
    XY_ens_ts = XYX_raveled[:XY_ens_size].reshape(N_XY, N_eXY, N_t)
    X_ens_ts = XYX_raveled[XY_ens_size:].reshape(N_X, N_eX, N_t)
    return XY_ens_ts, X_ens_ts
