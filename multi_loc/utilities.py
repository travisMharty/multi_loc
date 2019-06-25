import numpy as np
import scipy as sp
from scipy import integrate, interpolate, ndimage

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


def LM3(Z, K=32, I=12, F=15, b=10, c=2.5, alpha=None, beta=None):
    if alpha is None:
        alpha = (3 * I**2 + 3) / (2 * I**3 + 4 * I)
    if beta is None:
        beta = (2 * I**2 + 1) / (I**4 + 2 * I**2)
    X = window_sum_Z(Z, I=I, alpha=alpha, beta=beta)
    Y = Z - X

    T1 = bracket([X], K)
    T2 = b**2 * bracket([Y], 1)
    T3 = c * bracket([Y, X], 1)

    dZdt = T1 + T2 + T3 - X - b*Y + F
    return dZdt


def Z_sum_weights(I, alpha, beta):
    weights = np.abs(np.arange(-I, I + 1, dtype=float))
    weights = alpha - beta * weights
    weights[0] *= 1/2
    weights[-1] *= 1/2
    return weights


def window_sum_Z(Z,*, I, alpha, beta):
    weights = Z_sum_weights(I, alpha, beta)
    X = ndimage.convolve1d(Z, weights, mode='wrap', axis=0)
    return X


def window_sum_Z_matrix(N_Z, *, I, alpha, beta):
    weights = Z_sum_weights(I, alpha, beta)
    row1 = np.zeros(N_Z)
    row1[:weights.size] = weights
    row1 = np.roll(row1, -I)
    matrix = sp.linalg.circulant(row1).T
    return matrix


def bracket(XY, K):
    if len(XY) == 1:
        to_return = bracket_1(XY, K)
    else:
        to_return = bracket_2(XY, K)
    return to_return


def bracket_2(XY, K):
    """
    Only works for K=1 for now.
    """
    if K == 1:
        X, Y = XY
        T1 = -1 * np.roll(X, 2, axis=0) * np.roll(Y, 1, axis=0)
        T2 = np.roll(X, 1, axis=0) * np.roll(Y, -1, axis=0)
        return T1 + T2
    else:
        raise Exception('Currently only works for K=1')


def bracket_1(XY, K):
    X = XY[0]
    if K == 1:
        T1 = -1 * np.roll(X, 2, axis=0) * np.roll(X, 1, axis=0)
        T2 = np.roll(X, 1, axis=0) * np.roll(X, -1, axis=0)
        return T1 + T2
    elif K % 2 == 0:
        J = int(K/2)
        weights = np.ones(2 * J + 1)
        weights[0] *= 1/2
        weights[-1] *= 1/2
        weights /= K
    elif K % 2 == 1:
        J = int((K - 1)/2)
        weights = np.ones(2 * J + 1)
        weights /= K
    W = ndimage.convolve1d(X, weights, mode='wrap', axis=0)
    T1 = -1 * np.roll(W, 2 * K, axis=0) * np.roll(W, K, axis=0)
    WX = np.roll(W, K, axis=0) * np.roll(X, -K, axis=0)
    T2 = ndimage.convolve1d(WX, weights, mode='wrap', axis=0)
    return T1 + T2


def return_LM3_data_sp(Z0, t, K=32, I=12, F=15, b=10, c=2.5, alpha=None, beta=None):
    if alpha is None:
        alpha = (3 * I**2 + 3) / (2 * I**3 + 4 * I)
    if beta is None:
        beta = (2 * I**2 + 1) / (I**4 + 2 * I**2)

    def this_LM3(Z, t):
        dZdt = LM3(Z, K=K, I=I, F=F, b=b, c=c, alpha=alpha, beta=beta)
        return dZdt

    Z = integrate.odeint(this_LM3, Z0, t)
    Z = Z.T
    return Z


def generate_X_Z_LM3_ens(Z_ts, N_eX, N_eZ, coarse, I, alpha, beta):
    N_t = Z_ts.shape[1]
    fine_indices = np.random.choice(N_t, size=N_eZ, replace=False)
    Z_ens = Z_ts[:, fine_indices]
    fine_indices = np.random.choice(N_t, size=N_eX, replace=False)
    X_ens = Z_ts[:, fine_indices]
    X_ens = window_sum_Z(X_ens, I=I, alpha=alpha, beta=beta)
    X_ens = X_ens[::coarse]
    return X_ens, Z_ens


def upscale_on_loop(Zc, coarse):
    N_Z = Zc.shape[1] * coarse
    x = np.arange(N_Z)
    xc = x[::coarse]
    xc = np.concatenate([xc, [xc[-1] + coarse]])
    Zc = np.concatenate([Zc, Zc[:, 0][:, None]], axis=1)
    f = interpolate.interp1d(xc, Zc, axis=-1, kind='quadratic')
    Z = f(x)/np.sqrt(coarse)
    return Z


def interp_on_loop(Z, x, x_interp, x_max=None, kind='quadratic'):
    if x_max is None:
        x_max = x_interp.max()
    Z = np.concatenate([Z, [Z[0]]])
    x = np.concatenate([x, [x_max + 1 + x[0]]])
    if x[0] != x_interp[0]:
        Z = np.concatenate([[Z[-1]], Z])
        x = np.concatenate([[0], x])
    f = interpolate.interp1d(x, Z, kind=kind, axis=0)
    Z_interp = f(x_interp)
    return Z_interp

def lin_interp_matrix(N, coarse):
    N_c = N // coarse
    col1 = 1 - np.abs(np.linspace(-1 + 1/coarse, 1, coarse*2))
    N_zeros = (N_c - 2) * coarse
    if N_zeros != 0:
        col1 = np.concatenate([col1, np.zeros(N_zeros)])
    col1 = np.roll(col1, 1 - coarse)
    M = np.zeros([col1.size, N_c])
    for jj in range(N_c):
        M[:, jj] = np.roll(col1, coarse * jj)
    return M


def return_LM3_ens_data_sp(Z0_ens, t, K=32, I=12, F=15,
                        b=10, c=2.5, alpha=None, beta=None):
    if alpha is None:
        alpha = (3 * I**2 + 3) / (2 * I**3 + 4 * I)
    if beta is None:
        beta = (2 * I**2 + 1) / (I**4 + 2 * I**2)
    N_Z, N_eZ = Z0_ens.shape
    N_t = t.size

    def this_LM3(Z_1d_ens, t):
        Z_ens = Z_1d_ens.reshape(N_Z, N_eZ)
        dZdt = LM3(Z_ens, K=K, I=I, F=F, b=b, c=c, alpha=alpha, beta=beta)
        return dZdt.ravel()

    Z_ens_1d = integrate.odeint(this_LM3, Z0_ens.ravel(), t)
    Z_ens_1d = Z_ens_1d.T
    Z_ens_ts = Z_ens_1d.reshape(N_Z, N_eZ, N_t)
    return Z_ens_ts


def LM3_coar(X, Z, coarse, K=32, I=12, F=15, b=10, c=2.5, alpha=None, beta=None):
    if alpha is None:
        alpha = (3 * I**2 + 3) / (2 * I**3 + 4 * I)
    if beta is None:
        beta = (2 * I**2 + 1) / (I**4 + 2 * I**2)
    X_Z = window_sum_Z(Z, I=I, alpha=alpha, beta=beta)
    Y = Z - X_Z

    K_coar = K//coarse

    T1 = bracket([X], K_coar)
    T2 = b**2 * bracket([Y], 1)
    T2 = window_sum_Z(T2, I=I, alpha=alpha, beta=beta)
    T2 = T2[::coarse]
    T3 = c * bracket([Y, X_Z], 1)
    T3 = window_sum_Z(T3, I=I, alpha=alpha, beta=beta)
    T3 = T3[::coarse]
    T4 = -1 * X
    T5 = -b * Y
    T5 = window_sum_Z(T5, I=I, alpha=alpha, beta=beta)
    T5 = T5[::coarse]
    dXdt = T1 + T2 + T3 + T4 + T5 + F

    return dXdt


def return_LM3_coar_data_sp(X0, t, Z, coarse=8,
                         K=32, I=12, F=15,
                         b=10, c=2.5, alpha=None, beta=None):
    if alpha is None:
        alpha = (3 * I**2 + 3) / (2 * I**3 + 4 * I)
    if beta is None:
        beta = (2 * I**2 + 1) / (I**4 + 2 * I**2)

    dt = t[1] - t[0]
    def this_LM3_coar(X, t):
        Z_index = int(np.floor(t / dt))
        print(Z_index)
        dXdt = LM3_coar(X, Z[:, Z_index], coarse=coarse, K=K,
                        I=I, F=F, b=b, c=c, alpha=alpha, beta=beta)
        return dXdt
    X = integrate.odeint(this_LM3_coar, X0, t)
    X = X.T
    return X


def return_LM3_coar_ens_data_sp(X0_ens, t, Z0_ens_ts, coarse=8, K=32, I=12, F=15, b=10, c=2.5, alpha=None, beta=None):
    if alpha is None:
        alpha = (3 * I**2 + 3) / (2 * I**3 + 4 * I)
    if beta is None:
        beta = (2 * I**2 + 1) / (I**4 + 2 * I**2)

    N_t = t.size
    dt = t[1] - t[0]
    N_Xc, N_eXc = X0_ens.shape
    N_Z, N_eZ, N_Zt = Z0_ens_ts.shape
    N_eXpZ = N_eXc // N_eZ

    X_ens = np.ones([N_Xc, N_eXc, N_t])

    for ens_count in range(N_eZ):

        def this_LM3_coar(X_ens_ravel, t):
            Z_index = int(np.floor(t/dt))
            Z_index = np.min([Z_index,
                              Z0_ens_ts.shape[-1] - 1])
            X_ens = X_ens_ravel.reshape(N_Xc, N_eXpZ)
            # print('t/dt: ', t/dt)
            # print('Z_index: ', Z_index)
            dXdt = LM3_coar(X_ens,
                            Z0_ens_ts[:, ens_count, Z_index][:, None],
                            coarse=coarse,
                            K=K, I=I, F=F,
                            b=b, c=c,
                            alpha=alpha, beta=beta)
            return dXdt.ravel()

        this_slice = slice(ens_count*N_eXpZ, (ens_count+1)*N_eXpZ)
        aX_ens = integrate.odeint(this_LM3_coar,
                                  X0_ens[:, this_slice].ravel(), t)
        aX_ens = aX_ens.T
        aX_ens = aX_ens.reshape(N_Xc, N_eXpZ, N_t)
        X_ens[:, this_slice, :] = aX_ens
    return X_ens


def RK4(f, y, dt, t):
    K1 = dt * f(y, t)
    K2 = dt * f(y + K1/2, t + dt/2)
    K3 = dt * f(y + K2/2, t + dt/2)
    K4 = dt * f(y + K3, t + dt/2)
    y1 = y + (K1 + 2*K2 + 2*K3 + K4)/6
    return y1


def return_LM3_data(Z0, dt, T, dt_obs,
                    K=32, I=12, F=15, b=10, c=2.5,
                    alpha=None, beta=None):
    if alpha is None:
        alpha = (3 * I**2 + 3) / (2 * I**3 + 4 * I)
    if beta is None:
        beta = (2 * I**2 + 1) / (I**4 + 2 * I**2)
    N_Z = Z0.size
    def this_LM3(Z, t):
        dZdt = LM3(Z, K=K, I=I, F=F, b=b, c=c, alpha=alpha, beta=beta)
        return dZdt
    Nz = Z0.size
    Nt = int(T/dt) + 1
    Nto = int(T/dt_obs) + 1
    Z = np.ones([Nz, Nto])*np.nan
    every = int(dt_obs/dt)
    #print(Z0)
    Z[:, 0] = Z0.copy()
    Zprev = Z0.copy()
    count_obs = 0
    every_print = int(Nt/10)
    if every != 1:
        for count in range(Nt - 1):
            Zprev = RK4(this_LM3, Zprev, dt, np.nan)
            if (count + 1) % every == 0:
                count_obs += 1
                Z[:, count_obs] = Zprev
            if count + 1 % every_print == 0:
                print((ii*100)//Nt + 1)
    else:
        for count in range(Nt - 1):
            Zprev = RK4(this_LM3, Zprev, dt, np.nan)
            Z[:, count + 1] = Zprev
            if count + 1 % every_print == 0:
                print((ii*100)//Nt + 1)
    return Z


def return_LM3_ens_data(Z0_ens, dt, T, dt_obs, K=32, I=12, F=15,
                        b=10, c=2.5, alpha=None, beta=None):
    if alpha is None:
        alpha = (3 * I**2 + 3) / (2 * I**3 + 4 * I)
    if beta is None:
        beta = (2 * I**2 + 1) / (I**4 + 2 * I**2)
    Nz, Nez = Z0_ens.shape
    Nt = int(T/dt) + 1
    Nto = int(T/dt_obs) + 1

    def this_LM3(Z_1d_ens, t):
        Z_ens = Z_1d_ens.reshape(Nz, Nez)
        dZdt = LM3(Z_ens, K=K, I=I, F=F, b=b, c=c, alpha=alpha, beta=beta)
        return dZdt.ravel()
    Z_ens_ts = np.ones([Nz, Nez, Nto])*np.nan
    every = int(dt_obs/dt)
    #print(Z0)
    Z_ens_ts[:, :,  0] = Z0_ens.copy()
    Zprev = Z0_ens.ravel()
    count_obs = 0
    every_print = int(Nt/10)
    if every != 1:
        for count in range(Nt - 1):
            Zprev = RK4(this_LM3, Zprev, dt, np.nan)
            if (count + 1) % every == 0:
                count_obs += 1
                Z_ens_ts[:, :, count_obs] = Zprev.reshape(Nz, Nez)
            # if count + 1 % every_print == 0:
            #     print((ii*100)//Nt + 1)
    else:
        for count in range(Nt - 1):
            Zprev = RK4(this_LM3, Zprev, dt, np.nan)
            Z_ens_ts[:, :, count] = Zprev.reshape(Nz, Nez)
            # if count + 1 % every_print == 0:
            #     print((ii*100)//Nt + 1)
    return Z_ens_ts


def return_LM3_coar_data(X0, dt, T, dt_obs, Z, tz,
                         coarse=8,
                         K=32, I=12, F=15,
                         b=10, c=2.5, alpha=None, beta=None):
    if alpha is None:
        alpha = (3 * I**2 + 3) / (2 * I**3 + 4 * I)
    if beta is None:
        beta = (2 * I**2 + 1) / (I**4 + 2 * I**2)
    dtz = tz[1] - tz[0]
    def this_LM3_coar(X, t):
        Z_index = int(np.floor(t / dtz))
        Z_index = np.min([Z_index,
                          tz.size - 1])
        dXdt = LM3_coar(X, Z[:, Z_index], coarse=coarse, K=K,
                        I=I, F=F, b=b, c=c, alpha=alpha, beta=beta)
        return dXdt

    Nx = X0.size
    Nt = int(T/dt) + 1
    Nto = int(T/dt_obs) + 1
    X = np.ones([Nx, Nto])*np.nan
    every = int(dt_obs/dt)
    #print(Z0)
    X[:, 0] = X0.copy()
    Xprev = X0.copy()
    count_obs = 0
    every_print = int(Nt/10)
    if every != 1:
        for count in range(Nt - 1):
            Xprev = RK4(this_LM3_coar, Xprev, dt, tz[count])
            if (count + 1) % every == 0:
                count_obs += 1
                X[:, count_obs] = Xprev
            if count + 1 % every_print == 0:
                print((ii*100)//Nt + 1)
    else:
        for count in range(Nt - 1):
            Xprev = RK4(this_LM3_coar, Xprev, dt, tz[count])
            X[:, count] = Xprev
            if count + 1 % every_print == 0:
                print((ii*100)//Nt + 1)
    return X


def return_LM3_coar_ens_data(X0_ens, t, Z0_ens_ts, coarse=8, K=32, I=12, F=15, b=10, c=2.5, alpha=None, beta=None):
    if alpha is None:
        alpha = (3 * I**2 + 3) / (2 * I**3 + 4 * I)
    if beta is None:
        beta = (2 * I**2 + 1) / (I**4 + 2 * I**2)

    N_t = t.size
    dt = t[1] - t[0]
    N_Xc, N_eXc = X0_ens.shape
    N_Z, N_eZ, N_Zt = Z0_ens_ts.shape
    N_eXpZ = N_eXc // N_eZ

    X_ens = np.ones([N_Xc, N_eXc, N_t])

    for ens_count in range(N_eZ):

        def this_LM3_coar(X_ens_ravel, t):
            Z_index = int(np.floor(t/dt))
            Z_index = np.min([Z_index,
                              Z0_ens_ts.shape[-1] - 1])
            X_ens = X_ens_ravel.reshape(N_Xc, N_eXpZ)
            dXdt = LM3_coar(X_ens,
                            Z0_ens_ts[:, ens_count, Z_index][:, None],
                            coarse=coarse,
                            K=K, I=I, F=F,
                            b=b, c=c,
                            alpha=alpha, beta=beta)
            return dXdt.ravel()

        this_slice = slice(ens_count*N_eXpZ, (ens_count+1)*N_eXpZ)
        aX_ens = integrate.odeint(this_LM3_coar,
                                  X0_ens[:, this_slice].ravel(), t)
        aX_ens = aX_ens.T
        aX_ens = aX_ens.reshape(N_Xc, N_eXpZ, N_t)
        X_ens[:, this_slice, :] = aX_ens
    return X_ens
