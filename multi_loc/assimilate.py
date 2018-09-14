import math
import numpy as np
import scipy as sp
from multi_loc import covariance
from sklearn.utils.extmath import randomized_svd

import multi_loc.utilities as utilities


def generate_ensemble(ens_size, mu, P_sqrt=None, eig_val=None, eig_vec=None):
    """
    Return ensemble drawn from N(mu, P).

    Parameters
    ----------
    ens_size : scalar
        Number of ensemble members to return.

    mu : ndarray
        Mean of the returned ensemble samples. P.shape must equal
        (mu.size, mu.size).

    P_sqrt : ndarray
        Square root of the desired covariance of the ensemble samples.

    eig_val : ndarray
        Eigenvalues corresponding to the correlation matrix P.
        If eig_val and eig_vec are provided, P_sqrt is not used.

    eig_vec : ndarray
        Eigenvectors corresponding to the correlation matrix P.
        If eig_val and eig_vec are provided, P_sqrt is not used.

    Returns
    ------
    ens : ndarray
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


def is_diag(X, atol=1e-16, rtol=1e-16):
    """
    Returns True if matrix X is diagonal to a specific tolerance.

    Parameters
    ----------
    X : ndarray
        The Matrix which will be tested for being diagonal.

    atol : scalar
        The absolute tolerance which will be used in np.allclose().

    rtol : scalar
        The relative tolerance which will be used in np.allclose().

    Returns
    -------
    diag_bool : bool
        The Boolean which will be True is X is diagonal and false otherwise.
    """
    diag_bool = np.allclose(X, np.diag(np.diag(X)),
                            atol=atol, rtol=rtol)
    return diag_bool


def inverse_sqrt(C=None, eig_val=None, eig_vec=None):
    """
    Return square root and inverse square root of a matrix.

    Parameters
    ----------
    C : ndarray
        The matrix for which a square root and inverse square root will be
        calculated.

    eig_val : ndarray
        The eigenvalues for C as a vector.

    eig_vec : ndarray
        The eigenvectors for C corresponding to eig_val.

    Returns
    -------
    C_sqrt : ndarray
        The square root of C.

    C_inv_sqrt : ndarray
        The inverse square root of C.
    """
    if C is not None:
        C_diag = is_diag(C)
    else:
        C_diag = False
    eig_condition = ((eig_val is None)
                     and (eig_vec is None)
                     and not C_diag)
    if eig_condition:
        C_sqrt, eig_val, eig_vec = covariance.matrix_sqrt(C=C,
                                                          return_eig=True)
        C_sqrt = C_sqrt.real
        C_inv_sqrt = covariance.matrix_sqrt_inv(eig_val=eig_val,
                                                eig_vec=eig_vec)
        C_inv_sqrt = C_inv_sqrt.real
    elif C_diag:
        diag = np.diag(C)
        C_sqrt = np.sqrt(diag)
        C_inv_sqrt = 1/C_sqrt
        C_sqrt = np.diag(C_sqrt)
        C_inv_sqrt = np.diag(C_inv_sqrt)
    else:
        C_sqrt = covariance.matrix_sqrt(eig_val=eig_val,
                                        eig_vec=eig_vec)
        C_sqrt = C_sqrt.real
        C_inv_sqrt = covariance.matrix_sqrt_inv(eig_val=eig_val,
                                                eig_vec=eig_vec)
        C_inv_sqrt = C_inv_sqrt.real
    return C_sqrt, C_inv_sqrt


def transformation_matrices(H, eig_val_p=None, eig_vec_p=None, P=None,
                            eig_val_r=None, eig_vec_r=None, R=None,
                            return_Ts=False):
    """
    Return the matrices needed to perform the optimal localization. If P(R) is
    diagonal, then eig_val_p(eig_val_r) and eig_vec_p(eig_vec_r) will not be
    used. If P(R) is not diagonal and eig_val_p(eig_val_r) and
    eig_vec_p(eig_vec_r) are provided, then P(R) will not be used.

    Parameters
    ----------
    H : ndarray
        The forward observation matrix.

    eig_val_p : ndarray
        The eigenvalues of P.

    eig_vec_p : ndarray
        The eigenvectors of P.

    P : ndarray
        The state space error covariance matrix.

    eig_val_p : ndarray
        The eigenvalues of P.

    eig_vec_p : ndarray
        The eigenvectors of P.

    R : ndarray
        The observation error covariance matrix.

    eig_val_r : ndarray
        The eigenvalues of R.

    eig_vec_r : ndarray
        The eigenvectors of R.

    Returns
    -------
    P_sqrt : ndarray
        The square root of P.

    P_inv_sqrt : ndarray
        The inverse of the square root of P.

    R_sqrt : ndarray
        The square root of R.

    R_inv_sqrt : ndarray
        The inverse of the square root of R.

    U : ndarray
        The left singular vectors of R_inv_sqrt @ H @ P_sqrt.

    S : ndarray
        The singular values of R_inv_sqrt @ H @ P_sqrt.

    VT : ndarray
        The transpose of the right singular vectors of R_inv_sqrt @ H @ P_sqrt.

    Tx : ndarray
        Transformation to localized space for state space.

    Tx_inv : ndarray
        Inverse of Tx.

    Ty : ndarray
        Transformation to localized space for observation space.

    Ty_inv : ndarray
        Inverse of Ty.
    """
    dimension = H.shape[1]
    y_size = H.shape[0]

    P_sqrt, P_inv_sqrt = inverse_sqrt(C=P, eig_val=eig_val_p,
                                      eig_vec=eig_vec_p)
    R_sqrt, R_inv_sqrt = inverse_sqrt(C=R, eig_val=eig_val_r,
                                      eig_vec=eig_vec_r)
    U, S, VT = sp.linalg.svd(R_inv_sqrt @ H @ P_sqrt)
    S = np.diag(S)
    S = np.concatenate(
        [S, np.zeros([y_size, dimension - y_size])],
        axis=1)
    to_return = {'P_sqrt': P_sqrt, 'P_inv_sqrt': P_inv_sqrt,
                 'R_sqrt': R_sqrt, 'R_inv_sqrt': R_inv_sqrt,
                 'U': U, 'S': S, 'VT': VT, 'H': H}
    if return_Ts:
        Tx = VT @ P_inv_sqrt
        Tx_inv = P_sqrt @ VT.conj().T
        Ty = U.conj().T @ R_inv_sqrt
        Ty_inv = R_sqrt @ U
        to_return['Tx'] = Tx
        to_return['Tx_inv'] = Tx_inv
        to_return['Ty'] = Ty
        to_return['Ty_inv'] = Ty_inv
        return to_return
    else:
        return to_return


def random_H(N, obs_size):
    """
    Returns random forward observation matrix which will observe obs_size
    locations of a state space which is of size N. The observation locations
    will almost surely not fall on a state space location. The nearest two
    state locations will be interpolated to the observation location.
    """
    locs = np.random.choice(
        np.arange(N), size=obs_size, replace=False)
    locs.sort()
    locs = locs.astype(int)
    dist = np.random.uniform(0, 1, size=obs_size)
    H = np.zeros([obs_size, N])
    rows = np.arange(obs_size, dtype=int)
    H[rows, locs] = dist
    locs = (locs + 1) % N
    H[rows, locs] = 1 - dist
    return H


def multi_loc_opt(*, sig_array, rho_array, P_sample, P_sample_array,
                  H, U, S, VT, stopping_angle, est_rank):
    obs_size, dimension = H.shape
    ens_ens_size = P_sample_array.shape[-1]
    total_sig = sig_array.sum()
    sig_bin_num = sig_array.size
    dx = 1/dimension
    comb_num = round(
        math.factorial(ens_ens_size)
        / (math.factorial(2)
           * math.factorial(ens_ens_size - 2)))

    s_array = np.ones([total_sig, rho_array.size, ens_ens_size]) * np.nan
    U_array = np.ones([obs_size, total_sig,
                       rho_array.size, ens_ens_size]) * np.nan
    V_array = np.ones([dimension, total_sig,
                       rho_array.size, ens_ens_size]) * np.nan
    sig_num = sig_array[0]

    eye_array = np.repeat(np.eye(dimension)[:, :, None],
                          ens_ens_size, axis=-1)
    proj_array = eye_array.copy()
    proj = np.eye(dimension)

    V_average_angle_2_truth = np.ones(
        [sig_bin_num, rho_array.size]) * np.nan
    V_average_angle = np.ones(
        [sig_bin_num, rho_array.size]) * np.nan

    opt_rho_array = np.ones(sig_bin_num) * np.nan
    opt_s_array_ens = np.ones([total_sig, ens_ens_size]) * np.nan
    opt_U_array_ens = np.ones([obs_size, total_sig, ens_ens_size]) * np.nan
    opt_V_array_ens = np.ones([dimension, total_sig, ens_ens_size]) * np.nan

    opt_s_array = np.ones([0])
    opt_U_array = np.ones([obs_size, 0])
    opt_V_array = np.ones([dimension, 0])

    proj = np.eye(dimension)
    last_sig = 0
    for sig_count, sig_num in enumerate(sig_array):
        sig_slice = slice(last_sig, last_sig + sig_num)
        print('')
        print(sig_slice)
        last_sig = last_sig + sig_num
        true_V = VT[sig_slice].T
        for rho_count, rho_loc in enumerate(rho_array):
            print(rho_count, end='; ')
            [loc] = covariance.generate_circulant(
                dimension, dx, rho_loc, covariance.fft_sqd_exp_1d,
                return_Corr=True, return_eig=False)
            loc /= loc.max()
            for ens_count in range(ens_ens_size):
                P_loc = P_sample_array[:, :, ens_count] * loc
                this_P_sqrt = covariance.matrix_sqrt(P_loc).real
                aU, aS, aVT = randomized_svd(
                    H @ this_P_sqrt @ proj_array[:, :, ens_count],
                    n_components=sig_num)
                aS = np.diag(aS)

                U_array[:, sig_slice, rho_count, ens_count] = aU
                s_array[sig_slice, rho_count, ens_count] = aS.diagonal()
                V_array[:, sig_slice, rho_count, ens_count] = aVT.T
            angle_2_truth = np.ones(ens_ens_size) * np.nan
            comb_count = 0
            angles = np.ones(comb_num) * np.nan
            for ens_count in range(ens_ens_size):
                aV = V_array[:, sig_slice, rho_count, ens_count]
                angle_2_truth[ens_count] = utilities.angle(aV, true_V)
                for other_ens_count in range(ens_count + 1, ens_ens_size):
                    oV = V_array[:, sig_slice, rho_count, other_ens_count]
                    this_angle = utilities.angle(aV, oV)
                    angles[comb_count] = this_angle
                    comb_count += 1
            V_average_angle[sig_count, rho_count] = angles.mean()
            this_angle = angle_2_truth.mean()
            V_average_angle_2_truth[sig_count, rho_count] = this_angle
        opt_rho_index = V_average_angle[sig_count].argmin()
        if V_average_angle[sig_count, opt_rho_index] > stopping_angle:
            [loc] = covariance.generate_circulant(
                dimension, dx, opt_rho_array[sig_count - 1],
                covariance.fft_sqd_exp_1d,
                return_Corr=True, return_eig=False)
            loc /= loc.max()

            P_loc = P_sample * loc
            this_P_sqrt = covariance.matrix_sqrt(P_loc).real
            break
        else:
            opt_rho_array[sig_count] = rho_array[opt_rho_index]
            opt_U_array_ens[:, sig_slice] = U_array[:, sig_slice,
                                                    opt_rho_index, :]
            opt_s_array_ens[sig_slice] = s_array[sig_slice, opt_rho_index, :]
            opt_V_array_ens[:, sig_slice] = V_array[:, sig_slice,
                                                    opt_rho_index, :]
            proj_array = eye_array - np.einsum(
                'ij...,kj...->ik...',
                opt_V_array_ens[:, :sig_slice.stop],
                opt_V_array_ens[:, :sig_slice.stop])

            # calculate final V
            [loc] = covariance.generate_circulant(
                dimension, dx, opt_rho_array[sig_count],
                covariance.fft_sqd_exp_1d,
                return_Corr=True, return_eig=False)
            loc /= loc.max()

            P_loc = P_sample * loc
            this_P_sqrt = covariance.matrix_sqrt(P_loc).real
            aU, aS, aVT = randomized_svd(
                H @ this_P_sqrt @ proj,
                n_components=sig_num)
            opt_U_array = np.concatenate([opt_U_array, aU], axis=1)
            opt_s_array = np.concatenate([opt_s_array, aS], axis=0)
            opt_V_array = np.concatenate([opt_V_array, aVT.T], axis=1)
            proj = np.eye(dimension) - (opt_V_array
                                        @ opt_V_array.T)

    previous_sigs = sig_array[:sig_count - 1].sum()
    needed_sigs = est_rank - previous_sigs
    if needed_sigs > 0:
        aU, aS, aVT = randomized_svd(
            H @ this_P_sqrt @ proj,
            n_components=needed_sigs)
        opt_U_array = np.concatenate([opt_U_array, aU], axis=1)
        opt_s_array = np.concatenate([opt_s_array, aS], axis=0)
        opt_V_array = np.concatenate([opt_V_array, aVT.T], axis=1)
    a_dict = {'opt_rho_array': opt_rho_array,
              'opt_U_array_ens': opt_U_array_ens,
              'opt_s_array_ens': opt_s_array_ens,
              'opt_V_array_ens': opt_V_array_ens,
              'opt_U_array': opt_U_array,
              'opt_s_array': opt_s_array,
              'opt_V_array': opt_V_array,
              'V_average_angle': V_average_angle,
              'V_average_angle_2_truth': V_average_angle_2_truth}
    return a_dict


def multi_loc_dictate_rho(
        *, sig_array, opt_rho_array, P_sample,
        H, U, S, VT, est_rank):
    obs_size, dimension = H.shape

    total_sig = sig_array.sum()
    sig_bin_num = sig_array.size
    dx = 1/dimension

    opt_s_array = np.ones([0])
    opt_U_array = np.ones([obs_size, 0])
    opt_V_array = np.ones([dimension, 0])

    proj = np.eye(dimension)
    last_sig = 0
    for sig_count, sig_num in enumerate(sig_array):
        sig_slice = slice(last_sig, last_sig + sig_num)
        print('')
        print(sig_slice)
        last_sig = last_sig + sig_num
        [loc] = covariance.generate_circulant(
            dimension, dx, opt_rho_array[sig_count],
            covariance.fft_sqd_exp_1d,
            return_Corr=True, return_eig=False)
        loc /= loc.max()
        P_loc = P_sample * loc
        this_P_sqrt = covariance.matrix_sqrt(P_loc).real
        aU, aS, aVT = randomized_svd(
            H @ this_P_sqrt @ proj,
            n_components=sig_num)
        opt_U_array = np.concatenate([opt_U_array, aU], axis=1)
        opt_s_array = np.concatenate([opt_s_array, aS], axis=0)
        opt_V_array = np.concatenate([opt_V_array, aVT.T], axis=1)
        proj = np.eye(dimension) - (opt_V_array
                                    @ opt_V_array.T)
    previous_sigs = sig_array.sum()
    needed_sigs = est_rank - previous_sigs
    if needed_sigs > 0:
        aU, aS, aVT = randomized_svd(
            H @ this_P_sqrt @ proj,
            n_components=needed_sigs)
        opt_U_array = np.concatenate([opt_U_array, aU], axis=1)
        opt_s_array = np.concatenate([opt_s_array, aS], axis=0)
        opt_V_array = np.concatenate([opt_V_array, aVT.T], axis=1)
    a_dict = {'opt_U_array': opt_U_array,
              'opt_s_array': opt_s_array,
              'opt_V_array': opt_V_array}
    return a_dict


# def multi_loc_dictate_rho(
#         *, sig_array, opt_rho_array, P_sample,
#         H, U, S, VT, est_rank):
#     obs_size, dimension = H.shape
#     dx = 1/dimension

#     opt_s_array = np.ones([0])
#     opt_U_array = np.ones([obs_size, 0])
#     opt_V_array = np.ones([dimension, 0])

#     proj = np.eye(dimension)
#     last_sig = 0
#     for sig_count, sig_num in enumerate(sig_array):
#         sig_slice = slice(last_sig, last_sig + sig_num)
#         print('')
#         print(sig_slice)
#         last_sig = last_sig + sig_num
#         [loc] = covariance.generate_circulant(
#             dimension, dx, opt_rho_array[sig_count],
#             covariance.fft_sqd_exp_1d,
#             return_Corr=True, return_eig=False)
#         loc /= loc.max()
#         P_loc = P_sample * loc
#         this_P_sqrt = covariance.matrix_sqrt(P_loc).real
#         aU, aS, aVT = randomized_svd(
#             H @ this_P_sqrt @ proj,
#             n_components=sig_num)
#         opt_U_array = np.concatenate([opt_U_array, aU], axis=1)
#         opt_s_array = np.concatenate([opt_s_array, aS], axis=0)
#         opt_V_array = np.concatenate([opt_V_array, aVT.T], axis=1)
#         proj = np.eye(dimension) - (opt_V_array
#                                     @ opt_V_array.T)
#     previous_sigs = sig_array.sum()
#     needed_sigs = est_rank - previous_sigs
#     if needed_sigs > 0:
#         aU, aS, aVT = randomized_svd(
#             H @ this_P_sqrt @ proj,
#             n_components=needed_sigs)
#         opt_U_array = np.concatenate([opt_U_array, aU], axis=1)
#         opt_s_array = np.concatenate([opt_s_array, aS], axis=0)
#         opt_V_array = np.concatenate([opt_V_array, aVT.T], axis=1)
#     a_dict = {'opt_U_array': opt_U_array,
#               'opt_s_array': opt_s_array,
#               'opt_V_array': opt_V_array}
#     return a_dict
