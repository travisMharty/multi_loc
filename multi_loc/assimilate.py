import numpy as np
import scipy as sp
from multi_loc import covariance


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
    eig_condition = ((eig_val is None )
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


def trans_assim_trials(*, mu, H, ens_size, assim_num,
                       trial_num, true_mats, trans_mats=None,
                       ground_truth=None, obs_array=None,
                       ensemble_array=None):
    """

    """
    state_size = H.shape[1]
    obs_size = H.shape[0]

    # initialize arrays
    rmse = np.ones([trial_num, assim_num + 1]) * np.nan

    # load arrays from dictionary
    P_sqrt_truth = true_mats['P_sqrt']
    R_sqrt = true_mats['R_sqrt']

    # generate arrays
    if ground_truth is None:
        ground_truth = generate_ensemble(
            trial_num, mu, P_sqrt_truth)
    if obs_array is None:
        obs_array = ((H @ ground_truth)[:, :, None]
                     + np.einsum(
                         'ij, jk... ->ik...', R_sqrt,
                         np.random.randn(obs_size, trial_num, assim_num)))
    if ensemble_array is None:
        ensemble_array = np.ones(
            [state_size, ens_size, trial_num, assim_num + 1]) * np.nan
        for t_num in range(trial_num):
            ensemble_array[:, :, t_num, 0] = generate_ensemble(
                ens_size, mu, P_sqrt_truth)
    for t_num in range(trial_num):
        error = (ground_truth[:, t_num]
                 - ensemble_array[:, :, t_num, 0].mean(axis=1))
        rmse[t_num, 0] = np.sqrt((error**2).mean())
        for a_num in range(assim_num):
            # add test to generate trans matrices
            y_obs = obs_array[:, t_num, a_num]
            ensemble_array[:, :, t_num, a_num + 1] = assimilate_TEnKF(
                ensemble=ensemble_array[:, :, t_num, a_num],
                y_obs=y_obs, H=H, trans_mats=trans_mats)
            error = (ground_truth[:, t_num]
                     - ensemble_array[:, :, t_num, a_num + 1].mean(axis=1))
            rmse[t_num, a_num + 1] = np.sqrt((error**2).mean())
    to_return = {
        'ensemble_array': ensemble_array, 'ground_truth': ground_truth,
        'obs_array': obs_array, 'rmse': rmse, 'trans_mats': trans_mats
    }
    return to_return


def assimilate_TEnKF(*, ensemble, y_obs, H, trans_mats):
    """

    """
    obs_size = H.shape[0]
    ens_size = ensemble.shape[1]

    # Load or calculate transformation matrices
    if trans_mats is not None:
        Tx = trans_mats['Tx']
        Tx_inv = trans_mats['Tx_inv']
        Ty = trans_mats['Ty']
        S_reduced = trans_mats['S'].diagonal()[:, None]
        S = trans_mats['S']
    else:
        calc_trans_mats()

    # Transform to diagonal space
    y_trans = (Ty @ y_obs)[:, None]
    ens_trans = Tx @ ensemble

    y_trans = y_trans + np.random.randn(obs_size, ens_size)
    P_reduced = np.var(ens_trans[:obs_size], axis=1)[:, None]
    K = ((S_reduced * P_reduced)
         / (1 + S_reduced**2 * P_reduced))
    ens_trans[:obs_size] = (ens_trans[:obs_size]
                         + K * (y_trans - S @ ens_trans))
    return Tx_inv @ ens_trans


def calc_trans_mats():
    return None
