import numpy as np
import scipy as sp

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
