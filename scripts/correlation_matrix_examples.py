import numpy as np
import scipy as sp
import scipy.special
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import multi_loc.covariance as covariance

def main():
    # Compare matern to exp
    rho = np.arange(10)
    rho0 = 10
    nu = .5

    print('nu: ', nu)
    print('matern: ', covariance.correlation_matern(rho, rho0, nu), '\n')

    print('exp: ', covariance.correlation_exp(rho, rho0, nu), '\n')

    # compare matern to sqd_exp
    rho = np.arange(10)
    rho0 = 10
    nu = 150


    print('nu: ', nu)
    print('matern: ', covariance.correlation_matern(rho, rho0, nu), '\n')

    print('sqd_exp: ', covariance.correlation_sqd_exp(rho, rho0, nu), '\n')

    # plot example covariance matrix for a ring
    dimension = 100
    rho0 = 20
    rho = np.arange(dimension, dtype=float)
    rho = np.minimum(rho % dimension, (dimension - rho) % dimension)

    cov_P = covariance.make_correlation_matrix(rho, rho0,
                                           covariance.correlation_sqd_exp)

    plt.figure()
    image = plt.imshow(cov_P)
    plt.colorbar(image)
    plt.title('Covariance matrix for a ring')
    plt.show(block=False)

    eig_val, eig_vec = sp.linalg.eigh(cov_P)
    eig_val = eig_val.clip(min=0)
    eig_val = eig_val[::-1]
    eig_vec = eig_vec[:, ::-1]

    plt.figure()
    plt.semilogy(eig_val)
    plt.title('Eigenvalues after clipping at zero')
    plt.ylabel('Eigenvalue')
    plt.xlabel('Eigenvalue number')
    plt.show(block=False)

    sqrt_P = eig_vec @ np.diag(np.sqrt(eig_val)) @ eig_vec.T

    plt.figure()
    image = plt.imshow(sqrt_P @ sqrt_P)
    plt.colorbar(image)
    plt.title('Reconstructed covariance matrix')
    plt.show(block=False)


    # # Check ensemble est. of covariance
    dimension = 100
    ens_size = 100
    rho0 = 0.1
    num_realizations = 200
    subspace_dim = 5
    nu = 3/2
    rho = np.arange(dimension)/dimension

    cov_P = covariance.make_correlation_matrix(
        rho, rho0, covariance.correlation_matern, nu=nu)

    eig_val, eig_vec = sp.linalg.eigh(cov_P)
    eig_val = eig_val.clip(min=0)
    eig_val = eig_val[::-1]
    eig_vec = eig_vec[:, ::-1]
    sqrt_P = eig_vec @ np.sqrt(cov_P) @ eig_vec.T
    eig_vec_save = np.zeros([dimension, subspace_dim, num_realizations])
    eig_val_save = np.zeros([subspace_dim, num_realizations])

    for ii in range(num_realizations):
        ens_X = sqrt_P @ np.random.randn(dimension, ens_size)
        ens_X = ens_X - np.repeat(ens_X.mean(axis=1)[:, None], ens_size, axis=1)
        ens_X = ens_X / np.sqrt(ens_size - 1)
        hat_P = ens_X @ ens_X.T
        eig_val_save[:, ii], eig_vec_save[:, :, ii] = sp.linalg.eigh(
            hat_P, eigvals=((dimension - subspace_dim), dimension - 1))

    max_lambda = np.zeros([num_realizations, num_realizations])
    min_lambda = max_lambda.copy()
    sum_lambda = max_lambda.copy()
    for ii in range(num_realizations):
        for jj in range(num_realizations):
            IPs = eig_vec_save[:, :, ii].T @ eig_vec_save[:, :, jj]
            singular_values = sp.linalg.svd(IPs, compute_uv=False)
            max_lambda[ii, jj] = singular_values.max()
            min_lambda[ii, jj] = singular_values.min()
            sum_lambda[ii, jj] = (singular_values**2).sum()

    this = max_lambda.mean()

    plt.figure()
    image = plt.imshow(max_lambda)
    plt.colorbar(image)
    plt.title(f'max lambda; mean: {this:0.4}')
    plt.xlabel('Realization')
    plt.ylabel('Realization')
    plt.show(block=False)

    this = min_lambda.mean()

    plt.figure()
    image = plt.imshow(min_lambda)
    plt.colorbar(image)
    plt.title(f'min lambda; mean: {this:0.4}')
    plt.xlabel('Realization')
    plt.ylabel('Realization')
    plt.show(block=False)

    this = sum_lambda.mean()

    plt.figure()
    image = plt.imshow(sum_lambda)
    plt.colorbar(image)
    plt.title(f'sum lambda; mean: {this:0.4}')
    plt.xlabel('Realization')
    plt.ylabel('Realization')
    plt.show(block=False)

    plt.show()

if __name__ == '__main__':
    main()
