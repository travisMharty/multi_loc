import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import multi_loc.covariance as covariance

def imshow(X, title=None):
    plt.figure()
    im = plt.imshow(X)
    plt.colorbar(im)
    plt.title(title)
    plt.show(block=True)

def main():
    dimension = 100
    rho0 = 20
    ens_size = int(40)
    obs_size = int(1e4)
    mu = np.zeros(dimension)
    H = np.eye(dimension)[::2]
    y_size = H.shape[0]
    R = np.eye(y_size)
    R_inv = R.copy()
    R_sqrt = R.copy()
    R_inv_sqrt = R.copy()

    # Calculate distance on a ring
    rho = np.arange(dimension, dtype=float)
    rho = np.minimum(rho, (dimension - rho) % dimension)
    eig_val, eig_vec, P = covariance.generate_circulant(
        rho, rho0, covariance.correlation_exp,
        return_Corr = True)
    P = P.real

    P_sqrt = covariance.matrix_sqrt(eig_val=eig_val,
                                    eig_vec=eig_vec)
    P_sqrt = P_sqrt.real

    P_inv_sqrt = covariance.matrix_sqrt_inv(
        eig_val=eig_val,
        eig_vec=eig_vec)
    P_inv_sqrt = P_inv_sqrt.real

    X_ens = covariance.generate_ensemble(
        ens_size, mu, P_sqrt)
    ground_truth = covariance.generate_ensemble(
        1, mu, P_sqrt)

    Y_ens = H @ ground_truth + R_sqrt @ np.random.randn(y_size, obs_size)

    mu_sample = X_ens.mean(axis=1)
    P_sample = np.cov(X_ens)
    X_ens_til = (
        P_inv_sqrt @ X_ens)
    Y_ens_til = (
        R_inv_sqrt @ Y_ens)
    mu_til = X_ens_til.mean(axis=1)
    P_til = np.cov(X_ens_til)

    alpha = .5
    P_til_loc = (1 - alpha) * P_til + alpha * np.eye(P_til.shape[1])

    P_sample_loc = P_sqrt @ P_til_loc @ P_sqrt

    U, S, VT = sp.linalg.svd(R_inv_sqrt @ H @ P_sqrt)
    m = U.shape[0]
    n = VT.shape[0]
    S = np.diag(S)
    S = np.concatenate([S, np.zeros([m, n-m])], axis=1)

    X_ens_p = VT @ X_ens_til
    Y_ens_p = U.T @ Y_ens_til

    # one update
    mu_p = X_ens_p.mean(axis=1)[:, None]
    Y_obs = Y_ens_p[:, 0][:, None] + R_sqrt @ np.random.randn(y_size, ens_size)
    K = (S.diagonal()/(1 + S.diagonal()**2))[:, None]
    X_ens_p_hat = X_ens_p.copy()
    X_ens_p_hat[:y_size] = mu_p[:y_size] + K * (Y_obs - S @ mu_p)

    # Summary
    P = P.real

    P_sqrt = covariance.matrix_sqrt(eig_val=eig_val,
                                    eig_vec=eig_vec)
    P_sqrt = P_sqrt.real

    P_inv = covariance.matrix_inv(eig_val=eig_val,
                                  eig_vec=eig_vec)
    P_inv = P_inv.real

    P_inv_sqrt = covariance.matrix_sqrt_inv(
        eig_val=eig_val,
        eig_vec=eig_vec)

    P_inv_sqrt = P_inv_sqrt.real
    U, S, VT = sp.linalg.svd(R_inv_sqrt @ H @ P_sqrt)
    m = U.shape[0]
    n = VT.shape[0]
    S = np.diag(S)
    S = np.concatenate([S, np.zeros([m, n-m])], axis=1)
    # Summary

    X_ens_kf = X_ens_p.copy()

    plt.figure()
    plt.plot(S @ VT @ P_inv_sqrt @ ground_truth, '--')
    plt.plot(S @ X_ens_kf.mean(axis=1), '-')
    plt.legend(['S @ x\'', 'S @ x\' ens mean'])
    plt.title('In transformed space: 0')


    plt.figure()
    plt.plot(ground_truth, '--')
    plt.plot(P_sqrt @ VT.T @ X_ens_kf.mean(axis=1), '-')
    plt.legend(['x', 'x ens mean'])
    plt.title(f'In real space: 0; rmse: {rmse[0]}')

    iterations = 2000
    rmse = np.ones(iterations + 1)*np.nan
    error = (ground_truth
             - P_sqrt @ VT.T @ X_ens_kf.mean(axis=1)[:, None])
    rmse[0] = (error**2).mean()

    plt.figure()
    plt.plot(error)
    plt.title(f'Error: 0; rmse: {rmse[0]}')

    for ii in range(iterations):
        mu_p = X_ens_kf.mean(axis=1)[:, None]
        Y_obs = (Y_ens_p[:, ii][:, None]
                 + 0 * R_sqrt @ np.random.randn(y_size, ens_size))
        K = kalman_gain(X_ens)

        # K = (S.diagonal()/(1 + S.diagonal()**2))[:, None]
        X_ens_kf[:y_size] = mu_p[:y_size] + K * (Y_obs - S @ mu_p)
        error = (ground_truth
                 - P_sqrt @ VT.T @ X_ens_kf.mean(axis=1)[:, None])
        rmse[ii + 1] = (error**2).mean()
        if (ii + 1) % 100 == 0 or ii ==0 :
            plt.figure()
            plt.plot(S @ VT @ P_inv_sqrt @ ground_truth, '--')
            plt.plot(S @ X_ens_kf.mean(axis=1), '-')
            plt.scatter(np.arange(dimension/2), Y_ens_p[:, ii], marker='.')
            plt.legend(['S @ x\'', 'S @ x\' ens mean', 'y\''])
            plt.title(f'In transformed space: {ii + 1}')


            plt.figure()
            plt.plot(ground_truth, '--')
            plt.plot(P_sqrt @ VT.T @ X_ens_kf.mean(axis=1), '-')
            plt.scatter(np.arange(dimension)[::2], Y_ens[:, ii], marker='.')
            plt.legend(['x', 'x ens mean', 'y'])
            plt.title(f'In real space: {ii + 1}; rmse: {rmse[ii + 1]}')

            plt.figure()
            plt.plot(error)
            plt.title(f'Error: {ii + 1}; rmse: {rmse[ii + 1]}')


    # In[83]:


    plt.figure()
    plt.plot(rmse)
    plt.title('RMSE')
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')

    plt.show()

if __name__ == '__main__':
    main()
