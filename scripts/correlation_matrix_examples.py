import numpy as np
import scipy as sp
import scipy.special
import matplotlib.pyplot as plt


def correlation_matern(rho, rho0, nu=2.5):
    rho1 = np.sqrt(2 * nu) * np.abs(rho) / rho0
    zero_indices = np.where(rho1 == 0)[0]
    c = (rho1 ** nu 
         * sp.special.kv(nu, rho1) 
         / sp.special.gamma(nu) 
         / 2.0 ** (nu - 1))
    c[zero_indices] = 1.0
    return c


def correlation_exp(rho, rho0, dummy):
    c = np.exp(-np.abs(rho/rho0))
    return c


def correlation_sqd_exp(rho, rho0, dummy):
    c = np.exp(-(rho**2 / (2 * rho0**2)))
    return c


def make_correlation_matrix(rho, rho0, correlation, nu=None):
    Nx = rho.size
    cor_vec = correlation(rho, rho0, nu)                                                                                                                                                                                                                                                                        
    C = np.zeros([Nx, Nx])                                                                                                                                                                                                                                                                                          
    cor_vec = np.concatenate([cor_vec[:0:-1], cor_vec])                                                                                                                                                                                          
    for i in range(Nx):
        C[i] = cor_vec[Nx-1 - i:2 * Nx - 1 - i]
    return C


# # Check matern against exp

# In[6]:


rho = np.arange(10)
rho0 = 10
nu = .5


# In[7]:


print(nu)
correlation_matern(rho, rho0, nu)


# In[8]:


correlation_exp(rho, rho0, nu)


# # Check matern agains sqd exp

# In[9]:


rho = np.arange(10)
rho0 = 10
nu = 150


# In[10]:


print(nu)
correlation_matern(rho, rho0, nu)


# In[11]:


correlation_sqd_exp(rho, rho0, nu)


# # Correlation matrix for a ring

# In[12]:


n = 100
rho0 = 20
rho = np.arange(n, dtype=float)
rho = np.minimum(rho % n, (n - rho) % n)

P = make_correlation_matrix(rho, rho0, correlation_sqd_exp)

plt.figure()
im = plt.imshow(P)
plt.colorbar(im)
plt.title('Covariance matrix for a ring')


# In[13]:


D, E = sp.linalg.eigh(P)
D = D.clip(min=0)
D = D[::-1]
E = E[:, ::-1]


# In[14]:


plt.figure()
plt.semilogy(D)


# In[15]:


sqrtP = E @ np.diag(np.sqrt(D)) @ E.T


# In[16]:


plt.figure()
im = plt.imshow(sqrtP@sqrtP)
plt.colorbar(im)


# # Check ensemble est. of covariance

# In[17]:


n = 100
Ne = 100
rho0 = 0.1
Nrealizations = 200
NeigSubspace = 5
nu = 3/2
rho = np.arange(n)/n

P = make_correlation_matrix(rho, rho0, correlation_matern, nu=nu)

D, E = sp.linalg.eigh(P)
D = D.clip(min=0)
D = D[::-1]
E = E[:, ::-1]
sqrt_P = E @ np.sqrt(P) @ E.T
E_save = np.zeros([n, NeigSubspace, Nrealizations])
D_save = np.zeros([NeigSubspace, Nrealizations])


# In[18]:


for ii in range(Nrealizations):
    X = sqrt_P @ np.random.randn(n, Ne)
    X = X - np.repeat(X.mean(axis=1)[:, None], Ne, axis=1)
    X = X / np.sqrt(Ne - 1)
    P_hat = X @ X.T
    D_save[:, ii], E_save[:, :, ii] = sp.linalg.eigh(
        P_hat, eigvals=((n - NeigSubspace), n - 1))


# In[19]:


max_lambda = np.zeros([Nrealizations, Nrealizations])
min_lambda = max_lambda.copy()
sum_lambda = max_lambda.copy()
for ii in range(Nrealizations):
    for jj in range(Nrealizations):
        IPs = E_save[:, :, ii].T @ E_save[:, :, jj]
        Lambda = sp.linalg.svd(IPs, compute_uv=False)
        max_lambda[ii, jj] = Lambda.max()
        min_lambda[ii, jj] = Lambda.min()
        sum_lambda[ii, jj] = (Lambda**2).sum()


# In[20]:


this = max_lambda.mean()
plt.figure()
im = plt.imshow(max_lambda)
plt.colorbar(im)
plt.title(f'max lambda; mean: {this:0.4}')

this = min_lambda.mean()
plt.figure()
im = plt.imshow(min_lambda)
plt.colorbar(im)
plt.title(f'min lambda; mean: {this:0.4}')

this = sum_lambda.mean()
plt.figure()
im = plt.imshow(sum_lambda)
plt.colorbar(im)
plt.title(f'sum lambda; mean: {this:0.4}')

