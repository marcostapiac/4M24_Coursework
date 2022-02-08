from functools import lru_cache
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import norm
import matplotlib.cm as cm
import copy


def GaussianKernel(x, l):
    """ Generate Gaussian kernel matrix efficiently using scipy's distance matrix function"""
    D = distance_matrix(x, x)
    return np.exp(-pow(D, 2) / (2 * pow(l, 2)))


def subsample(N, factor, seed=None):
    assert factor >= 1, 'Subsampling factor must be greater than or equal to one.'
    N_sub = int(np.ceil(N / factor))
    if seed: np.random.seed(seed)
    idx = np.random.choice(N, size=N_sub, replace=False)  # Indexes of the randomly sampled points
    return idx


def get_G(N, idx):
    """Generate the observation matrix based on datapoint locations.
    Inputs:
        N - Length of vector to subsample
        idx - Indexes of subsampled coordinates
    Outputs:
        G - Observation matrix"""
    M = len(idx)
    G = np.zeros((M, N))
    for i in range(M):
        G[i, idx[i]] = 1
    return G


def probit(v):
    return np.array([0 if x <= 0 else 1 for x in v])


def predict_t(samples):
    # Assume samples are "v" from classification
    N = samples.shape[0]
    M = samples.shape[1]
    MC_sum = np.zeros(M)
    for i in range(N):
        MC_sum += norm.cdf(samples[i, :])
    return MC_sum / N

def calc_MC_counts(samples):
    N = samples.shape[0]
    M = samples.shape[1]  # Equal to dimension of sample
    MC_sum = np.zeros(M)
    for i in range(N):
        MC_sum += np.exp(samples[i, :])
    MC_estimate = MC_sum/N  # Inferred expected counts
    return MC_estimate

###--- Density functions ---###

@lru_cache(maxsize=1)
def log_prior_const(shape: Tuple[int, int], K_inverse_str: bytes, dtype: np.dtype) -> float:
    K_inverse = np.frombuffer(K_inverse_str, dtype=dtype).reshape(shape)
    return (-shape[0] / 2) * np.log(2 * np.pi) + (0.5) * np.linalg.slogdet(K_inverse)[1]


def log_prior(u, K_inverse, prior_const_term=None):
    if not prior_const_term:
        prior_const_term = log_prior_const(K_inverse.shape, K_inverse.tostring(), K_inverse.dtype)
    u = u.reshape((u.shape[0], 1))
    return prior_const_term - 0.5 * u.T @ (K_inverse @ u)


def log_continuous_likelihood(u, v, G):
    M = v.shape[0]
    const = -0.5 * M * np.log(2 * np.pi)
    vect = (v - np.matmul(G, u))
    return const - 0.5 * np.sum(vect**2)


def log_probit_likelihood(u, t, G):
    # TODO: Return log likelihood p(t|u)
    phi = norm.cdf(G @ u)
    return np.sum([(t[i] == 1) * np.log(phi[i]) + (t[i] == 0) * (np.log(1 - phi[i])) for i in range(G.shape[0])])


def log_poisson_likelihood(u, c, G):
    Gu = np.matmul(G, u)
    theta = np.exp(Gu)
    # Ignore factorial of counts ONLY if using likelihood for pcN since it is constant
    return np.sum(-theta) + np.matmul(c.T,Gu) # TODO: Return likelihood p(c|u)


def log_continuous_target(u, y, K_inverse, G, prior_const_term = None):
    return log_prior(u, K_inverse, prior_const_term) + log_continuous_likelihood(u, y, G)


def log_probit_target(u, t, K_inverse, G):
    return log_prior(u, K_inverse) + log_probit_likelihood(u, t, G)


def log_poisson_target(u, c, K_inverse, G):
    return log_prior(u, K_inverse) + log_poisson_likelihood(u, c, G)


###--- MCMC ---###

def grw(log_target, u0, data, K, G, n_iters, beta):
    """ Gaussian random walk Metropolis-Hastings MCMC method
        for sampling from pdf defined by log_target.
    Inputs:
        log_target - log-target density
        u0 - initial sample
        y - observed data
        K - prior covariance
        G - observation matrix
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    X = []
    acc = 0
    u_prev = u0
    N = K.shape[0]
    # Inverse computed before the for loop for speed
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    Kc_inverse = np.linalg.inv(Kc)
    K_inverse = Kc_inverse.T @ Kc_inverse

    prior_const_term = log_prior_const(K_inverse.shape, K_inverse.tostring(), K_inverse.dtype)

    lt_prev = log_target(u_prev, data, K_inverse, G, prior_const_term)

    for _ in (range(n_iters)):

        u_new = u_prev + beta* np.matmul(Kc, np.random.randn(N, ))

        lt_new = log_target(u_new, data, K_inverse, G, prior_const_term)

        log_alpha = min(0,
                        lt_new - lt_prev)
        log_u = np.log(np.random.random())

        # Accept/Reject
        accept = log_u <= log_alpha
        if accept:
            acc += 1
            X.append(u_new)
            u_prev = u_new
            lt_prev = lt_new
        else:
            X.append(u_prev)

    return np.array(X), acc / n_iters


def pcn(log_likelihood, u0, y, K, G, n_iters, beta):
    """ pCN MCMC method for sampling from pdf defined by log_prior and log_likelihood.
    Inputs:
        log_likelihood - log-likelihood function
        u0 - initial sample
        y - observed data
        K - prior covariance
        G - observation matrix
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    X = []
    acc = 0
    u_prev = u0
    N = u_prev.shape[0]
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    ll_prev = log_likelihood(u_prev, y, G)

    for _ in (range(n_iters)):
        # Generate sample from N(0, K)
        u_new = np.sqrt(1 - beta ** 2) * u_prev + beta * np.matmul(Kc, np.random.randn(N, ))
        ll_new = log_likelihood(u_new, y, G)
        log_alpha = min(ll_new - ll_prev, 0)
        log_u = np.log(np.random.random())

        # Accept/Reject
        accept = log_u <= log_alpha
        if accept:
            acc += 1
            X.append(u_new)
            u_prev = u_new
            ll_prev = ll_new
        else:
            X.append(u_prev)

    return np.array(X), acc / n_iters


###--- Plotting ---###

def plot_3D(u, x, y, title=None):
    """Plot the latent variable field u given the list of x,y coordinates"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_trisurf(x, y, u, cmap='viridis', linewidth=0, antialiased=False)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("u")
    cbar = fig.colorbar(surf, shrink=0.9)
    cbar.set_label("Latent variable value $u$", rotation=270, labelpad=14)
    if title:  plt.title(title)
    plt.show()


def plot_result(u, data, x, y, x_d, y_d, title=None, l=None):
    """Plot the latent variable field u with the observations,
        using the latent variable coordinate lists x,y and the
        data coordinate lists x_d, y_d"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_trisurf(x, y, u, cmap='viridis', linewidth=0, antialiased=False)
    cbar = fig.colorbar(surf, shrink=0.9)
    cbar.set_label("Latent variable value $u$", rotation=270, labelpad=14)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("u")
    ax.scatter(x_d, y_d, data, marker='x', color='r', label="Noisy Observations")
    ax.legend()
    if title:  plt.title(title)
    plt.show()


def plot_2D(counts, xi, yi, title=None, cbar_label=None, colors='viridis'):
    """Visualise count data given the index lists"""
    Z = -np.ones((max(yi) + 1, max(xi) + 1))
    for i in range(len(counts)):
        Z[(yi[i], xi[i])] = counts[i]
    my_cmap = copy.copy(cm.get_cmap(colors))
    my_cmap.set_under('k', alpha=0)
    fig, ax = plt.subplots()
    im = ax.imshow(Z, origin='lower', cmap=my_cmap, clim=[-0.1, np.max(counts)])
    cbar = fig.colorbar(im)
    cbar.set_label(cbar_label, rotation=270, labelpad=14)
    if title:  plt.title(title)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    plt.show()
