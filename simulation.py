import numpy as np
from scipy.stats import *
from functions import *

###--- Data Generation ---###
D = 16
### Inference grid defining {ui}, i=1:Dx*Dy
Dx = D
Dy = D
N = Dx * Dy  # Total number of coordinates
points = [(i, j) for j in np.arange(Dx) for i in np.arange(Dy)]  # Indexes for the inference grid
coords = [(x, y) for y in np.linspace(0, 1, Dy) for x in np.linspace(0, 1, Dx)]  # Coordinates for the inference grid
xi, yi = np.array([c[0] for c in points]), np.array([c[1] for c in points])  # Get x, y index lists
x, y = np.array([c[0] for c in coords]), np.array([c[1] for c in coords])  # Get x, y coordinate lists

### Data grid defining {vi}i=1,N/subsample_factor - subsampled from inference grid
subsample_factor = 4
idx = subsample(N, subsample_factor)  # Given N coordinates, obtain N/subsample_factor indices chosen randomly
M = len(idx)  # Total number of data points equal to np.ceil(N/subsample_factor)
print(M)
### Set MCMC parameters
n = 100000
beta = 0.2
### Set the likelihood and target, for sampling p(u|v)
log_target = log_continuous_target
log_likelihood = log_continuous_likelihood

l = 0.3
K = GaussianKernel(coords, l)
z = np.random.randn(N, )  # One for each latent variable
Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
u = Kc @ z
### Observation model: v = G(u) + e,   e~N(0,I)
G = get_G(N, idx)
v = G @ u + np.random.randn(M)  # M number of datapoints


### Generate K, the covariance of the Gaussian process, and sample from N(0,K) using a stable Cholesky decomposition
# TODO: Complete Simulation questions (a).
def varying_l():
    ls = [0.01, 0.1, 0.3, 1, 10, 20]
    for l in tqdm(ls):
        K = GaussianKernel(coords, l)
        z = np.random.randn(N, )  # One for each latent variable
        Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
        u = Kc @ z
        ### Observation model: v = G(u) + e,   e~N(0,I)
        G = get_G(N, idx)
        v = G @ u + np.random.randn(M)  # M number of datapoints

        ### Sample from prior for MCMC initialisation

        plot_3D(u, x, y, title="Gaussian Prior Sample Function, $l = " + str(l) + " $")  # Plot original u surface
        plot_result(u, v, x, y, x[idx], y[idx],
                    title="Gaussian Prior Sample Function, $l = " + str(l) + " $", l=l)  # Plot original u with data v


def exB():
    # Observations=data is v, and initial sample should be from
    l = 0.000000000000001
    K = GaussianKernel(coords, l)
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    U = Kc @ np.random.randn(N, )
    grw_samples, acc_prob_grw = grw(log_target, U, v, K, G, n_iters=n, beta=beta)
    grw_samples = np.mean(np.array(grw_samples), axis=0)

    pcn_samples, acc_prob_pcn = pcn(log_likelihood, U, v, K, G, n_iters=n, beta=beta)
    pcn_samples = np.mean(np.array(pcn_samples), axis=0)

    # plot_3D(u, x, y, title="Gaussian Prior Sample Function, $l = " + str(l) + " $")  # Plot original u surface
    # plot_result(u, v, x, y, x[idx], y[idx],
    #            title="Gaussian Prior Sample Function, $l = " + str(l) + " $", l=l)  # Plot original u with data v
    plot_3D(U, x, y,
            title="Gaussian Prior Sample Function, $l = " + str(l) + " $")  # Plot original u with data v
    plot_3D(grw_samples, x, y,
            title="GRW-MH Sample Mean Field, $l = " + str(l) + " $")  # Plot original u with data v
    plot_2D(np.abs(u - grw_samples), xi, yi, title="GRW-MH Error Field, $l = " + str(l) + " $",
            cbar_label="Absolute Error")  # Plot original u surface
    plot_result(pcn_samples, v, x, y, x[idx], y[idx],
                title="PCN Sample Mean Field, $l = " + str(l) + " $")  # Plot original u with data v
    plot_2D(np.abs(u - pcn_samples), xi, yi, title="PCN Error Field, $l = " + str(l) + " $",
            cbar_label="Absolute Error")  # Plot original u surface
    print("Absolute error of GRW-MH: {}".format(np.mean(np.abs(u - grw_samples))))
    print("Absolute error of PCN: {}".format(np.mean(np.abs(u - pcn_samples))))


exB()


def varying_niters():
    L = 5
    ns = [1000, 10000, 100000, 1000000, 10000000]
    grw_errors = np.zeros(L)
    pcn_errors = np.zeros(L)

    for i in tqdm(range(len(ns))):
        grw_samples, acc_prob_grw = grw(log_target, Kc @ np.random.randn(N, ), v, K, G, n_iters=ns[i], beta=beta)
        pcn_samples, acc_prob_pcn = pcn(log_likelihood, Kc @ np.random.randn(N, ), v, K, G, n_iters=ns[i], beta=beta)
        grw_samples = np.mean(np.array(grw_samples), axis=0)
        pcn_samples = np.mean(np.array(pcn_samples), axis=0)
        grw_errors[i] = np.mean(np.abs(u - grw_samples))
        pcn_errors[i] = np.mean(np.abs(u - pcn_samples))
    print(grw_errors)
    print(pcn_errors)
    plt.plot(ns, pcn_errors, label="pCN")
    plt.plot(ns, grw_errors, label="GRWMH")
    plt.xlabel("Run Time")
    plt.ylabel("Mean Absolute Error")
    plt.title("Mean Absolute Error for Different Run Times")
    plt.legend()
    plt.show()


def varying_stepsize():
    l = 0.3
    K = GaussianKernel(coords, l)
    z = np.random.randn(N, )  # One for each latent variable
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    u = Kc @ z
    ### Observation model: v = G(u) + e,   e~N(0,I)
    G = get_G(N, idx)
    v = G @ u + np.random.randn(M)  # M number of datapoints
    B = 100
    betas = np.linspace(0.0, 1.0, B)
    pcn_probs = np.zeros(B)
    grw_probs = np.zeros(B)
    for i in range(B):
        grw_samples, acc_prob_grw = grw(log_target, Kc @ np.random.randn(N, ), v, K, G, n_iters=n, beta=betas[i])
        pcn_samples, acc_prob_pcn = pcn(log_likelihood, Kc @ np.random.randn(N, ), v, K, G, n_iters=n, beta=betas[i])
        pcn_probs[i] = acc_prob_pcn
        grw_probs[i] = acc_prob_grw
    plt.plot(betas, pcn_probs, label="pCN")
    plt.plot(betas, grw_probs, label="GRWMH")
    plt.xlabel("Step Size")
    plt.ylabel("Acceptance Probability")
    plt.title("Acceptance Probabilities for Different Step Sizes")
    plt.legend()
    plt.show()


def varying_dim():
    B = 100
    Ds = np.linspace(8, 50, B, dtype=int)
    pcn_probs = np.zeros(B)
    grw_probs = np.zeros(B)
    for i in tqdm(range(B)):
        Dx = Ds[i]
        Dy = Ds[i]
        N = Dx * Dy  # Total number of coordinates
        points = [(i, j) for j in np.arange(Dx) for i in np.arange(Dy)]  # Indexes for the inference grid
        coords = [(x, y) for y in np.linspace(0, 1, Dy) for x in
                  np.linspace(0, 1, Dx)]  # Coordinates for the inference grid
        xi, yi = np.array([c[0] for c in points]), np.array([c[1] for c in points])  # Get x, y index lists
        x, y = np.array([c[0] for c in coords]), np.array([c[1] for c in coords])  # Get x, y coordinate lists

        ### Data grid defining {vi}i=1,N/subsample_factor - subsampled from inference grid
        subsample_factor = N / 64
        idx = subsample(N, subsample_factor)  # Given N coordinates, obtain N/subsample_factor indices chosen randomly
        M = len(idx)  # Total number of data points equal to np.ceil(N/subsample_factor)
        l = 0.3
        K = GaussianKernel(coords, l)
        z = np.random.randn(N, )  # One for each latent variable
        Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
        u = Kc @ z
        ### Observation model: v = G(u) + e,   e~N(0,I)
        G = get_G(N, idx)
        v = G @ u + np.random.randn(M)  # M number of datapoints
        grw_samples, acc_prob_grw = grw(log_target, Kc @ np.random.randn(N, ), v, K, G, n_iters=n, beta=beta)
        pcn_samples, acc_prob_pcn = pcn(log_likelihood, Kc @ np.random.randn(N, ), v, K, G, n_iters=n, beta=beta)
        pcn_probs[i] = acc_prob_pcn
        grw_probs[i] = acc_prob_grw
    plt.plot(Ds, pcn_probs, label="pCN")
    plt.plot(Ds, grw_probs, label="GRWMH")
    plt.xlabel("Dimensions")
    plt.ylabel("Acceptance Probability")
    plt.title("Acceptance Probabilities for Different Dimensions")
    plt.legend()
    plt.show()


def varying_initial_state():
    D = 16
    Dx = D
    Dy = D
    N = Dx * Dy  # Total number of coordinates
    points = [(i, j) for j in np.arange(Dx) for i in np.arange(Dy)]  # Indexes for the inference grid
    coords = [(x, y) for y in np.linspace(0, 1, Dy) for x in
              np.linspace(0, 1, Dx)]  # Coordinates for the inference grid
    xi, yi = np.array([c[0] for c in points]), np.array([c[1] for c in points])  # Get x, y index lists
    x, y = np.array([c[0] for c in coords]), np.array([c[1] for c in coords])  # Get x, y coordinate lists

    ### Data grid defining {vi}i=1,N/subsample_factor - subsampled from inference grid
    subsample_factor = 4
    idx = subsample(N, subsample_factor)  # Given N coordinates, obtain N/subsample_factor indices chosen randomly
    M = len(idx)  # Total number of data points equal to np.ceil(N/subsample_factor)
    l = 0.3
    n = 10000
    K = GaussianKernel(coords, l)
    z = np.random.randn(N, )  # One for each latent variable
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    u = Kc @ z
    ### Observation model: v = G(u) + e,   e~N(0,I)
    G = get_G(N, idx)
    v = G @ u + np.random.randn(M)  # M number of datapoints
    L = 1
    initial_states = np.array([Kc @ np.random.randn(N, ) for l in range(L)])
    pcn_sample_diff = np.zeros((L, n))
    grw_sample_diff = np.zeros((L, n))
    for i in tqdm(range(L)):
        # grw_sample, acc_prob_grw = grw(log_target, initial_states[i], v, K, G, n_iters=n, beta=beta)
        pcn_sample, acc_prob_pcn = pcn(log_likelihood, initial_states[i], v, K, G, n_iters=n, beta=beta)
        pcn_sample_diff[i] = np.array([np.mean(pcn_sample[j, :]) for j in range(n)])
        # grw_sample_diff[i] = np.array([np.linalg.norm(grw_sample[j]-grw_sample[j-1]) for j in range(n)])

    plt.plot(np.linspace(0, n, num=n), pcn_sample_diff[0, :], label="pCN")
    # plt.plot(np.linspace(n), grw_samples, label="GRWMH")
    plt.ylabel("Absolute Difference in Sample Magnitude")
    plt.xlabel("Iteration Number")
    plt.title("Convergence of pCN for different Initial States")
    plt.legend()
    plt.show()


def question_c():
    t = probit(v)  # Probit transform of data (M data points)
    true_class = probit(u)

    pcn_samples, _ = pcn(log_likelihood, Kc @ np.random.randn(N, ), v, K, G, n_iters=n, beta=beta)
    pcn_samples = np.array(pcn_samples)
    predictions = predict_t(pcn_samples)
    plot_2D(true_class, xi, yi, title='True Class Assignments',
            cbar_label="Class")  # Plot true class assignments (yellow = 1)
    plot_2D(predictions, xi, yi, title='Predictive Distribution',
            cbar_label="Probability")  # Plot predictive assignments
    plot_2D(t, xi[idx], yi[idx], title='Subsampled Assignments', cbar_label="Class")  # Plot subsampled data assignments
    hard_class = 1 * (predictions >= 0.5)
    abs_error = np.abs(hard_class - true_class)
    plot_2D(abs_error, xi, yi, title='Absolute Error $l= ' + str(l) + "$",
            cbar_label="Absolute Error")  # Plot true class assignments (yellow = 1)
    print(np.mean(abs_error))


def question_d():
    n = 100000
    ls = np.logspace(-2, 1, 100)
    ls = np.append(ls, 0.3)
    ls.sort()
    Ks = [GaussianKernel(coords, l) for l in ls]
    mean_pred_errors = np.zeros(101)
    true_class = probit(u)
    t = probit(v)
    for i in tqdm(range(len(ls))):
        Kc = np.linalg.cholesky(Ks[i] + 1e-6 * np.eye(N))
        pcn_samples, _ = pcn(log_likelihood, Kc @ np.random.randn(N, ), v, Ks[i], G, n_iters=n, beta=beta)
        predictions = predict_t(np.array(pcn_samples))
        hard_class = 1 * (predictions >= 0.5)
        abs_error = np.abs(hard_class - true_class)
        mean_pred_errors[i] = np.mean(abs_error)
        if ls[i] == 0.3:
            plot_2D(true_class, xi, yi, title='True Class Assignments $l= ' + str(ls[i]) + "$",
                    cbar_label="Class")  # Plot true class assignments (yellow = 1)
            plot_2D(hard_class, xi, yi,
                    title='Hard Classification based on Predictive Probabilities $l= ' + str(ls[i]) + "$",
                    cbar_label="Class")  # Plot predictive assignments
            plot_2D(t, xi[idx], yi[idx], title='Subsampled Assignments $l= ' + str(ls[i]) + "$",
                    cbar_label="Class")  # Plot subsampled data assignments
            plot_2D(abs_error, xi, yi, title='Absolute Error $l= ' + str(ls[i]) + "$",
                    cbar_label="Absolute Error")  # Plot true class assignments (yellow = 1)

    fig, ax = plt.subplots()
    id = np.argmin(mean_pred_errors)
    ax.set_xlabel("Length-Scale")
    ax.set_ylabel("Mean Prediction Error")
    ax.set_title("1-D Search Grid for Length-Scale Optimisation")
    ax.plot(ls[id], mean_pred_errors[id], 'g*', label="Optimal Length Scale: $l= " + str(round(ls[id], 3)) + "$")
    ax.plot(ls, mean_pred_errors)
    plt.legend()
    plt.show()
