import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import *

###--- Import spatial data ---###

### Read in the data
df = pd.read_csv('data.csv')

### Generate the arrays needed from the dataframe
data = np.array(df["bicycle.theft"])
xi = np.array(df['xi'])
yi = np.array(df['yi'])
N = len(data)
coords = [(xi[i],yi[i]) for i in range(N)]

### Subsample the original data set
subsample_factor = 1
idx = subsample(N, subsample_factor, seed=42)
G = get_G(N,idx)
c = G @ data
###--- MCMC ---####

### Set MCMC parameters
l = 100
n = 10000
beta = 0.2

# Assume latent variables arise from Gaussian process
K = GaussianKernel(coords, l)
N = K.shape[0]
Kc = np.linalg.cholesky(K + 1e-6*np.eye(N))

### Set the likelihood and target, for sampling p(u|c)
log_target = log_poisson_target
log_likelihood = log_poisson_likelihood

# The data is the sub-sampled counts
pcn_samples, acc_prob = pcn(log_likelihood, Kc @ np.random.randn(N,), c, K, G, n_iters=n, beta=beta)
pcn_samples = np.array(pcn_samples)
MC_counts = calc_MC_counts(pcn_samples)

plot_2D(MC_counts, xi, yi, title="MC estimate expected theft counts $l = " + str(l) + " $", cbar_label="Expected Counts")
plot_2D(data, xi, yi, title="True counts bike theft counts", cbar_label="True Counts")
plot_2D(c, xi[idx], yi[idx], title="Subsampled counts bike theft counts", cbar_label="True Counts")
plot_2D(np.abs(data-MC_counts), xi, yi, title="Error Field $l = " + str(l) + " $", cbar_label="Absolute Error in Counts")

ls= np.logspace(-2, 1,100)

Ks = [GaussianKernel(coords, l) for l in ls]
mean_pred_errors = np.zeros(100)

for i in tqdm(range(len(ls))):
    Kc = np.linalg.cholesky(Ks[i] + 1e-6 * np.eye(N))
    pcn_samples, _ = pcn(log_likelihood, Kc @ np.random.randn(N, ), c, Ks[i], G, n_iters=n, beta=beta)
    MC_counts = calc_MC_counts(pcn_samples)
    abs_error = np.abs(MC_counts - data)
    mean_pred_errors[i] = np.mean(abs_error)

fig, ax = plt.subplots()
id = np.argmin(mean_pred_errors)
ax.set_xlabel("Length-Scale")
ax.set_xscale("log")
ax.set_ylabel("Mean Prediction Error")
ax.set_title("1-D Search Grid for Length-Scale Optimisation")
print(mean_pred_errors[id])
ax.plot(ls[id], mean_pred_errors[id], 'g*', label="Optimal Length Scale: $l= " + str(round(ls[id],5))+"$")
ax.plot(ls, mean_pred_errors)
plt.legend()
plt.show()

fig, ax = plt.subplots()
ids = np.argwhere(ls>0.6)
id = np.argmin(mean_pred_errors[ids])
l = ls[ids][id][0]
Ks = GaussianKernel(coords, l)
Kc = np.linalg.cholesky(Ks + 1e-6 * np.eye(N))
pcn_samples, _ = pcn(log_likelihood, Kc @ np.random.randn(N, ), c, Ks, G, n_iters=n, beta=beta)
MC_counts = calc_MC_counts(pcn_samples)
plot_2D(MC_counts, xi, yi, title="MC estimate expected theft counts $l = " + str(round(l,5)) + " $", cbar_label="Expected Counts")
abs_error = np.abs(MC_counts - data)
plot_2D(np.abs(data-MC_counts), xi, yi, title="Error Field $l = " + str(round(l,5)) + " $", cbar_label="Absolute Error in Counts")
print(np.mean(abs_error))

### Plotting examples
plot_2D(data, xi, yi, title='Bike Theft Data') # Plot bike theft count data
plot_2D(c, xi[idx], yi[idx], title="Subsampled counts bike theft counts")
